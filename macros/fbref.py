"""Macros do FBref — leitura dos CSVs e adapter → schema canônico do projeto.

Reúne (em polars) tudo que é específico do FBref e usado por mais de uma tabela:

  • Leitor `read_fbref_csv()` dos CSVs do FBref, que possuem cabeçalho de 3 linhas
    (grupo / estatística / chave) — usado na camada flatfile.
  • Adapter `fbref_partidas / fbref_stats_jogador / fbref_escalacoes`, que
    converte os parquet brutos do FBref para EXATAMENTE o mesmo schema das tabelas
    bronze do StatsBomb (`bronze_partidas`, `bronze_escalacoes`,
    `bronze_stats_jogador`), para que a união seja trivial.

Limitações conhecidas do FBref (stat_type='summary' do soccerdata):
  • Não há `player_id` no FBref → usamos o id CANÔNICO por nome
    (`player_uid_expr`), IDÊNTICO ao do StatsBomb, de modo que o mesmo jogador
    receba o mesmo id nas duas fontes (identidade unificada).
  • Várias estatísticas do StatsBomb não existem no summary do FBref → ficam NULAS.
  • Cartões: o FBref não separa "segundo amarelo" de vermelho direto →
    `cards_second_yellow` fica 0 e `cards_expulsion` = vermelhos (CrdR).

Tudo aqui é função pura de DataFrame → DataFrame, testável offline.
"""
from __future__ import annotations

import polars as pl

from macros.features import (
    add_calendar_columns,
    player_uid_expr,
    position_group_from_fbref,
)


# ---------------------------------------------------------------------------
# Leitura dos CSVs do FBref (cabeçalho de 3 linhas)
# ---------------------------------------------------------------------------
def _build_fbref_names(group_row, stat_row, key_row) -> list[str]:
    """Combina as 3 linhas de cabeçalho do FBref em nomes de coluna únicos.

    Layout do FBref export:
      • linha 0 (group_row): categoria do grupo, ex. 'Playing Time', 'Performance'
      • linha 1 (stat_row):  nome da estatística, ex. 'MP', 'Gls', '90s'
      • linha 2 (key_row):   chaves de identificação nas 1ras colunas:
                             'league','season','team','player'
    Regra de nomeação por coluna:
      - se houver chave (key_row) → usa a chave (league/season/team/player);
      - senão, se houver estatística → 'Grupo_Estatística' (ex. 'Performance_Gls'),
        desambiguando 'Gls' que aparece em 'Performance' e em 'Per 90 Minutes';
      - senão usa só o grupo (ex. coluna 'Club').
    Sufixos numéricos são adicionados a nomes repetidos para garantir unicidade.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, (g, s, k) in enumerate(zip(group_row, stat_row, key_row)):
        g = (g or "").strip()
        s = (s or "").strip()
        k = (k or "").strip()
        if k:
            base = k
        elif s:
            base = f"{g}_{s}" if g and g != s else s
        elif g:
            base = g
        else:
            base = f"col{i}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


def read_fbref_csv(path: str) -> pl.DataFrame:
    """Lê um CSV do FBref, tolerante a DOIS formatos:

    • CRU (export do FBref/soccerdata): cabeçalho hierárquico de 3 linhas
      (grupo / estatística / chave) → resolvemos com `_build_fbref_names`.
    • LIMPO (cabeçalho único, ex.: arquivo já normalizado/regenerado) → lemos
      direto com `has_header=True`.

    Detecção: se a 1ª linha já contém a chave 'league', é o formato limpo.
    """
    head = pl.read_csv(path, has_header=False, infer_schema_length=0, n_rows=1)
    primeira_linha = [str(v) for v in head.row(0)]
    if "league" in primeira_linha:
        # Formato limpo (cabeçalho único).
        return pl.read_csv(path, infer_schema_length=0)

    # Formato cru com 3 linhas de cabeçalho.
    raw = pl.read_csv(path, has_header=False, infer_schema_length=0)
    names = _build_fbref_names(raw.row(0), raw.row(1), raw.row(2))
    data = raw.slice(3)
    rename_map = {data.columns[i]: names[i] for i in range(len(names))}
    return data.rename(rename_map)


# ---------------------------------------------------------------------------
# Mapeamento canônico ← candidatos de coluna no summary do FBref (flat).
# Tentamos cada candidato na ordem; o 1º presente vence. Sem candidato → nulo.
# (Nomes "Grupo_Stat" como o fetcher achata; incluímos fallback só pelo leaf.)
# ---------------------------------------------------------------------------
_STAT_CANDIDATES: dict[str, list[str]] = {
    "gols": ["Performance_Gls", "Gls"],
    "assistencias": ["Performance_Ast", "Ast"],
    "chutes_no_alvo": ["Performance_SoT", "SoT"],
    "ev_shot": ["Performance_Sh", "Sh"],
    "xg_total": ["Expected_xG", "xG"],
    "passes_completos": ["Passes_Cmp", "Cmp"],
    "ev_pass": ["Passes_Att", "Att"],
    "ev_interception": ["Performance_Int", "Int"],
    "ev_block": ["Performance_Blocks", "Blocks"],
    "ev_dribble": ["Take-Ons_Succ", "Dribbles_Succ", "Succ"],
    # As demais estatísticas do feature_config não existem no summary do FBref e
    # serão preenchidas com nulo automaticamente (ver `_stat_expr`).
}

_CARD_YELLOW_CANDIDATES = ["Performance_CrdY", "CrdY"]
_CARD_RED_CANDIDATES = ["Performance_CrdR", "CrdR"]


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _stat_expr(columns: list[str], canonical: str) -> pl.Expr:
    """Expressão polars para uma estatística canônica a partir do summary FBref.

    Usa o 1º candidato presente (cast Float64); se nenhum existir, devolve nulo
    tipado em Float64 (mantém o schema estável para a união).
    """
    src = _first_present(columns, _STAT_CANDIDATES.get(canonical, []))
    if src is None:
        return pl.lit(None, dtype=pl.Float64).alias(canonical)
    return pl.col(src).cast(pl.Float64, strict=False).alias(canonical)


def _player_id_expr(name_col: str = "player") -> pl.Expr:
    """ID canônico unificado por nome — IDÊNTICO ao usado no StatsBomb.

    Reusa `player_uid_expr` (mesma normalização NFKD/sem-acento/minúsculas), para
    que o MESMO jogador receba o MESMO id nas duas fontes. (Sem prefixo 'fbref:'.)
    """
    return player_uid_expr(name_col)


def _parse_score(score_col: str = "score"):
    """Extrai gols mandante/visitante do placar 'H–A' (traço en-dash do FBref)."""
    # Normaliza qualquer tipo de traço para '-' e extrai os dois inteiros.
    norm = pl.col(score_col).cast(pl.Utf8).str.replace_all("[–—−]", "-")
    home = norm.str.extract(r"^\s*(\d+)\s*-\s*\d+", 1).cast(pl.Int64, strict=False)
    away = norm.str.extract(r"^\s*\d+\s*-\s*(\d+)", 1).cast(pl.Int64, strict=False)
    return home, away


# ---------------------------------------------------------------------------
# 1) PARTIDAS  (schema de bronze_partidas)
# ---------------------------------------------------------------------------
def fbref_partidas(schedule: pl.DataFrame) -> pl.DataFrame:
    """schedule bruto do FBref → schema canônico de partidas."""
    if schedule.is_empty():
        return pl.DataFrame()

    home_score, away_score = _parse_score("score")
    df = schedule.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("match_date"),
        home_score.alias("home_score"),
        away_score.alias("away_score"),
        pl.col("game_id").cast(pl.Utf8).alias("match_id"),
        pl.col("competition_code").alias("competition_name"),
        pl.col("season").cast(pl.Utf8).alias("season"),
    )
    df = add_calendar_columns(df, date_col="match_date")
    return df.select(
        "match_id",
        "match_date",
        "year_month",
        "month_index",
        pl.lit(None, dtype=pl.Int64).alias("competition_id"),
        "competition_name",
        pl.lit(None, dtype=pl.Utf8).alias("season_id"),
        "season",
        pl.lit(None, dtype=pl.Int64).alias("home_team_id"),
        pl.col("home_team").cast(pl.Utf8),
        pl.lit(None, dtype=pl.Int64).alias("away_team_id"),
        pl.col("away_team").cast(pl.Utf8),
        "home_score",
        "away_score",
        pl.lit("fbref").alias("source"),
    )


# ---------------------------------------------------------------------------
# 2) STATS POR JOGADOR/PARTIDA  (schema de bronze_stats_jogador)
# ---------------------------------------------------------------------------
def fbref_stats_jogador(
    summary: pl.DataFrame,
    schedule: pl.DataFrame,
    lineup: pl.DataFrame,
    stat_cols: list[str],
) -> pl.DataFrame:
    """summary do FBref → schema canônico de stats por jogador/partida."""
    if summary.is_empty():
        return pl.DataFrame()

    cols = summary.columns

    # Contexto da partida (data/calendário/lado) a partir do schedule canônico.
    partidas = fbref_partidas(schedule).select(
        "match_id", "match_date", "year_month", "month_index",
        "competition_name", "season", "home_team",
    )

    # Titularidade + setor (de lineup), chaveada por (match_id, team, player).
    if lineup is not None and not lineup.is_empty() and "is_starter" in lineup.columns:
        pos_expr = (
            position_group_from_fbref("position")
            if "position" in lineup.columns
            else pl.lit(None, dtype=pl.Utf8).alias("position_group")
        )
        starters = lineup.with_columns(pos_expr).select(
            pl.col("game_id").cast(pl.Utf8).alias("match_id"),
            pl.col("team").cast(pl.Utf8),
            pl.col("player").cast(pl.Utf8).alias("player_name"),
            pl.col("is_starter").cast(pl.Boolean),
            "position_group",
        ).unique(subset=["match_id", "team", "player_name"])
    else:
        starters = None

    df = summary.with_columns(
        pl.col("game_id").cast(pl.Utf8).alias("match_id"),
        pl.col("team").cast(pl.Utf8),
        pl.col("player").cast(pl.Utf8).alias("player_name"),
        _player_id_expr("player"),
        *[_stat_expr(cols, s) for s in stat_cols],
    )

    df = df.join(partidas, on="match_id", how="inner").with_columns(
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.lit("home"))
        .otherwise(pl.lit("away"))
        .alias("side"),
    ).drop("home_team")

    if starters is not None:
        df = df.join(starters, on=["match_id", "team", "player_name"], how="left")
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("is_starter"),
            pl.lit(None, dtype=pl.Utf8).alias("position_group"),
        )

    df = df.with_columns(pl.lit("fbref").alias("source"))
    key_cols = [
        "match_id", "match_date", "year_month", "month_index",
        "competition_name", "season", "team", "side",
        "player_id", "player_name", "is_starter", "position_group", "source",
    ]
    return df.select(key_cols + stat_cols)


# ---------------------------------------------------------------------------
# 3) ESCALAÇÕES  (schema de bronze_escalacoes)
# ---------------------------------------------------------------------------
def fbref_escalacoes(
    lineup: pl.DataFrame,
    summary: pl.DataFrame,
) -> pl.DataFrame:
    """lineup + cartões do summary → schema canônico de escalações."""
    if lineup is None or lineup.is_empty():
        return pl.DataFrame()

    pos_expr = (
        position_group_from_fbref("position")
        if "position" in lineup.columns
        else pl.lit(None, dtype=pl.Utf8).alias("position_group")
    )
    base = lineup.with_columns(pos_expr).select(
        pl.col("game_id").cast(pl.Utf8).alias("match_id"),
        pl.col("team").cast(pl.Utf8),
        pl.col("player").cast(pl.Utf8).alias("player_name"),
        _player_id_expr("player"),
        pl.col("jersey_number").cast(pl.Int64, strict=False),
        pl.col("is_starter").cast(pl.Boolean) if "is_starter" in lineup.columns
        else pl.lit(None, dtype=pl.Boolean).alias("is_starter"),
        "position_group",
    )

    # Cartões vêm do summary (CrdY/CrdR), juntados por (match_id, team, player).
    if summary is not None and not summary.is_empty():
        scols = summary.columns
        y = _first_present(scols, _CARD_YELLOW_CANDIDATES)
        r = _first_present(scols, _CARD_RED_CANDIDATES)
        cards = summary.select(
            pl.col("game_id").cast(pl.Utf8).alias("match_id"),
            pl.col("team").cast(pl.Utf8),
            pl.col("player").cast(pl.Utf8).alias("player_name"),
            (pl.col(y).cast(pl.Int64, strict=False) if y else pl.lit(0)).alias("cards_yellow"),
            (pl.col(r).cast(pl.Int64, strict=False) if r else pl.lit(0)).alias("cards_red"),
        ).unique(subset=["match_id", "team", "player_name"])
        base = base.join(cards, on=["match_id", "team", "player_name"], how="left")
    else:
        base = base.with_columns(
            pl.lit(0).alias("cards_yellow"), pl.lit(0).alias("cards_red")
        )

    return base.with_columns(
        pl.col("cards_yellow").fill_null(0).cast(pl.Int64),
        pl.col("cards_red").fill_null(0).cast(pl.Int64),
        # FBref não distingue 2º amarelo; expulsão ≈ vermelhos (CrdR).
        pl.lit(0, dtype=pl.Int64).alias("cards_second_yellow"),
    ).with_columns(
        pl.col("cards_red").alias("cards_expulsion"),
    ).select(
        "match_id",
        "team",
        pl.lit(None, dtype=pl.Int64).alias("competition_id"),
        pl.lit(None, dtype=pl.Utf8).alias("season_id"),
        "player_id",
        "player_name",
        "jersey_number",
        "is_starter",
        "position_group",
        "cards_yellow",
        "cards_second_yellow",
        "cards_red",
        "cards_expulsion",
        pl.lit("fbref").alias("source"),
    )


__all__ = [
    "read_fbref_csv",
    "fbref_partidas",
    "fbref_stats_jogador",
    "fbref_escalacoes",
]
