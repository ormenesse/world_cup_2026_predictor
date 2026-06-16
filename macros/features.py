"""Macros compartilhados do pipeline de análise de futebol.

Este módulo concentra a lógica reutilizável entre as camadas (bronze, silver,
gold) — qualquer função usada por VÁRIAS tabelas mora aqui (em `macros/`), em vez
de ser duplicada nos jobs. Antes vivia em `etl/_features.py`; foi promovido a
macro do projeto na reestruturação para a arquitetura medallion.

Conteúdo:
  • Leitura do `configs/feature_config.yaml` (estatísticas e janela
    parametrizáveis) — `load_feature_config()` e getters.
  • Funções utilitárias de calendário (year_month / month_index) usadas para
    janelas temporais — `add_calendar_columns()`.
  • Identidade unificada de jogador por nome — `player_uid_expr()`.
  • Classificação de posição → setor (StatsBomb / FBref) — `position_group_*`.
  • Parsing dos campos textuais do StatsBomb (cartões e titularidade) —
    `card_count_expr()` e `is_starter_expr()`.
  • Construção das expressões de agregação das silvers — `silver_agg_expressions()`.
  • Cálculo da média/percentil móveis de N meses usados na gold —
    `rolling_window_means()` / `rolling_window_quantiles()`.
  • Contexto defensivo coletivo (gols/xG sofridos) — `team_match_conceded()`.

Tudo é escrito em polars (lazy/eager DataFrames), conforme exigido pelo projeto.

(A leitura dos CSVs de 3 cabeçalhos do FBref e o adapter FBref vivem em
`macros/fbref.py`; helpers específicos do dataset FIFA vivem em `macros/fifa.py`.)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml

# Raiz do projeto = .../football_analysis (pasta acima de macros/). Usada para
# localizar o YAML de configuração de forma independente do diretório de
# execução.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FEATURE_CONFIG_PATH = _PROJECT_ROOT / "configs" / "feature_config.yaml"


# ---------------------------------------------------------------------------
# Configuração parametrizável
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_feature_config() -> dict:
    """Carrega (e cacheia) `configs/feature_config.yaml`.

    O `lru_cache` garante que o arquivo seja lido uma única vez por processo,
    mesmo que vários jobs chamem esta função.
    """
    with open(_FEATURE_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_window_months() -> int:
    """Janela móvel (em meses) configurada para as médias históricas."""
    return int(load_feature_config().get("rolling_window_months", 18))


def get_player_stats() -> list[str]:
    """Lista de colunas de estatística (sb_stats_jogador) a agregar."""
    return list(load_feature_config().get("player_match_stats", []))


def get_aggregations() -> list[str]:
    """Funções de agregação a aplicar a cada estatística nas silvers."""
    return list(load_feature_config().get("aggregations", ["mean"]))


def get_percentiles() -> list[float]:
    """Quantis a calcular na gold (ex.: [0.75] = percentil 75)."""
    return [float(q) for q in load_feature_config().get("percentiles", [])]


def get_percentile_stats() -> list[str]:
    """Estatísticas (importantes) para as quais calcular percentis na gold."""
    return list(load_feature_config().get("percentile_stats", []))


def get_position_groups() -> list[str]:
    """Grupos de posição para agregação por setor (ex.: ['GK','DEF','MID','FWD'])."""
    return list(load_feature_config().get("position_groups", ["GK", "DEF", "MID", "FWD"]))


# ---------------------------------------------------------------------------
# Classificação de posição → grupo de setor (GK / DEF / MID / FWD)
# ---------------------------------------------------------------------------
def player_uid_expr(name_col: str = "player_name") -> pl.Expr:
    """ID CANÔNICO e UNIFICADO de jogador, derivado do nome (igual entre fontes).

    O StatsBomb usa id inteiro próprio e o FBref não expõe id — esquemas
    incompatíveis. Para UNIFICAR (o mesmo jogador = o mesmo id em qualquer fonte),
    usamos o único campo comum: o NOME, normalizado de forma idêntica:
      NFKD → remove acentos (\\p{M}) → minúsculas → só [a-z0-9 ] → espaços únicos.
    Ex.: "Alejandro Grimaldo García" → "alejandro grimaldo garcia".

    Limitação honesta: nomes grafados de forma diferente entre provedores
    (ex.: "Vini Jr." vs "Vinícius Júnior") não casam — é o melhor possível sem
    uma tabela de-para (crosswalk) oficial de ids.
    """
    return (
        pl.col(name_col)
        .cast(pl.Utf8)
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("player_id")
    )


def position_group_from_statsbomb(positions_col: str = "positions") -> pl.Expr:
    """Deriva o grupo de setor a partir do campo textual `positions` do StatsBomb.

    Usa a PRIMEIRA posição listada (a inicial do jogador no jogo). Os rótulos do
    StatsBomb seguem padrões claros, então classificamos por substring, NESTA
    ordem (a ordem importa: 'Wing Back' contém 'Back' → DEF, não FWD):
      1) 'Goalkeeper'            → GK
      2) contém 'Back'           → DEF  (Center/Left/Right/Wing Back)
      3) contém 'Midfield'       → MID  (Defensive/Central/Attacking Midfield)
      4) caso contrário          → FWD  (Wing, Forward, Striker)
    Sem posição → nulo.
    """
    primary = pl.col(positions_col).str.extract(r"'position': '([^']+)'", 1)
    return (
        pl.when(primary.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(primary.str.contains("Goalkeeper"))
        .then(pl.lit("GK"))
        .when(primary.str.contains("Back"))
        .then(pl.lit("DEF"))
        .when(primary.str.contains("Midfield"))
        .then(pl.lit("MID"))
        .otherwise(pl.lit("FWD"))
        .alias("position_group")
    )


def position_group_from_fbref(pos_col: str = "position") -> pl.Expr:
    """Deriva o grupo de setor a partir da posição abreviada do FBref.

    O FBref usa siglas (ex.: 'GK','CB','LB','RB','WB','DM','CM','AM','MF','DF',
    'FW','LW','RW'), às vezes combinadas ('DF,MF'). Olhamos a 1ª sigla:
      GK→GK; D*/CB/LB/RB/WB→DEF; M*/DM/CM/AM→MID; F*/LW/RW/ST→FWD.
    """
    first = pl.col(pos_col).cast(pl.Utf8).str.strip_chars().str.split(",").list.first()
    return (
        pl.when(first.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(first.str.starts_with("GK"))
        .then(pl.lit("GK"))
        .when(first.str.starts_with("D") | first.str.contains("B$"))
        .then(pl.lit("DEF"))
        .when(first.str.starts_with("M") | first.str.contains("M$"))
        .then(pl.lit("MID"))
        .otherwise(pl.lit("FWD"))
        .alias("position_group")
    )


# ---------------------------------------------------------------------------
# Contexto defensivo coletivo: gols/xG SOFRIDOS por time em cada partida
# ---------------------------------------------------------------------------
def team_match_conceded(stats: pl.DataFrame, partidas: pl.DataFrame) -> pl.DataFrame:
    """Calcula gols/xG SOFRIDOS por time em cada partida.

    Como defender é coletivo, a "contribuição" de um jogador é melhor medida pelo
    que o TIME sofre quando ele joga. Esta função produz, por (partida, time):
      • `gols_sofridos` = gols do adversário (do placar em `partidas`);
      • `xg_sofrido`    = xG do time ADVERSÁRIO (soma do xG dos jogadores dele).

    Retorno (long): match_id, team, side, year_month, month_index,
                    gols_sofridos, xg_sofrido.
    """
    # xG do próprio time por partida (uma linha por time-partida).
    team_xg = stats.group_by(
        ["match_id", "side", "team", "year_month", "month_index"]
    ).agg(pl.col("xg_total").sum().alias("team_xg"))

    # xG do adversário = xG do outro lado da MESMA partida.
    opp_xg = (
        team_xg.select(
            "match_id",
            pl.col("side").alias("opp_side"),
            pl.col("team_xg").alias("xg_sofrido"),
        )
    )
    # Junta cada lado ao xG do lado oposto.
    out = team_xg.join(
        opp_xg,
        left_on="match_id",
        right_on="match_id",
        how="inner",
    ).filter(pl.col("side") != pl.col("opp_side")).drop("opp_side", "team_xg")

    # Gols sofridos vêm do placar: lado home sofre away_score, e vice-versa.
    placar = partidas.select(
        "match_id",
        pl.col("home_score").cast(pl.Int64),
        pl.col("away_score").cast(pl.Int64),
    )
    out = out.join(placar, on="match_id", how="left").with_columns(
        pl.when(pl.col("side") == "home")
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .cast(pl.Float64)
        .alias("gols_sofridos"),
    ).drop("home_score", "away_score")

    return out.select(
        "match_id", "team", "side", "year_month", "month_index",
        "gols_sofridos", "xg_sofrido",
    )


# ---------------------------------------------------------------------------
# Calendário: year_month (YYYYMM) e month_index (inteiro contínuo de meses)
# ---------------------------------------------------------------------------
def add_calendar_columns(df: pl.DataFrame, date_col: str = "match_date") -> pl.DataFrame:
    """Adiciona `year_month` e `month_index` a partir de uma coluna de data.

    • `year_month` (Int64, formato YYYYMM, ex.: 202404) — é a coluna incremental
      do projeto (ver etl_config.yaml) e a chave temporal das silvers.
    • `month_index` (Int64) — número contínuo de meses desde o ano 0
      (`ano*12 + mês - 1`). Serve para aritmética de janela: a diferença entre
      dois `month_index` é exatamente a distância em meses, o que torna trivial
      filtrar "os últimos 18 meses".

    Espera `date_col` como `pl.Date` (faça o parse antes de chamar).
    """
    return (
        df.with_columns(
            pl.col(date_col).dt.year().alias("_yr"),
            pl.col(date_col).dt.month().alias("_mo"),
        )
        .with_columns(
            (pl.col("_yr") * 100 + pl.col("_mo")).cast(pl.Int64).alias("year_month"),
            (pl.col("_yr") * 12 + pl.col("_mo") - 1).cast(pl.Int64).alias("month_index"),
        )
        .drop("_yr", "_mo")
    )


def year_month_to_month_index(year_month_col: str = "year_month") -> pl.Expr:
    """Reconstrói `month_index` a partir de `year_month` (YYYYMM).

    Útil nas silvers, que persistem `year_month` mas não `month_index`.
    `month_index = (YYYYMM // 100) * 12 + (YYYYMM % 100) - 1`.
    """
    ym = pl.col(year_month_col)
    return ((ym // 100) * 12 + (ym % 100) - 1).cast(pl.Int64)


# ---------------------------------------------------------------------------
# Parsing dos campos textuais do StatsBomb (sb_escalacoes)
# ---------------------------------------------------------------------------
def card_count_expr(card_type: str, source_col: str = "cards") -> pl.Expr:
    """Conta cartões de um tipo dentro do campo textual `cards`.

    O campo `cards` é o `repr` de uma lista de dicionários, ex.:
        [{'time': '36:20', 'card_type': 'Yellow Card', ...}]
    Em vez de fazer `ast.literal_eval` linha a linha (caro em ~146k linhas),
    contamos ocorrências do padrão `'card_type': '<tipo>'` direto na string —
    rápido e vetorizado em polars.

    Tipos presentes na base: 'Yellow Card', 'Second Yellow', 'Red Card'.
    """
    needle = f"'card_type': '{card_type}'"
    return pl.col(source_col).fill_null("").str.count_matches(needle, literal=True)


def is_starter_expr(positions_col: str = "positions") -> pl.Expr:
    """Marca titulares (Starting XI) a partir do campo textual `positions`.

    `positions` é o `repr` de uma lista de posições ocupadas no jogo; quem começa
    a partida tem `'start_reason': 'Starting XI'` na primeira posição. Detectamos
    a titularidade pela presença dessa marca na string (vetorizado).
    """
    return pl.col(positions_col).fill_null("").str.contains("Starting XI", literal=True)


# ---------------------------------------------------------------------------
# Agregação das silvers (parametrizável)
# ---------------------------------------------------------------------------
def silver_agg_expressions(
    stats: Iterable[str],
    aggregations: Iterable[str],
) -> list[pl.Expr]:
    """Constrói as expressões de agregação para as silvers (jogador/time × mês).

    Para cada estatística e cada agregação configurada gera `{stat}_{agg}`
    (ex.: `gols_mean`, `gols_sum`). Além disso, SEMPRE garante:
      • `{stat}_sum`   — necessário para a média ponderada da janela na gold;
      • `n_partidas`   — número de partidas no grupo (denominador da média).

    Assim, mesmo que o usuário remova "sum"/"mean" do YAML, a gold continua
    funcionando.
    """
    stats = list(stats)
    aggregations = list(aggregations)
    exprs: list[pl.Expr] = []
    emitted: set[str] = set()

    for stat in stats:
        for agg in aggregations:
            alias = f"{stat}_{agg}"
            exprs.append(getattr(pl.col(stat), agg)().alias(alias))
            emitted.add(alias)
        # Garante a soma (usada pela média móvel ponderada da gold).
        sum_alias = f"{stat}_sum"
        if sum_alias not in emitted:
            exprs.append(pl.col(stat).sum().alias(sum_alias))
            emitted.add(sum_alias)

    # Contagem de partidas do grupo (denominador da média móvel).
    exprs.append(pl.len().alias("n_partidas"))
    return exprs


# ---------------------------------------------------------------------------
# Média móvel ponderada de N meses (coração da gold)
# ---------------------------------------------------------------------------
def rolling_window_means(
    monthly: pl.DataFrame,
    requests: pl.DataFrame,
    *,
    entity_keys: list[str],
    row_keys: list[str],
    stats: list[str],
    window_months: int,
    prefix: str,
) -> pl.DataFrame:
    """Média móvel ponderada das estatísticas nos `window_months` meses anteriores.

    Esta é a operação central que transforma agregados mensais em "forma
    recente" no instante de cada partida.

    Parâmetros
    ----------
    monthly : DataFrame
        Agregado mensal por entidade (jogador ou time). Deve conter:
        `entity_keys`, `month_index`, `{stat}_sum` (p/ cada stat) e `n_partidas`.
    requests : DataFrame
        Uma linha por "consulta" (ex.: cada lado de cada partida, ou cada
        jogador titular de cada partida). Deve conter `row_keys`, `entity_keys`
        e `ref_month_index` (o month_index da partida de referência).
    entity_keys : list[str]
        Chave(s) que ligam `requests` a `monthly` (ex.: ["team"] ou ["player_id"]).
    row_keys : list[str]
        Chave(s) que identificam unicamente cada linha de `requests`
        (ex.: ["match_id","side"] ou ["match_id","side","player_id"]).
    stats : list[str]
        Estatísticas a calcular (devem existir como `{stat}_sum` em `monthly`).
    window_months : int
        Tamanho da janela (ex.: 18).
    prefix : str
        Prefixo das colunas de saída (ex.: "home_team_form_").

    Retorno
    -------
    DataFrame com `row_keys` + `{prefix}{stat}_mean_{N}m` + `{prefix}n_partidas_{N}m`.
    Linhas sem histórico na janela recebem nulos (não são descartadas).

    Por que média PONDERADA?
    ------------------------
    A média verdadeira das últimas N partidas/meses é Σ(soma_mensal) /
    Σ(partidas_mensais). Tirar a média das médias mensais daria peso igual a
    meses com 1 ou 8 jogos — incorreto. Por isso somamos `{stat}_sum` e
    `n_partidas` na janela e dividimos no fim.

    A janela é [ref_month_index - N, ref_month_index - 1]: estritamente ANTES do
    mês da partida, evitando vazamento de informação (data leakage) do próprio
    jogo ou de jogos futuros.
    """
    # 1) Liga cada consulta a todo o histórico mensal da mesma entidade.
    joined = requests.join(monthly, on=entity_keys, how="left")

    # 2) Mantém apenas os meses dentro da janela anterior à partida.
    joined = joined.filter(
        pl.col("month_index").is_not_null()
        & (pl.col("month_index") >= pl.col("ref_month_index") - window_months)
        & (pl.col("month_index") <= pl.col("ref_month_index") - 1)
    )

    # 3) Soma, por linha de consulta, as somas mensais e a contagem de partidas.
    agg_exprs = [pl.col(f"{s}_sum").sum().alias(f"{s}__win_sum") for s in stats]
    agg_exprs.append(pl.col("n_partidas").sum().alias("__win_n"))
    grouped = joined.group_by(row_keys).agg(agg_exprs)

    # 4) Média ponderada = soma da janela / nº de partidas da janela.
    mean_cols = [
        (pl.col(f"{s}__win_sum") / pl.col("__win_n")).alias(f"{prefix}{s}_mean_{window_months}m")
        for s in stats
    ]
    grouped = grouped.with_columns(
        mean_cols + [pl.col("__win_n").alias(f"{prefix}n_partidas_{window_months}m")]
    )
    out_cols = (
        row_keys
        + [f"{prefix}{s}_mean_{window_months}m" for s in stats]
        + [f"{prefix}n_partidas_{window_months}m"]
    )
    grouped = grouped.select(out_cols)

    # 5) Re-liga a TODAS as consultas (left) para preservar partidas sem
    #    histórico na janela — elas ficam com nulos em vez de sumir.
    base = requests.select(row_keys).unique()
    return base.join(grouped, on=row_keys, how="left")


def quantile_col_name(prefix: str, stat: str, quantile: float, window_months: int) -> str:
    """Nome padronizado de coluna de percentil, ex.: 'home_team_form_gols_p75_18m'."""
    return f"{prefix}{stat}_p{int(round(quantile * 100))}_{window_months}m"


def rolling_window_quantiles(
    events: pl.DataFrame,
    requests: pl.DataFrame,
    *,
    entity_keys: list[str],
    row_keys: list[str],
    stats: list[str],
    quantiles: list[float],
    window_months: int,
    prefix: str,
) -> pl.DataFrame:
    """Percentis das estatísticas nos `window_months` meses anteriores à partida.

    Ao contrário de `rolling_window_means`, que usa agregados MENSAIS (soma +
    contagem), o percentil exige a distribuição PARTIDA A PARTIDA. Por isso esta
    função recebe `events` em granularidade de evento/partida (uma linha por
    partida da entidade), não os agregados das silvers.

    Parâmetros
    ----------
    events : DataFrame
        Valores POR PARTIDA da entidade: `entity_keys` + `month_index` + as
        colunas de `stats` (uma linha por partida).
    requests : DataFrame
        Uma linha por consulta, com `row_keys`, `entity_keys` e
        `ref_month_index` (mês da partida de referência).
    quantiles : list[float]
        Quantis (ex.: [0.75]).
    prefix : str
        Prefixo das colunas de saída.

    Retorno
    -------
    DataFrame com `row_keys` + uma coluna por (stat, quantil):
    `{prefix}{stat}_p{q}_{N}m`. Mesma janela e proteção contra leakage de
    `rolling_window_means`: [ref_month_index - N, ref_month_index - 1].
    """
    joined = requests.join(events, on=entity_keys, how="left")
    joined = joined.filter(
        pl.col("month_index").is_not_null()
        & (pl.col("month_index") >= pl.col("ref_month_index") - window_months)
        & (pl.col("month_index") <= pl.col("ref_month_index") - 1)
    )

    agg_exprs = [
        pl.col(s).quantile(q, interpolation="linear").alias(quantile_col_name(prefix, s, q, window_months))
        for s in stats
        for q in quantiles
    ]
    grouped = joined.group_by(row_keys).agg(agg_exprs)

    base = requests.select(row_keys).unique()
    return base.join(grouped, on=row_keys, how="left")


__all__ = [
    "load_feature_config",
    "get_window_months",
    "get_player_stats",
    "get_aggregations",
    "get_percentiles",
    "get_percentile_stats",
    "get_position_groups",
    "add_calendar_columns",
    "year_month_to_month_index",
    "card_count_expr",
    "is_starter_expr",
    "silver_agg_expressions",
    "rolling_window_means",
    "quantile_col_name",
    "rolling_window_quantiles",
    "player_uid_expr",
    "position_group_from_statsbomb",
    "position_group_from_fbref",
    "team_match_conceded",
]
