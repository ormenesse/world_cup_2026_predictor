"""Macros do dataset FIFA × partidas (usados por VÁRIAS tabelas FIFA).

Concentra a lógica compartilhada entre as camadas FIFA (bronze/silver/gold) que
cruzam os RESULTADOS de partida (`schedule`) com os ratings de jogador do FIFA
(`fifa_aggregated.csv`):

  • Constantes (atributos do jogador, nº de titulares, projeção do flatfile).
  • Classificação posição → setor (GK/DEF/MID/FWD) — expr polars e função python.
  • Normalização de nome de time (FBref/football-data ↔ FIFA) e casamento fuzzy
    (`normalize_team_name`, `build_team_matcher`) — usados na resolução de nomes
    da gold (clubes e seleções).
  • Aliases de nationality (StatsBomb/FBref → FIFA).
  • Seleção do XI titular em polars (clube/seleção oficial e fallback de seleção).
  • Leitura do `configs/fifa_match_sources.yaml` (pool de league_id por competição,
    data mínima).

Regra de "dados mais recentes do time": para cada partida usamos o SNAPSHOT do
FIFA (uma `fifa_update_date`) mais recente com data <= data do jogo (as-of join),
refletindo elenco/ratings vigentes na época — sem usar dados do futuro.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import polars as pl
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIFA_SOURCES_PATH = _PROJECT_ROOT / "configs" / "fifa_match_sources.yaml"

# Atributos do jogador atrelados a cada slot do XI e aos agregados do time.
FIFA_PLAYER_ATTRS = [
    "overall", "potential", "value_eur", "age",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
]
N_STARTERS = 11

# Colunas (cruas) que o flatfile do FIFA precisa projetar do CSV gigante.
FIFA_FLATFILE_COLUMNS = [
    "short_name", "fifa_version", "fifa_update_date", "league_id", "club_name",
    "club_position", "nationality", "nation_position", "positions",
    *FIFA_PLAYER_ATTRS,
]

# Mapa posição → setor (para ordenar o XI e agregar por setor).
_DEF = {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"}
_MID = {"CDM", "LDM", "RDM", "CM", "LCM", "RCM", "LM", "RM", "CAM", "LAM", "RAM"}
_FWD = {"ST", "LS", "RS", "CF", "LF", "RF", "LW", "RW"}
SECTOR_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


# ---------------------------------------------------------------------------
# Posição → setor
# ---------------------------------------------------------------------------
def sector_of(pos) -> str:
    """Setor (GK/DEF/MID/FWD) de uma posição do FIFA. Fallback MID p/ atípicas."""
    p = (str(pos) if pos is not None else "").upper()
    if p == "GK":
        return "GK"
    if p in _DEF:
        return "DEF"
    if p in _MID:
        return "MID"
    if p in _FWD:
        return "FWD"
    return "MID"


def sector_expr(pos_col: str) -> pl.Expr:
    """Versão polars de `sector_of` (vetorizada), produzindo a coluna `sector`."""
    p = pl.col(pos_col).cast(pl.Utf8).str.to_uppercase()
    return (
        pl.when(p == "GK").then(pl.lit("GK"))
        .when(p.is_in(list(_DEF))).then(pl.lit("DEF"))
        .when(p.is_in(list(_MID))).then(pl.lit("MID"))
        .when(p.is_in(list(_FWD))).then(pl.lit("FWD"))
        .otherwise(pl.lit("MID"))
        .alias("sector")
    )


def sector_order_expr(sector_col: str = "sector") -> pl.Expr:
    """Ordem numérica do setor (GK<DEF<MID<FWD) para ordenar o XI."""
    s = pl.col(sector_col)
    return (
        pl.when(s == "GK").then(0)
        .when(s == "DEF").then(1)
        .when(s == "MID").then(2)
        .otherwise(3)
        .alias("sector_order")
    )


# ---------------------------------------------------------------------------
# Normalização de nome de time (FBref/football-data ↔ FIFA)
# ---------------------------------------------------------------------------
# Só sufixos/prefixos corporativos GENÉRICOS (que nunca diferenciam clubes).
# NÃO removemos "real/atletico/athletic/sporting/..." pois são distintivos.
_STOPWORDS = {
    "fc", "cf", "sc", "ssc", "ssd", "ud", "cd", "rcd", "afc", "cfc", "bk", "if",
    "fk", "sk", "ec", "ce", "aa", "se", "ca", "cs", "club", "calcio", "futebol",
    "futbol", "clube", "the", "vfl", "vfb", "tsg", "tsv", "sv", "fsv", "bsc",
    "1", "04", "05", "1846", "1860", "1899", "1900", "1909",
}

# Sufixos de Unidade Federativa do Brasil (football-data usa "Flamengo RJ" etc.).
_UF_SUFFIX = re.compile(
    r"\s(rj|sp|mg|sc|rs|pr|ba|pe|ce|go|pa|am|rn|pb|al|mt|ms|df|es|to|se|ac|ap|rr|ma|pi)$"
)

# Apelidos de CLUBE (football-data.co.uk → FIFA) que a normalização não resolve.
# Aplicados ao nome JÁ normalizado da partida, ANTES do casamento (ver gold).
CLUB_ALIASES = {
    "ath bilbao": "athletic club",
    "m gladbach": "borussia monchengladbach",   # football-data: "M'gladbach"
    "mgladbach": "borussia monchengladbach",
    "fc koln": "koln",
    "sp gijon": "sporting gijon",
}

# StatsBomb/FBref → nationality do FIFA (casos que normalização não resolve).
NATION_ALIASES = {
    "south korea": "korea republic", "korea": "korea republic",
    "north korea": "korea dpr", "iran": "iran", "ir iran": "iran",
    "china": "china pr", "china pr": "china pr", "usa": "united states",
    "united states": "united states", "ivory coast": "cote divoire",
    "czechia": "czech republic", "cape verde": "cabo verde",
    "drc": "congo dr", "dr congo": "congo dr",
}


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_team_name(name) -> str:
    """Normaliza um nome de time: sem acento/pontuação + remove stopwords genéricas.

    Se a remoção de stopwords zerar o nome (ex.: "Athletic Club" → tudo stopword?),
    mantém os tokens originais.
    """
    if name is None:
        return ""
    try:
        if isinstance(name, float) and name != name:  # NaN
            return ""
    except TypeError:
        pass
    s = strip_accents(str(name)).lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    s = " ".join(s.split())
    s = _UF_SUFFIX.sub("", s)  # remove sufixo de UF (ex.: "flamengo rj" → "flamengo")
    toks = [t for t in s.split() if t]
    kept = [t for t in toks if t not in _STOPWORDS]
    return " ".join(kept if kept else toks).strip()


def build_team_matcher(norm_keys: list[str]):
    """query→key: exato → fuzzy (difflib) → contenção/token-overlap de tokens.

    A contenção (um conjunto de tokens ⊆ o outro) resolve "Inter Milan"→"inter",
    "Hoffenheim"→"tsg 1899 hoffenheim". O overlap por token longo (≥5) resolve
    "Hertha Berlin"→"hertha bsc" sem casar erroneamente "Real Madrid"↔"Real Betis"
    (o token "real" tem 4 e não dispara overlap).
    """
    norm_keys = list(norm_keys)
    keyset = set(norm_keys)
    key_tokens = {k: set(k.split()) for k in norm_keys}

    def match(q: str):
        if not q:
            return None
        if q in keyset:
            return q
        near = difflib.get_close_matches(q, norm_keys, n=1, cutoff=0.82)
        if near:
            return near[0]
        qt = set(q.split())
        best, best_score = None, ()
        for k, kt in key_tokens.items():
            shared = qt & kt
            if not shared:
                continue
            contained = qt <= kt or kt <= qt
            long_shared = [t for t in shared if len(t) >= 5]
            ok_contain = contained and max((len(t) for t in shared), default=0) >= 4
            if not (ok_contain or long_shared):
                continue
            ratio = difflib.SequenceMatcher(None, q, k).ratio()
            score = (1 if ok_contain else 0,
                     max((len(t) for t in long_shared), default=0),
                     len(shared), ratio)
            if score > best_score:
                best, best_score = k, score
        return best

    return match


# ---------------------------------------------------------------------------
# Seleção do XI titular (polars, vetorizado por grupo)
# ---------------------------------------------------------------------------
def _rank_within_group(df: pl.DataFrame, group_keys: list[str]) -> pl.Expr:
    """0-based row number dentro de cada grupo, seguindo a ordem ATUAL do frame.

    O frame deve ter sido ordenado por `group_keys + [critério]` antes.
    """
    return pl.int_range(pl.len()).over(group_keys).alias("_rk")


def starting_xi_long(
    players: pl.DataFrame,
    *,
    pos_col: str,
    group_keys: list[str],
) -> pl.DataFrame:
    """XI titular "real" de cada grupo: posições válidas (sem SUB/RES), ordenado
    por setor (GK→DEF→MID→FWD) e overall desc, no máximo 11, com `slot` 1..11.

    Usado para o XI de CLUBE (pos_col='club_position') e o XI OFICIAL de seleção
    (pos_col='nation_position'). Devolve formato longo (1 linha por jogador).
    """
    g = players.filter(
        pl.col(pos_col).is_not_null()
        & (~pl.col(pos_col).cast(pl.Utf8).str.to_uppercase().is_in(["SUB", "RES"]))
        & pl.col("overall").is_not_null()
    )
    if g.is_empty():
        return g.with_columns(pl.lit(1).alias("slot")).head(0)

    g = g.with_columns(
        sector_expr(pos_col),
        pl.col(pos_col).cast(pl.Utf8).alias("position"),
    ).with_columns(sector_order_expr("sector"))

    g = g.sort(
        group_keys + ["sector_order", "overall"],
        descending=[False] * len(group_keys) + [False, True],
    )
    g = g.with_columns(_rank_within_group(g, group_keys))
    return (
        g.filter(pl.col("_rk") < N_STARTERS)
        .with_columns((pl.col("_rk") + 1).alias("slot"))
        .drop("_rk", "sector_order")
    )


def nation_fallback_xi_long(
    players: pl.DataFrame,
    *,
    group_keys: list[str],
) -> pl.DataFrame:
    """XI sintético de seleção: melhor GK + os melhores restantes por overall.

    Usado quando o FIFA não traz escalação OFICIAL (`nation_position`) para aquela
    nationality/snapshot — comum, pois o FIFA licencia só parte das seleções. A
    posição efetiva vem de `club_position`; se SUB/RES/ausente, cai para a 1ª de
    `positions`. Devolve formato longo (1 linha por jogador, `slot` 1..11).
    """
    g = players.filter(pl.col("overall").is_not_null())
    if g.is_empty():
        return g.with_columns(pl.lit(1).alias("slot")).head(0)

    cp = pl.col("club_position").cast(pl.Utf8).str.to_uppercase()
    eff = (
        pl.when(cp.is_not_null() & (~cp.is_in(["SUB", "RES", "NAN"])))
        .then(cp)
        .otherwise(
            pl.col("positions").cast(pl.Utf8).str.split(",").list.first()
            .str.strip_chars().str.to_uppercase()
        )
        .alias("eff_position")
    )
    g = g.with_columns(eff).with_columns(
        pl.col("eff_position").alias("position"),
    ).with_columns(
        sector_expr("eff_position"),
    ).with_columns(sector_order_expr("sector"))

    # Ordena por overall desc dentro do grupo p/ ranquear.
    g = g.sort(group_keys + ["overall"], descending=[False] * len(group_keys) + [True])
    # rank de TODOS por overall (0-based) e rank só entre GKs.
    g = g.with_columns(pl.int_range(pl.len()).over(group_keys).alias("_ovr_rk"))
    g = g.with_columns(
        pl.when(pl.col("sector") == "GK")
        .then(pl.int_range(pl.len()).over(group_keys + ["sector"]))
        .otherwise(None)
        .alias("_gk_rk")
    )
    # Prioridade: melhor GK do grupo vem sempre (−1); demais pela ordem de overall.
    g = g.with_columns(
        pl.when(pl.col("_gk_rk") == 0).then(-1).otherwise(pl.col("_ovr_rk")).alias("_prio")
    )
    g = g.sort(group_keys + ["_prio"])
    g = g.with_columns(pl.int_range(pl.len()).over(group_keys).alias("_rk"))
    g = g.filter(pl.col("_rk") < N_STARTERS)
    # Reordena o XI por setor (GK→FWD) e overall p/ atribuir o slot final.
    g = g.sort(
        group_keys + ["sector_order", "overall"],
        descending=[False] * len(group_keys) + [False, True],
    )
    g = g.with_columns(pl.int_range(pl.len()).over(group_keys).alias("_rk2"))
    return (
        g.with_columns((pl.col("_rk2") + 1).alias("slot"))
        .drop("_ovr_rk", "_gk_rk", "_prio", "_rk", "_rk2", "sector_order", "eff_position")
    )


# ---------------------------------------------------------------------------
# Config das fontes de partida do FIFA
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_fifa_match_config() -> dict:
    with open(_FIFA_SOURCES_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def fifa_min_match_date() -> str:
    return str(load_fifa_match_config().get("min_match_date", "2015-01-01"))


def competition_league_ids() -> dict[str, list[int]]:
    """code da competição → lista de fifa_league_ids (clubes). Vazio p/ seleções."""
    out: dict[str, list[int]] = {}
    for c in load_fifa_match_config().get("competitions", []) or []:
        out[c["code"]] = [int(x) for x in (c.get("fifa_league_ids") or [])]
    return out


__all__ = [
    "FIFA_PLAYER_ATTRS",
    "N_STARTERS",
    "FIFA_FLATFILE_COLUMNS",
    "SECTOR_ORDER",
    "sector_of",
    "sector_expr",
    "sector_order_expr",
    "strip_accents",
    "normalize_team_name",
    "build_team_matcher",
    "NATION_ALIASES",
    "CLUB_ALIASES",
    "starting_xi_long",
    "nation_fallback_xi_long",
    "load_fifa_match_config",
    "fifa_min_match_date",
    "competition_league_ids",
]
