"""Testes dos macros do dataset FIFA × partidas (offline, dados sintéticos).

Cobrem o que é fácil regredir: normalização de nome de time (acento/stopwords/
sufixo de UF), casamento fuzzy + aliases, e a seleção do XI titular em polars
(clube e fallback de seleção).
"""
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macros.fifa import (  # noqa: E402
    CLUB_ALIASES,
    NATION_ALIASES,
    build_team_matcher,
    nation_fallback_xi_long,
    normalize_team_name,
    starting_xi_long,
)


def test_normalize_team_name():
    assert normalize_team_name("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert normalize_team_name("Inter") == "inter"
    assert normalize_team_name("FC Köln") == "koln"          # stopword 'fc' removido
    assert normalize_team_name("Flamengo RJ") == "flamengo"  # sufixo de UF removido
    assert normalize_team_name("M'gladbach") == "m gladbach"  # apóstrofo → espaço
    assert normalize_team_name(None) == ""


def test_aliases_resolve_to_fifa_keys():
    # As chaves de alias devem casar o nome JÁ normalizado da partida.
    assert CLUB_ALIASES[normalize_team_name("M'gladbach")] == "borussia monchengladbach"
    assert CLUB_ALIASES[normalize_team_name("Ath Bilbao")] == "athletic club"
    assert NATION_ALIASES[normalize_team_name("South Korea")] == "korea republic"


def test_matcher_exact_fuzzy_and_containment():
    keys = [normalize_team_name(x) for x in ["Inter", "Hertha BSC", "Real Madrid", "Real Betis"]]
    m = build_team_matcher(keys)
    assert m(normalize_team_name("Inter Milan")) == "inter"        # contenção de token
    assert m(normalize_team_name("Hertha Berlin")) == "hertha"     # overlap token longo
    # "real" tem 4 chars → não dispara overlap; não casa Madrid↔Betis erroneamente.
    assert m(normalize_team_name("Real Madrid")) == "real madrid"


def _club_frame():
    pos = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "CAM", "ST", "LW", "RW", "SUB", "RES"]
    return pl.DataFrame({
        "snapshot_date": ["s"] * len(pos),
        "league_id": [19] * len(pos),
        "club_key": ["x"] * len(pos),
        "club_name": ["X"] * len(pos),
        "nationality": ["Italy"] * len(pos),
        "short_name": [f"p{i}" for i in range(len(pos))],
        "club_position": pos,
        "positions": pos,
        "overall": [float(90 - i) for i in range(len(pos))],
    })


def test_starting_xi_excludes_sub_res_and_orders_by_sector():
    xi = starting_xi_long(_club_frame(), pos_col="club_position",
                          group_keys=["snapshot_date", "league_id", "club_key"]).sort("slot")
    assert xi.height == 11
    names = xi["short_name"].to_list()
    assert "p11" not in names and "p12" not in names      # SUB/RES fora
    assert xi["sector"][0] == "GK"                          # GK no slot 1
    assert xi["slot"].to_list() == list(range(1, 12))


def test_nation_fallback_picks_best_gk_plus_top_overall():
    # club_position SUB/RES vira eff_position; fallback usa overall → SUB/RES elegíveis.
    fb = nation_fallback_xi_long(
        _club_frame().with_columns(pl.lit("Brazil").alias("nationality")),
        group_keys=["snapshot_date"],
    )
    assert fb.height == 11
    assert (fb.filter(pl.col("sector") == "GK").height) >= 1   # tem ao menos 1 GK
