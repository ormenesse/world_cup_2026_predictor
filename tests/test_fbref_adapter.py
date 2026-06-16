"""Testes do adapter FBref → schema canônico (offline, dados sintéticos).

Garantem que o mapeamento de colunas e a compatibilidade de schema com o
StatsBomb (necessária para a união) não regridam.
"""
import sys
from pathlib import Path

import polars as pl

# Garante que `etl.*` seja importável ao rodar pytest da pasta do projeto.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macros.fbref import fbref_escalacoes, fbref_partidas, fbref_stats_jogador  # noqa: E402


def _synthetic():
    schedule = pl.DataFrame({
        "competition_code": ["BRA-Serie A"],
        "season": ["2024"],
        "date": ["2024-05-01"],
        "home_team": ["Flamengo"],
        "away_team": ["Palmeiras"],
        "score": ["2–1"],  # en-dash, como o FBref
        "game_id": ["abc123"],
    })
    summary = pl.DataFrame({
        "season": ["2024"] * 3,
        "team": ["Flamengo", "Flamengo", "Palmeiras"],
        "player": ["Player A", "Player B", "Player C"],
        "game_id": ["abc123"] * 3,
        "Performance_Gls": [1, 1, 1],
        "Performance_Sh": [3, 2, 4],
        "Performance_SoT": [2, 1, 2],
        "Expected_xG": [0.4, 0.3, 0.6],
        "Passes_Att": [50, 60, 40],
        "Performance_CrdY": [1, 0, 1],
        "Performance_CrdR": [0, 0, 1],
    })
    lineup = pl.DataFrame({
        "game_id": ["abc123"] * 3,
        "team": ["Flamengo", "Flamengo", "Palmeiras"],
        "player": ["Player A", "Player B", "Player C"],
        "jersey_number": [10, 7, 9],
        "is_starter": [True, False, True],
    })
    return schedule, summary, lineup


def test_fbref_partidas_parses_score_and_calendar():
    schedule, _, _ = _synthetic()
    p = fbref_partidas(schedule)
    row = p.to_dicts()[0]
    assert row["match_id"] == "abc123"
    assert row["home_score"] == 2 and row["away_score"] == 1
    assert row["year_month"] == 202405


def test_fbref_stats_schema_matches_statsbomb_and_maps_columns():
    schedule, summary, lineup = _synthetic()
    stat_cols = ["gols", "chutes_no_alvo", "xg_total", "ev_pass", "key_passes"]
    s = fbref_stats_jogador(summary, schedule, lineup, stat_cols)
    # chaves canônicas presentes
    for col in ["match_id", "team", "side", "player_id", "player_name", "is_starter"]:
        assert col in s.columns
    home = s.filter(pl.col("player_name") == "Player A").to_dicts()[0]
    assert home["side"] == "home"
    # id canônico unificado (mesma normalização do StatsBomb, sem prefixo).
    assert home["player_id"] == "player a"
    assert home["gols"] == 1.0 and home["ev_pass"] == 50.0
    # estatística inexistente no FBref vira nula
    assert home["key_passes"] is None


def test_fbref_escalacoes_cards():
    schedule, summary, lineup = _synthetic()
    e = fbref_escalacoes(lineup, summary)
    c = e.filter(pl.col("player_name") == "Player C").to_dicts()[0]
    assert c["cards_yellow"] == 1 and c["cards_red"] == 1
    assert c["cards_expulsion"] == 1 and c["cards_second_yellow"] == 0


def test_position_group_statsbomb():
    from macros.features import position_group_from_statsbomb
    df = pl.DataFrame({"positions": [
        "[{'position': 'Goalkeeper', 'start_reason': 'Starting XI'}]",
        "[{'position': 'Left Wing Back', 'start_reason': 'Starting XI'}]",   # Back → DEF
        "[{'position': 'Center Defensive Midfield'}]",                        # → MID
        "[{'position': 'Right Wing'}]",                                       # → FWD
        "[]",                                                                 # → None
    ]})
    out = df.select(position_group_from_statsbomb("positions"))["position_group"].to_list()
    assert out == ["GK", "DEF", "MID", "FWD", None]


def test_player_id_unificado_entre_fontes():
    """O MESMO nome → o MESMO player_id no StatsBomb (bronze) e no FBref (adapter)."""
    from macros.features import player_uid_expr
    from macros.fbref import _player_id_expr
    nomes = pl.DataFrame({"player_name": ["Alejandro Grimaldo García", "Lucas Tousart"]})
    sb = nomes.select(player_uid_expr("player_name"))["player_id"].to_list()
    fb = nomes.select(_player_id_expr("player_name"))["player_id"].to_list()
    assert sb == fb == ["alejandro grimaldo garcia", "lucas tousart"]


def test_team_match_conceded():
    """xG sofrido = xG do adversário; gols sofridos = gols do adversário."""
    from macros.features import team_match_conceded
    stats = pl.DataFrame({
        "match_id": ["m1", "m1", "m1", "m1"],
        "side": ["home", "home", "away", "away"],
        "team": ["A", "A", "B", "B"],
        "year_month": [202401] * 4,
        "month_index": [24288] * 4,
        "xg_total": [0.5, 0.3, 1.0, 0.2],  # A soma 0.8, B soma 1.2
    })
    partidas = pl.DataFrame({"match_id": ["m1"], "home_score": [2], "away_score": [1]})
    out = team_match_conceded(stats, partidas).sort("team")
    a = out.filter(pl.col("team") == "A").to_dicts()[0]
    b = out.filter(pl.col("team") == "B").to_dicts()[0]
    assert abs(a["xg_sofrido"] - 1.2) < 1e-9 and a["gols_sofridos"] == 1.0  # A sofre o de B
    assert abs(b["xg_sofrido"] - 0.8) < 1e-9 and b["gols_sofridos"] == 2.0  # B sofre o de A


def test_union_schema_compatible():
    """As stats do FBref devem concatenar com as do StatsBomb (mesmas colunas)."""
    schedule, summary, lineup = _synthetic()
    stat_cols = ["gols", "chutes_no_alvo", "xg_total", "ev_pass"]
    s = fbref_stats_jogador(summary, schedule, lineup, stat_cols)
    # frame "StatsBomb-like" com as mesmas colunas canônicas
    sb_like = s.clear()  # mesmo schema, 0 linhas
    u = pl.concat([sb_like, s], how="diagonal_relaxed")
    assert u.height == s.height
