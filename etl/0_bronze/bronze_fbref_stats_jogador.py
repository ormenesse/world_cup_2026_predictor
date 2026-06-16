"""Bronze (FBref) — stats por jogador/partida no schema canônico.

Converte o `player_match_summary` do FBref para o schema de
`bronze_stats_jogador`, mapeando as colunas disponíveis (gols, chutes, SoT, xG,
passes, interceptações, etc.) e preenchendo com nulo as estatísticas que o FBref
não fornece. As colunas de estatística seguem `feature_config.yaml`. Base
tolerante (vazio quando FBref ausente).
"""
import polars as pl

from macros.fbref import fbref_stats_jogador
from macros.features import get_player_stats


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    summary = input_tables.get("summary", pl.DataFrame())
    schedule = input_tables.get("schedule", pl.DataFrame())
    lineup = input_tables.get("lineup", pl.DataFrame())
    if summary.is_empty():
        return pl.DataFrame()
    return fbref_stats_jogador(summary, schedule, lineup, get_player_stats())
