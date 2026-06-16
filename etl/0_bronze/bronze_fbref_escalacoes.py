"""Bronze (FBref) — escalações no schema canônico.

Combina `lineup` (titularidade) + cartões do `player_match_summary` e devolve o
schema de `bronze_escalacoes`. Base tolerante (vazio quando FBref ausente).
"""
import polars as pl

from macros.fbref import fbref_escalacoes


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    lineup = input_tables.get("lineup", pl.DataFrame())
    summary = input_tables.get("summary", pl.DataFrame())
    if lineup.is_empty():
        return pl.DataFrame()
    return fbref_escalacoes(lineup, summary)
