"""Silver (FIFA) — atribui a cada partida o snapshot do FIFA vigente na época.

"Dados mais recentes do time sem vazar o futuro": para cada partida pegamos a
`fifa_update_date` mais recente <= `match_date` (as-of join "backward"). Partidas
anteriores ao 1º snapshot do FIFA ficam sem snapshot e são descartadas (não há
elenco de referência para elas).

Saída = bronze_fifa_matches + `snapshot_date`.
"""
import polars as pl


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    matches = input_tables["matches"]
    players = input_tables["players"]

    snaps = (
        players.select(pl.col("fifa_update_date").alias("snapshot_date"))
        .unique()
        .drop_nulls()
        .sort("snapshot_date")
    )
    matches = matches.sort("match_date")

    out = matches.join_asof(
        snaps,
        left_on="match_date",
        right_on="snapshot_date",
        strategy="backward",
    )
    # Sem snapshot vigente (partida anterior ao 1º update do FIFA) → fora.
    return out.filter(pl.col("snapshot_date").is_not_null())
