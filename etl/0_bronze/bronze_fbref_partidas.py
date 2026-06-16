"""Bronze (FBref) — partidas no schema canônico.

Lê o `schedule` bruto gravado pelo fetcher do FBref e o converte para o MESMO
schema de `bronze_partidas` (StatsBomb), via `macros.fbref`. Usa base
TOLERANTE: se o FBref ainda não foi baixado, a entrada vem vazia e devolvemos um
DataFrame vazio (o pipeline segue só com o StatsBomb).
"""
import polars as pl

from macros.fbref import fbref_partidas


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    schedule = input_tables.get("schedule", pl.DataFrame())
    if schedule.is_empty():
        return pl.DataFrame()
    return fbref_partidas(schedule)
