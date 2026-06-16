"""Flatfile do FIFA — materializa o `fifa_aggregated.csv` (vários GB) como Parquet.

A leitura pesada (lazy `scan_csv` + projeção das ~20 colunas usadas + engine de
streaming) é feita pela base `macros.bases.ETLFifaPlayersFlatfilePolars`, que já
entrega o DataFrame projetado em `input_tables`. Aqui só devolvemos esse frame
para ser persistido como `flatfile_fifa_players` — uma "foto" enxuta do dado
bruto, de onde a bronze faz a tipagem de negócio.
"""
import polars as pl


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return next(iter(input_tables.values()))
