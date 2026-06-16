"""Flatfile genérico (pass-through) para os CSVs do StatsBomb.

Materializa um CSV "limpo" como dataset Parquet na camada flatfile, sem
transformações de negócio — apenas a tipagem nativa que o polars infere na
leitura. É reutilizado por todos os arquivos sb_*.csv (partidas, escalações,
stats de jogador, competições): o mesmo módulo serve a vários jobs do
etl_config.yaml, cada um com seu próprio `input_tables`/`output_table_name`.

O ETLBase (classe ETLBaseParquetPolars) já leu o CSV indicado em `input_tables`
e o entregou em `input_tables` (dict alias -> DataFrame). Aqui só devolvemos
esse DataFrame para ser persistido como `flatfile_<output_table_name>`.
"""
import polars as pl


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    # Cada job pass-through declara exatamente uma fonte; pegamos a primeira.
    df = next(iter(input_tables.values()))
    # Pass-through: a camada flatfile é a "foto fiel" do dado bruto em Parquet.
    # Qualquer limpeza/tipagem de negócio acontece nas camadas bronze/silver.
    return df
