"""Bases de ETL customizadas do projeto (infraestrutura compartilhada).

O `class_name` de cada job no etl_config.yaml pode ser um nome embutido
(ex.: `ETLBaseParquetPolars`) OU um caminho pontilhado para uma classe própria
(ex.: `macros.bases.ETLFlatfilePolars`). Como essas classes são reaproveitadas
por vários jobs, vivem aqui em `macros/`.

Classes
-------
ETLFlatfilePolars
    Lê CSV forçando a inferência de schema sobre o ARQUIVO INTEIRO
    (`infer_schema_length=None`), evitando quebras em colunas com valores
    mistos/raros (ex.: co-técnicos em `sb_partidas.away_manager_id` = "4268, 228").
ETLTolerantPolars
    NÃO quebra quando uma fonte de entrada está ausente/vazia (fontes opcionais
    como FBref e os jobs de UNIÃO): a tabela vira um DataFrame vazio.
ETLFifaPlayersFlatfilePolars
    Lê o `fifa_aggregated.csv` (vários GB) de forma PREGUIÇOSA via `scan_csv`,
    projetando apenas as ~20 colunas usadas e materializando com o engine de
    streaming — evita carregar o arquivo inteiro na memória.
"""
from __future__ import annotations

import logging

import polars as pl

from bolt_pipeliner.bases._io import detect_file_format
from bolt_pipeliner.bases.polars_parquet import ETLBaseParquetPolars


class ETLFlatfilePolars(ETLBaseParquetPolars):
    """Base flatfile com inferência de schema robusta para CSVs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inferência sobre o arquivo todo: evita erros de tipo em colunas com
        # valores mistos/raros (vide docstring do módulo).
        self.storage_options = {**(self.storage_options or {}), "infer_schema_length": None}


class ETLTolerantPolars(ETLBaseParquetPolars):
    """Base que NÃO quebra quando uma fonte de entrada está ausente/vazia.

    Usada nas fontes opcionais (ex.: FBref, que só existe depois de o usuário
    rodar o fetcher) e nos jobs de UNIÃO. Se o glob de uma entrada não casa com
    nenhum arquivo (fonte ainda não materializada) ou a leitura falha, em vez de
    levantar exceção a tabela vira um DataFrame VAZIO. Assim o pipeline inteiro
    continua rodando mesmo sem os dados opcionais — os jobs de união
    simplesmente ignoram as fontes vazias.
    """

    def load_data(self):
        if not self.input_table_names:
            return
        for key, source in self.input_table_names.items():
            path = self._resolve_input_path(source)
            fmt = detect_file_format(path)
            try:
                if fmt == "csv":
                    df = pl.read_csv(path, **self.storage_options)
                elif fmt == "parquet":
                    df = pl.read_parquet(path, **self.storage_options)
                else:
                    # Reusa o leitor da base para excel/json.
                    super().load_data()
                    return
            except Exception as exc:  # fonte ausente/vazia → tabela vazia
                logging.info(
                    "%s - fonte opcional '%s' indisponível (%s); tratada como vazia.",
                    self.logging_string,
                    source,
                    type(exc).__name__,
                )
                df = pl.DataFrame()
            self.input_tables[key] = df


class ETLFifaPlayersFlatfilePolars(ETLBaseParquetPolars):
    """Flatfile do `fifa_aggregated.csv` com leitura preguiçosa e projeção.

    O agregado do FIFA tem 100+ colunas e vários GB. A base padrão faria um
    `pl.read_csv` ANSIOSO (arquivo inteiro na memória, com inferência de tipos)
    e poderia estourar a RAM. Aqui usamos `pl.scan_csv` (lazy), lendo TUDO como
    texto (`infer_schema_length=0` → sem custo de inferência e à prova de valores
    sujos), projetamos apenas as colunas que o pipeline FIFA usa
    (`macros.fifa.FIFA_FLATFILE_COLUMNS`) e materializamos com o engine de
    STREAMING. A tipagem de negócio acontece depois, na bronze.
    """

    def load_data(self):
        from macros.fifa import FIFA_FLATFILE_COLUMNS

        if not self.input_table_names:
            return
        for key, source in self.input_table_names.items():
            path = self._resolve_input_path(source)
            lf = pl.scan_csv(
                path,
                infer_schema_length=0,   # tudo como Utf8: sem inferência, robusto
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
            available = set(lf.collect_schema().names())
            cols = [c for c in FIFA_FLATFILE_COLUMNS if c in available]
            df = lf.select(cols).collect(engine="streaming")
            self.input_tables[key] = df
            logging.info(
                "%s - FIFA players carregado (streaming, %d cols) de %s",
                self.logging_string, len(cols), path,
            )


__all__ = [
    "ETLFlatfilePolars",
    "ETLTolerantPolars",
    "ETLFifaPlayersFlatfilePolars",
]
