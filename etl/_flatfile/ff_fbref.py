"""Flatfile para os CSVs do FBref (cabeçalho de 3 linhas).

Os exports do FBref trazem um cabeçalho hierárquico em 3 linhas
(grupo / estatística / chave), que o leitor de CSV padrão do ETLBase não
entende — ele trataria a 1ª linha como header e as outras duas como dados.

Como o contrato do bolt_pipeliner não permite passar opções de leitura por job,
reabrimos o arquivo aqui com a função `read_fbref_csv`, que resolve o cabeçalho
e devolve nomes de coluna únicos (ex.: 'Performance_Gls', 'Per 90 Minutes_Gls').
O DataFrame pré-carregado pelo ETLBase (com cabeçalho "errado") é ignorado de
propósito.

Reutilizado por todos os fbref_*.csv (goleiros, temporada, chutes, diversos,
minutos).
"""
import polars as pl

from macros.fbref import read_fbref_csv


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    # Recupera o caminho real do CSV declarado em `input_tables` (o ETLBase
    # guarda os caminhos-fonte em `self.input_table_names`) e o resolve contra o
    # bucket de dados configurado.
    source = next(iter(self.input_table_names.values()))
    path = self._resolve_input_path(source)

    # Releitura com o cabeçalho de 3 linhas corretamente combinado.
    return read_fbref_csv(path)
