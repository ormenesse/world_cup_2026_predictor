"""Bronze — dimensão de partidas (uma linha por jogo).

Limpa e tipa o flatfile `sb_partidas`:
  • converte `match_date` (texto YYYY-MM-DD) para Date;
  • deriva `year_month` (YYYYMM) e `month_index` (ver macros/features.py);
  • normaliza placares para inteiros;
  • mantém só as colunas úteis ao restante do pipeline.

Esta tabela é a fonte das datas (para as janelas temporais) e dos alvos de
placar (gols por lado / resultado), além de definir quem é mandante/visitante.
"""
import polars as pl

from macros.features import add_calendar_columns


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    partidas = input_tables["partidas"]

    partidas = partidas.with_columns(
        # match_date vem como string; convertemos para Date para o calendário.
        pl.col("match_date").str.to_date("%Y-%m-%d").alias("match_date"),
        pl.col("home_score").cast(pl.Int64, strict=False),
        pl.col("away_score").cast(pl.Int64, strict=False),
    )

    # Adiciona year_month e month_index (usados pelas silvers e pela janela 18m).
    partidas = add_calendar_columns(partidas, date_col="match_date")

    return partidas.select(
        "match_id",
        "match_date",
        "year_month",
        "month_index",
        "competition_id",
        "competition_name",
        "season_id",
        "season",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_score",
        "away_score",
        # Provedor de origem — essencial para normalizar features entre fontes
        # (StatsBomb e FBref/Opta não estão na mesma escala). Ver PIPELINE.md.
        pl.lit("statsbomb").alias("source"),
    )
