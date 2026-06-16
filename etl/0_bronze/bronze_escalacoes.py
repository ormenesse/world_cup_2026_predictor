"""Bronze — escalações (uma linha por jogador escalado por partida).

Mantém a ESCALAÇÃO COMPLETA (titulares + reservas que entraram). A flag
`is_starter` permite que a gold use só os titulares agora, e que mais tarde se
teste o uso do elenco inteiro sem reprocessar esta camada.

Transformações:
  • `is_starter`         — titular (Starting XI)?  (parse de `positions`)
  • `cards_yellow`       — nº de cartões amarelos
  • `cards_second_yellow`— nº de segundos amarelos
  • `cards_red`          — nº de cartões vermelhos diretos
  • `cards_expulsion`    — expulsões = segundos amarelos + vermelhos diretos

Os campos `cards`/`positions` são `repr` de listas de dicts; o parsing é feito
de forma vetorizada por contagem de padrão (ver macros/features.py).
"""
import polars as pl

from macros.features import (
    card_count_expr,
    is_starter_expr,
    player_uid_expr,
    position_group_from_statsbomb,
)


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    esc = input_tables["escalacoes"]

    esc = esc.with_columns(
        is_starter_expr("positions").alias("is_starter"),
        position_group_from_statsbomb("positions"),  # GK/DEF/MID/FWD
        # ID canônico unificado (mesmo nome → mesmo id em qualquer fonte).
        player_uid_expr("player_name"),
        card_count_expr("Yellow Card").alias("cards_yellow"),
        card_count_expr("Second Yellow").alias("cards_second_yellow"),
        card_count_expr("Red Card").alias("cards_red"),
    ).with_columns(
        # Expulsão = saiu de campo por cartão: 2º amarelo OU vermelho direto.
        (pl.col("cards_second_yellow") + pl.col("cards_red")).alias("cards_expulsion"),
    )

    return esc.select(
        "match_id",
        "team",
        "competition_id",
        "season_id",
        "player_id",
        "player_name",
        "jersey_number",
        "is_starter",
        "position_group",
        "cards_yellow",
        "cards_second_yellow",
        "cards_red",
        "cards_expulsion",
        pl.lit("statsbomb").alias("source"),
    )
