"""Bronze (FIFA) — ratings de jogador por edição/snapshot, limpos e tipados.

Converte o flatfile `fifa_players` (tudo string) para tipos de negócio:
  • `fifa_update_date` → Date (cada data é um SNAPSHOT do sofifa);
  • atributos do jogador (overall, pace, value_eur, ...) → Float64;
  • `league_id` → Int64;
  • chaves de casamento de time normalizadas (`club_key`, `nation_key`) — nome
    sem acento/pontuação e sem stopwords genéricas (ver `macros.fifa`), para
    cruzar com os nomes de time do schedule na gold.

Mantém apenas linhas com snapshot e overall válidos (FC26/linhas sem data ou sem
overall ficam fora do casamento temporal). Uma linha = um jogador numa edição.
"""
import polars as pl

from macros.fifa import FIFA_PLAYER_ATTRS, normalize_team_name


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    players = input_tables["players"]

    players = players.with_columns(
        pl.col("fifa_update_date").str.to_date("%Y-%m-%d", strict=False).alias("fifa_update_date"),
        pl.col("league_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        *[pl.col(a).cast(pl.Float64, strict=False) for a in FIFA_PLAYER_ATTRS],
    )

    # Só snapshots/overall válidos (sem dado temporal não há como cruzar com a partida).
    players = players.filter(
        pl.col("fifa_update_date").is_not_null() & pl.col("overall").is_not_null()
    )

    # Chaves normalizadas de time (clube e seleção). A normalização é uma UDF
    # python (mesma usada no casamento de nomes da gold), então a aplicamos só nos
    # nomes DISTINTOS e juntamos de volta — evita ~20M chamadas em 10M+ linhas.
    clubs = players.select("club_name").unique().with_columns(
        pl.col("club_name").map_elements(normalize_team_name, return_dtype=pl.Utf8).alias("club_key")
    )
    nations = players.select("nationality").unique().with_columns(
        pl.col("nationality").map_elements(normalize_team_name, return_dtype=pl.Utf8).alias("nation_key")
    )
    players = players.join(clubs, on="club_name", how="left").join(nations, on="nationality", how="left")

    return players.select(
        "fifa_update_date",
        "league_id",
        "club_name",
        "club_key",
        "club_position",
        "nationality",
        "nation_key",
        "nation_position",
        "positions",
        pl.col("short_name").alias("player_name"),
        *FIFA_PLAYER_ATTRS,
    )
