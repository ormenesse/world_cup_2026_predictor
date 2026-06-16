"""Silver defensiva — gols/xG SOFRIDOS pelo time COM o jogador × year_month.

Para cada jogador, considera as partidas em que ele foi TITULAR e atribui a ele
o que o time sofreu naquele jogo. Agregado por mês, vira a base do indicador de
solidez defensiva "com o jogador em campo". No gold, comparamos isto com a
baseline do time (silver_time_defensivo) para obter o **on-off**:
    on_off = (sofrido com o jogador)  −  (média do time)
valores negativos ⇒ o time sofre MENOS com ele ⇒ bom defensivamente NAQUELE time.

Saída por (player_id, player_name, year_month): `gols_sofridos_sum`,
`xg_sofrido_sum`, `n_partidas` (= jogos como titular), `month_index`.
"""
import polars as pl

from macros.features import team_match_conceded, year_month_to_month_index


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    stats = input_tables["stats"]
    partidas = input_tables["partidas"]

    # Gols/xG sofridos por time-partida.
    conceded = team_match_conceded(stats, partidas).select(
        "match_id", "team", "gols_sofridos", "xg_sofrido"
    )

    # Titulares (a contribuição defensiva conta quando o jogador está em campo).
    starters = (
        stats.filter(pl.col("is_starter") == True)  # noqa: E712
        .select("match_id", "team", "player_id", "player_name", "year_month")
        .filter(pl.col("player_id").is_not_null())
    )

    # Atribui ao jogador o que o time dele sofreu naquela partida.
    joined = starters.join(conceded, on=["match_id", "team"], how="inner")

    silver = joined.group_by(["player_id", "player_name", "year_month"]).agg(
        pl.col("gols_sofridos").sum().alias("gols_sofridos_sum"),
        pl.col("xg_sofrido").sum().alias("xg_sofrido_sum"),
        pl.len().alias("n_partidas"),
    )
    silver = silver.with_columns(
        year_month_to_month_index("year_month").alias("month_index")
    )
    return silver.sort(["player_id", "year_month"])
