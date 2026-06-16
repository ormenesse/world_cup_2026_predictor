"""Silver defensiva — gols/xG SOFRIDOS por TIME × year_month.

Baseline defensiva do time: quanto ele sofre por partida, por mês. Serve de
referência para o on-off dos jogadores (o gold compara "sofrido com o jogador"
contra esta média do time).

Saída por (team, year_month): `gols_sofridos_sum`, `xg_sofrido_sum`,
`n_partidas`, `month_index` — no mesmo formato que as outras silvers consomem
(soma + contagem → média ponderada da janela no gold).
"""
import polars as pl

from macros.features import team_match_conceded, year_month_to_month_index


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    stats = input_tables["stats"]
    partidas = input_tables["partidas"]

    # Gols/xG sofridos por time em cada partida.
    conceded = team_match_conceded(stats, partidas)

    silver = conceded.group_by(["team", "year_month"]).agg(
        pl.col("gols_sofridos").sum().alias("gols_sofridos_sum"),
        pl.col("xg_sofrido").sum().alias("xg_sofrido_sum"),
        pl.len().alias("n_partidas"),
    )
    silver = silver.with_columns(
        year_month_to_month_index("year_month").alias("month_index")
    )
    return silver.sort(["team", "year_month"])
