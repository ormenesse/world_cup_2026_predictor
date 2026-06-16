"""Silver — estatísticas agregadas por TIME × year_month.

Resume a forma mensal de cada time. Diferença importante para a silver de
jogador: em bronze_stats_jogador há VÁRIAS linhas por partida de um time (uma
por jogador). Para obter estatísticas "por partida" do time, agregamos em duas
etapas:

  1) por (time, partida): soma das contribuições dos jogadores  →  total do
     time naquela partida;
  2) por (time, year_month): agrega esses totais-por-partida. Assim:
       • `{stat}_mean`  = média do total do time POR PARTIDA no mês;
       • `{stat}_sum`   = total do time no mês;
       • `n_partidas`   = nº de partidas do time no mês (distinct match_id).

Sem a etapa (1), `n_partidas` contaria linhas de jogador (≈ 11–14 por jogo) e a
média ficaria sem sentido. As estatísticas e agregações são parametrizáveis
(feature_config.yaml), e `{stat}_sum`/`n_partidas` são sempre gravados para a
média móvel ponderada da gold.
"""
import polars as pl

from macros.features import (
    get_aggregations,
    get_player_stats,
    silver_agg_expressions,
    year_month_to_month_index,
)


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    stats = input_tables["stats"]

    stat_cols = get_player_stats()
    aggs = get_aggregations()

    # Etapa 1: total do time em cada partida (soma das contribuições dos jogadores).
    por_partida = stats.group_by(["team", "year_month", "match_id"]).agg(
        [pl.col(c).sum().alias(c) for c in stat_cols]
    )

    # Etapa 2: agrega os totais-por-partida no mês. As colunas de `por_partida`
    # têm os mesmos nomes das stats, então reaproveitamos as expressões padrão —
    # aqui cada "linha" agregada é uma PARTIDA do time, logo n_partidas = nº jogos.
    agg_exprs = silver_agg_expressions(stat_cols, aggs)
    silver = por_partida.group_by(["team", "year_month"]).agg(agg_exprs)

    silver = silver.with_columns(
        year_month_to_month_index("year_month").alias("month_index")
    )

    return silver.sort(["team", "year_month"])
