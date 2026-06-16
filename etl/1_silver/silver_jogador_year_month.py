"""Silver — estatísticas agregadas por JOGADOR × year_month.

Resume o histórico de cada jogador em granularidade mensal. Como em
bronze_stats_jogador há exatamente uma linha por (jogador, partida), agregar por
mês dá diretamente a "forma" mensal do jogador:

  • `{stat}_mean`  — média POR PARTIDA no mês (mean sobre as partidas do mês);
  • `{stat}_sum`   — total no mês;
  • `n_partidas`   — partidas disputadas no mês;
  • demais agregações configuradas em feature_config.yaml.

As colunas são totalmente parametrizáveis: vêm de `player_match_stats` e
`aggregations` no YAML. `{stat}_sum` e `n_partidas` são sempre gravados porque a
gold os usa para a média móvel ponderada de 18 meses.

Esta silver cobre o ELENCO INTEIRO (todos os jogadores com estatística), não só
titulares — a seleção de titulares acontece na gold.
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

    # Descarta linhas sem id de jogador (não dá para chavear histórico estável).
    stats = stats.filter(pl.col("player_id").is_not_null())

    # Uma linha por (jogador, partida) => agregar por mês resume a forma mensal.
    agg_exprs = silver_agg_expressions(stat_cols, aggs)

    silver = stats.group_by(["player_id", "player_name", "year_month"]).agg(agg_exprs)

    # month_index facilita a aritmética de janela na gold (reconstruído de YYYYMM).
    silver = silver.with_columns(
        year_month_to_month_index("year_month").alias("month_index")
    )

    return silver.sort(["player_id", "year_month"])
