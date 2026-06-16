"""Bronze — estatísticas por jogador por partida (granularidade base).

Esta é a tabela mais importante para as features históricas: o StatsBomb fornece
estatísticas por jogador EM CADA PARTIDA (gols, chutes no alvo, xG, passes,
faltas, etc.). É a única fonte com granularidade por partida — por isso é a base
das médias móveis de 18 meses (os CSVs do FBref são agregados por temporada).

O job:
  • enriquece cada linha de stats com a data/calendário da partida (de
    bronze_partidas) — necessário para janelas temporais;
  • define o lado (`side` = home/away) comparando o time da linha com o mandante;
  • anexa `player_id` e `is_starter` (de bronze_escalacoes), pois sb_stats_jogador
    só traz o NOME do jogador — e queremos chavear o histórico por id estável;
  • mantém apenas as estatísticas configuradas em feature_config.yaml.
"""
import polars as pl

from macros.features import get_player_stats, player_uid_expr


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    stats = input_tables["stats"]
    partidas = input_tables["partidas"]
    escalacoes = input_tables["escalacoes"]

    stats_cols = get_player_stats()
    stats = stats.rename({"player": "player_name"})

    # Índices defensivos DERIVADOS dos eventos brutos (antes do cast geral, pois
    # entram em stats_cols via feature_config e fluem para silver/gold):
    #   • def_acoes = volume de ações defensivas positivas;
    #   • def_erros = ações que favorecem o adversário (quanto menor, melhor).
    # Componentes ausentes/nulos contam como 0.
    _DEF_POS = ["ev_interception", "ev_block", "ev_clearance", "ev_ball_recovery", "ev_duel"]
    _DEF_NEG = ["ev_error", "ev_dribbled_past", "ev_own_goal_against", "ev_foul_committed"]
    stats = stats.with_columns(
        pl.sum_horizontal(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in _DEF_POS if c in stats.columns]
        ).alias("def_acoes"),
        pl.sum_horizontal(
            [pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in _DEF_NEG if c in stats.columns]
        ).alias("def_erros"),
    )

    # Garante tipo numérico nas estatísticas configuradas (algumas podem vir como
    # string se houver valores ausentes). def_acoes/def_erros já existem aqui.
    stats = stats.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in stats_cols if c in stats.columns]
    )

    # Contexto temporal e mando da partida.
    partidas_ctx = partidas.select(
        "match_id",
        "match_date",
        "year_month",
        "month_index",
        "competition_name",
        "season",
        "home_team",
    )

    # Titularidade e setor (posição) do jogador naquele jogo. O player_id NÃO vem
    # daqui: derivamos o id canônico (por nome) abaixo, unificado entre fontes.
    escalacoes_ctx = escalacoes.select(
        "match_id", "team", "player_name", "is_starter", "position_group"
    )

    out = (
        stats.join(partidas_ctx, on="match_id", how="inner")
        .join(escalacoes_ctx, on=["match_id", "team", "player_name"], how="left")
        .with_columns(
            # Lado do jogador na partida (mandante x visitante).
            pl.when(pl.col("team") == pl.col("home_team"))
            .then(pl.lit("home"))
            .otherwise(pl.lit("away"))
            .alias("side"),
        )
        .drop("home_team")
        .with_columns(
            pl.lit("statsbomb").alias("source"),
            player_uid_expr("player_name"),  # id canônico unificado (por nome)
        )
    )

    key_cols = [
        "match_id",
        "match_date",
        "year_month",
        "month_index",
        "competition_name",
        "season",
        "team",
        "side",
        "player_id",
        "player_name",
        "is_starter",
        "position_group",
        "source",
    ]
    return out.select(key_cols + stats_cols)
