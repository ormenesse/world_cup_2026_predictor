"""Gold — tabela final orientada a partidas (1 linha por jogo).

Esta é a entrega central do pipeline. Cada linha é uma partida e contém:

  FEATURES (calculadas com dados ESTRITAMENTE anteriores ao mês do jogo — janela
  de `rolling_window_months`, default 18 meses — para evitar data leakage):
    • home_team_form_<stat>_mean_18m / away_team_form_<stat>_mean_18m
        → "forma" recente do TIME (média ponderada por partida nos últimos 18m),
          a partir de silver_time_year_month.
    • home_lineup_<stat>_mean_18m / away_lineup_<stat>_mean_18m
        → média, entre os TITULARES escalados, da forma de 18m de cada jogador,
          a partir de silver_jogador_year_month.
      (Hoje usamos só os titulares; trocar para o elenco inteiro é só remover o
       filtro `is_starter`, pois a silver de jogador já cobre todos.)

  TARGETS (prefixo `target_` — o que realmente aconteceu no jogo, p/ modelagem):
    • target_result (H/D/A), target_home_goals, target_away_goals
    • target_<lado>_shots_on_target, target_<lado>_shots
    • target_<lado>_fouls
    • target_<lado>_yellow_cards, target_<lado>_expulsions, target_<lado>_red_cards

  NORMALIZAÇÃO (consolidada aqui, não há mais camada diamond):
    • <feature>_q — versão rank/quantil em (0,1] de cada feature de 18m,
      calculada DENTRO de cada `source` (comparável entre StatsBomb e FBref).

Tudo em polars. A média móvel ponderada é delegada a
`macros/features.rolling_window_means` (ver docstring lá para a matemática).
"""
import re

import polars as pl

from macros.features import (
    get_percentile_stats,
    get_percentiles,
    get_player_stats,
    get_position_groups,
    get_window_months,
    quantile_col_name,
    rolling_window_means,
    rolling_window_quantiles,
)

# Estatísticas defensivas coletivas (gols/xG sofridos) usadas no on-off.
_DEF_CONCEDED_STATS = ["gols_sofridos", "xg_sofrido"]


def _widen_by_side(df_side: pl.DataFrame, value_cols: list[str]) -> pl.DataFrame:
    """Transforma um frame longo (match_id, side, <cols>) em largo home_/away_.

    Em vez de `pivot` (cujos nomes de coluna são ambíguos com múltiplos valores),
    separamos explicitamente os lados e prefixamos as colunas, garantindo nomes
    previsíveis: `home_<col>` e `away_<col>`. Junta-se por match_id.
    """
    home = df_side.filter(pl.col("side") == "home").drop("side").rename(
        {c: f"home_{c}" for c in value_cols}
    )
    away = df_side.filter(pl.col("side") == "away").drop("side").rename(
        {c: f"away_{c}" for c in value_cols}
    )
    return home.join(away, on="match_id", how="full", coalesce=True)


def _widen_by_side_group(
    df: pl.DataFrame, value_cols: list[str], groups: list[str]
) -> pl.DataFrame:
    """Largura por LADO e GRUPO de posição: `home_<grp>_<col>` / `away_<grp>_<col>`.

    Entrada longa: (match_id, side, position_group, <value_cols>). Para cada
    (lado, grupo) filtra, renomeia e junta por match_id. Permite features como
    `home_DEF_lineup_ev_interception_mean_18m` ou `away_MID_...`.
    """
    base = df.select("match_id").unique()
    for side in ("home", "away"):
        for grp in groups:
            part = (
                df.filter((pl.col("side") == side) & (pl.col("position_group") == grp))
                .drop("side", "position_group")
                .rename({c: f"{side}_{grp}_{c}" for c in value_cols})
            )
            base = base.join(part, on="match_id", how="left")
    return base


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    partidas = input_tables["partidas"]
    escalacoes = input_tables["escalacoes"]
    stats = input_tables["stats"]
    silver_time = input_tables["silver_time"]
    silver_jogador = input_tables["silver_jogador"]
    silver_time_def = input_tables["silver_time_def"]
    silver_jogador_def = input_tables["silver_jogador_def"]

    stat_cols = get_player_stats()
    window = get_window_months()
    groups = get_position_groups()
    mean_names = [f"{s}_mean_{window}m" for s in stat_cols]  # nomes de saída do rolling
    def_mean_names = [f"{s}_mean_{window}m" for s in _DEF_CONCEDED_STATS]

    # ------------------------------------------------------------------
    # Base: uma linha por partida + chaves de calendário e identificação.
    # ------------------------------------------------------------------
    base = partidas.select(
        "match_id",
        "match_date",
        "year_month",
        "month_index",
        "competition_name",
        "season",
        # Provedor da partida (statsbomb/fbref). As features de uma linha são
        # calculadas só com histórico da MESMA fonte; esta coluna permite
        # normalizar/segmentar por provedor no downstream (escalas diferentes).
        "source",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )

    # Mapa partida -> (mês de referência, times de cada lado), em formato longo
    # (uma linha por lado), para alimentar as janelas móveis.
    sides = pl.concat(
        [
            partidas.select(
                "match_id",
                pl.lit("home").alias("side"),
                pl.col("home_team").alias("team"),
                pl.col("month_index").alias("ref_month_index"),
            ),
            partidas.select(
                "match_id",
                pl.lit("away").alias("side"),
                pl.col("away_team").alias("team"),
                pl.col("month_index").alias("ref_month_index"),
            ),
        ]
    )

    # ==================================================================
    # FEATURES 1 — forma recente do TIME (silver_time, janela de 18m)
    # ==================================================================
    team_form_long = rolling_window_means(
        monthly=silver_time,
        requests=sides,
        entity_keys=["team"],
        row_keys=["match_id", "side"],
        stats=stat_cols,
        window_months=window,
        prefix="team_form_",
    )
    team_value_cols = [f"team_form_{n}" for n in mean_names] + [f"team_form_n_partidas_{window}m"]
    team_form_wide = _widen_by_side(team_form_long, team_value_cols)

    # ==================================================================
    # FEATURES 2 — forma média dos TITULARES (silver_jogador, janela 18m)
    # ==================================================================
    # Titulares de cada partida, com lado, mês de referência e SETOR (posição).
    titulares = (
        escalacoes.filter(pl.col("is_starter"))
        .select("match_id", "team", "player_id", "position_group")
        .join(
            sides.select("match_id", "side", "team", "ref_month_index"),
            on=["match_id", "team"],
            how="inner",
        )
    )
    # Setor de cada (partida, lado, jogador) — para reatar às features (o rolling
    # devolve só as row_keys).
    titular_pos = titulares.select("match_id", "side", "player_id", "position_group")

    # Forma de 18m de CADA titular...
    player_form = rolling_window_means(
        monthly=silver_jogador,
        requests=titulares,
        entity_keys=["player_id"],
        row_keys=["match_id", "side", "player_id"],
        stats=stat_cols,
        window_months=window,
        prefix="p_",
    ).join(titular_pos, on=["match_id", "side", "player_id"], how="left")

    # ...agregada (média) entre TODOS os titulares -> feature do lado.
    lineup_long = player_form.group_by(["match_id", "side"]).agg(
        [pl.col(f"p_{n}").mean().alias(f"lineup_{n}") for n in mean_names]
        + [
            pl.len().alias("lineup_n_titulares"),
            pl.col(f"p_n_partidas_{window}m").is_not_null().sum().alias("lineup_n_com_historico"),
        ]
    )
    lineup_value_cols = [f"lineup_{n}" for n in mean_names] + [
        "lineup_n_titulares",
        "lineup_n_com_historico",
    ]
    lineup_wide = _widen_by_side(lineup_long, lineup_value_cols)

    # ...e agregada por SETOR (GK/DEF/MID/FWD) -> avalia zagueiros, meio-campo etc.
    # Gera home_DEF_<stat>_mean_18m, home_MID_<stat>_mean_18m, away_FWD_..., etc.
    lineup_pos_long = player_form.group_by(["match_id", "side", "position_group"]).agg(
        [pl.col(f"p_{n}").mean().alias(f"setor_{n}") for n in mean_names]
        + [pl.len().alias("setor_n_titulares")]
    ).filter(pl.col("position_group").is_not_null())
    lineup_pos_value_cols = [f"setor_{n}" for n in mean_names] + ["setor_n_titulares"]
    lineup_pos_wide = _widen_by_side_group(lineup_pos_long, lineup_pos_value_cols, groups)

    # ==================================================================
    # FEATURES 3 — PERCENTIS (ex.: p75) de variáveis importantes, 18m
    # ------------------------------------------------------------------
    # Percentil exige a distribuição PARTIDA A PARTIDA (não dá para reconstruir
    # de soma/contagem mensal), então calculamos sobre os valores por partida.
    # ==================================================================
    pstats = get_percentile_stats()
    pquants = get_percentiles()

    team_pct_wide = None
    lineup_pct_wide = None
    if pstats and pquants:
        # ---- TIME: percentil dos TOTAIS do time por partida na janela ----
        # Total do time em cada partida (soma das contribuições dos jogadores),
        # mantendo month_index para a janela temporal.
        team_match_events = stats.group_by(["team", "match_id", "month_index"]).agg(
            [pl.col(s).sum().alias(s) for s in pstats]
        )
        team_pct_long = rolling_window_quantiles(
            events=team_match_events,
            requests=sides,
            entity_keys=["team"],
            row_keys=["match_id", "side"],
            stats=pstats,
            quantiles=pquants,
            window_months=window,
            prefix="team_form_",
        )
        team_pct_cols = [
            quantile_col_name("team_form_", s, q, window) for s in pstats for q in pquants
        ]
        team_pct_wide = _widen_by_side(team_pct_long, team_pct_cols)

        # ---- TITULARES: percentil por jogador, depois média entre os 11 ----
        # bronze_stats já é 1 linha por (jogador, partida); usamos como eventos.
        player_match_events = stats.select(["player_id", "month_index", *pstats])
        player_pct = rolling_window_quantiles(
            events=player_match_events,
            requests=titulares,
            entity_keys=["player_id"],
            row_keys=["match_id", "side", "player_id"],
            stats=pstats,
            quantiles=pquants,
            window_months=window,
            prefix="p_",
        )
        lineup_pct_long = player_pct.group_by(["match_id", "side"]).agg(
            [
                pl.col(quantile_col_name("p_", s, q, window))
                .mean()
                .alias(quantile_col_name("lineup_", s, q, window))
                for s in pstats
                for q in pquants
            ]
        )
        lineup_pct_cols = [
            quantile_col_name("lineup_", s, q, window) for s in pstats for q in pquants
        ]
        lineup_pct_wide = _widen_by_side(lineup_pct_long, lineup_pct_cols)

    # ==================================================================
    # FEATURES 4 — SOLIDEZ DEFENSIVA: gols/xG sofridos + on-off (18m)
    # ------------------------------------------------------------------
    # Defesa é coletiva: medimos o que o TIME sofre e o quanto sofre COM o
    # jogador/setor em campo. on-off = (com o jogador) − (baseline do time);
    # NEGATIVO ⇒ time sofre menos com ele ⇒ bom defensivamente NAQUELE time.
    # ==================================================================
    # (a) Baseline do time (média de gols/xG sofridos por partida na janela).
    team_def_long = rolling_window_means(
        monthly=silver_time_def,
        requests=sides,
        entity_keys=["team"],
        row_keys=["match_id", "side"],
        stats=_DEF_CONCEDED_STATS,
        window_months=window,
        prefix="team_def_",
    )
    team_def_cols = [f"team_def_{n}" for n in def_mean_names]
    team_def_wide = _widen_by_side(team_def_long, team_def_cols)

    # (b) "Com o jogador": gols/xG sofridos pelo time quando cada titular jogou.
    player_def = rolling_window_means(
        monthly=silver_jogador_def,
        requests=titulares,
        entity_keys=["player_id"],
        row_keys=["match_id", "side", "player_id"],
        stats=_DEF_CONCEDED_STATS,
        window_months=window,
        prefix="p_def_",
    ).join(titular_pos, on=["match_id", "side", "player_id"], how="left")

    # média entre todos os titulares...
    lineup_def_long = player_def.group_by(["match_id", "side"]).agg(
        [pl.col(f"p_def_{n}").mean().alias(f"lineup_def_{n}") for n in def_mean_names]
    )
    lineup_def_wide = _widen_by_side(lineup_def_long, [f"lineup_def_{n}" for n in def_mean_names])

    # ...e por setor (DEF/MID/...): solidez do time com aquele setor titular.
    setor_def_long = player_def.group_by(["match_id", "side", "position_group"]).agg(
        [pl.col(f"p_def_{n}").mean().alias(f"setor_def_{n}") for n in def_mean_names]
    ).filter(pl.col("position_group").is_not_null())
    setor_def_wide = _widen_by_side_group(setor_def_long, [f"setor_def_{n}" for n in def_mean_names], groups)

    # ==================================================================
    # TARGETS — o que aconteceu de fato na partida
    # ==================================================================
    # (a) Estatísticas reais por lado, somando as contribuições dos jogadores.
    stats_side = stats.group_by(["match_id", "side"]).agg(
        pl.col("chutes_no_alvo").sum().alias("shots_on_target"),
        pl.col("ev_shot").sum().alias("shots"),
        pl.col("ev_foul_committed").sum().alias("fouls"),
    )
    # (b) Cartões/expulsões por lado, a partir das escalações (mais fiel).
    cartoes_side = (
        escalacoes.join(
            sides.select("match_id", "team", "side"),
            on=["match_id", "team"],
            how="inner",
        )
        .group_by(["match_id", "side"])
        .agg(
            pl.col("cards_yellow").sum().alias("yellow_cards"),
            pl.col("cards_expulsion").sum().alias("expulsions"),
            pl.col("cards_red").sum().alias("red_cards"),
        )
    )
    actuals_side = stats_side.join(cartoes_side, on=["match_id", "side"], how="full", coalesce=True)
    actuals_value_cols = [
        "shots_on_target",
        "shots",
        "fouls",
        "yellow_cards",
        "expulsions",
        "red_cards",
    ]
    actuals_wide = _widen_by_side(actuals_side, actuals_value_cols)

    # ==================================================================
    # Montagem final: base + features + targets
    # ==================================================================
    final = (
        base.join(team_form_wide, on="match_id", how="left")
        .join(lineup_wide, on="match_id", how="left")
        .join(lineup_pos_wide, on="match_id", how="left")   # ataque por setor
        .join(team_def_wide, on="match_id", how="left")     # baseline defensiva do time
        .join(lineup_def_wide, on="match_id", how="left")   # sofrido com os titulares
        .join(setor_def_wide, on="match_id", how="left")    # sofrido por setor
    )
    # Percentis (opcionais — só se configurados em feature_config.yaml).
    if team_pct_wide is not None:
        final = final.join(team_pct_wide, on="match_id", how="left")
    if lineup_pct_wide is not None:
        final = final.join(lineup_pct_wide, on="match_id", how="left")
    final = final.join(actuals_wide, on="match_id", how="left")

    # ------------------------------------------------------------------
    # on-off defensivo: (sofrido com o jogador/setor) − (baseline do time).
    # NEGATIVO = defende melhor que a média do time. Geral e por setor.
    # ------------------------------------------------------------------
    onoff_exprs = []
    for side in ("home", "away"):
        for n in def_mean_names:
            onoff_exprs.append(
                (pl.col(f"{side}_lineup_def_{n}") - pl.col(f"{side}_team_def_{n}"))
                .alias(f"{side}_onoff_{n}")
            )
            for grp in groups:
                onoff_exprs.append(
                    (pl.col(f"{side}_{grp}_setor_def_{n}") - pl.col(f"{side}_team_def_{n}"))
                    .alias(f"{side}_{grp}_onoff_{n}")
                )
    final = final.with_columns(onoff_exprs)

    # ------------------------------------------------------------------
    # Índice de confiabilidade defensiva = ações / (ações + erros), 18m.
    # Geral (lineup) e por setor. Mais alto = mais ações "limpas".
    # ------------------------------------------------------------------
    acoes = f"def_acoes_mean_{window}m"
    erros = f"def_erros_mean_{window}m"
    rel_exprs = []
    for side in ("home", "away"):
        rel_exprs.append(
            (
                pl.col(f"{side}_lineup_{acoes}")
                / (pl.col(f"{side}_lineup_{acoes}") + pl.col(f"{side}_lineup_{erros}"))
            ).alias(f"{side}_lineup_def_confiabilidade_{window}m")
        )
        for grp in groups:
            rel_exprs.append(
                (
                    pl.col(f"{side}_{grp}_setor_{acoes}")
                    / (pl.col(f"{side}_{grp}_setor_{acoes}") + pl.col(f"{side}_{grp}_setor_{erros}"))
                ).alias(f"{side}_{grp}_def_confiabilidade_{window}m")
            )
    final = final.with_columns(rel_exprs)

    # Targets de placar/resultado, prefixados com `target_`.
    final = final.with_columns(
        pl.col("home_score").alias("target_home_goals"),
        pl.col("away_score").alias("target_away_goals"),
        pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(pl.lit("H"))
        .when(pl.col("home_score") < pl.col("away_score"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("target_result"),
    )

    # Renomeia os agregados reais por lado para o namespace `target_`.
    rename_targets = {}
    for side in ("home", "away"):
        for col in actuals_value_cols:
            rename_targets[f"{side}_{col}"] = f"target_{side}_{col}"
    final = final.rename(rename_targets)

    # ==================================================================
    # NORMALIZAÇÃO por FONTE (rank/quantil) — antes ficava na diamond; agora
    # TUDO na gold. Para cada feature de 18m gera uma versão `_q` em (0,1]
    # comparável entre provedores (StatsBomb vs FBref/Opta), agrupando por
    # `source` (preserva diferença de força entre ligas da mesma fonte).
    # Linhas sem histórico (feature nula) permanecem nulas no `_q`.
    # ==================================================================
    feature_cols = [
        c
        for c in final.columns
        if (c.endswith(f"_mean_{window}m") or re.search(rf"_p\d+_{window}m$", c))
        and not c.startswith("target_")
    ]
    norm_exprs = [
        (pl.col(c).rank(method="average") / pl.col(c).count())
        .over("source")
        .alias(f"{c}_q")
        for c in feature_cols
    ]
    final = final.with_columns(norm_exprs)

    return final.sort("match_date")
