"""Gold (FIFA) — dataset wide: 1 linha por partida × XI do FIFA dos dois lados.

Junta os RESULTADOS de partida (silver_fifa_match_snapshot, já com o snapshot do
FIFA vigente) ao XI titular de cada time naquele snapshot (silver_fifa_team_snapshot)
e monta UMA linha por partida com:

  • identificação + alvos: home/away_team, placar, `result` (H/D/A),
    `target`/`home_result`/`away_result` (W/D/L);
  • para CADA lado, os 11 titulares (`{lado}_p01..p11_{name,position,<attrs>}`)
    com seus stats do FIFA (overall, potential, value_eur, age, pace, shooting,
    passing, dribbling, defending, physic);
  • o SCORE AGREGADO do time: `{lado}_xi_count`, `{lado}_xi_<attr>_mean`,
    `{lado}_xi_overall_{max,min}` e médias por setor `{lado}_xi_overall_mean_<SEC>`.

Casamento de nome (schedule ↔ FIFA): normalização (sem acento/pontuação/stopwords)
+ fallback fuzzy (difflib), reaproveitando `macros.fifa`. Clubes casam DENTRO do
pool de `fifa_league_ids` da competição (resolve Serie A ITA=31 vs BRA=7);
seleções casam pela nationality (com aliases). Lados não casados ficam com slots
nulos e `{lado}_matched=False`.
"""
import polars as pl

from macros.fifa import (
    CLUB_ALIASES,
    FIFA_PLAYER_ATTRS,
    N_STARTERS,
    NATION_ALIASES,
    build_team_matcher,
    competition_league_ids,
)

_NATION_SCOPE = -1  # scope_id sentinela p/ seleções (não têm league_id)


def _resolve_crosswalk(matches: pl.DataFrame, xi: pl.DataFrame) -> pl.DataFrame:
    """Resolve, por (partida, lado), o team_key do FIFA correspondente.

    Devolve: match_id, side, snapshot_date, scope_id, team_key (ou None se não
    casou). O casamento é por (snapshot, pool de league_id) p/ clubes e por
    snapshot p/ seleções, com matcher fuzzy cacheado por escopo.
    """
    pools = competition_league_ids()

    # Conjuntos de team_keys disponíveis por escopo (do XI do FIFA).
    club_keys: dict[tuple, list[str]] = {}
    for snap, lid, key in (
        xi.filter(pl.col("team_type") == "club")
        .select("snapshot_date", "league_id", "team_key").unique().iter_rows()
    ):
        club_keys.setdefault((snap, lid), []).append(key)
    nation_keys: dict[object, list[str]] = {}
    for snap, key in (
        xi.filter(pl.col("team_type") == "nation")
        .select("snapshot_date", "team_key").unique().iter_rows()
    ):
        nation_keys.setdefault(snap, []).append(key)

    club_matchers: dict[tuple, object] = {}
    nation_matchers: dict[object, object] = {}

    def resolve_club(snap, comp_code, key):
        key = CLUB_ALIASES.get(key, key)  # apelidos football-data → FIFA
        for lid in pools.get(comp_code, []):
            roster = club_keys.get((snap, lid))
            if not roster:
                continue
            fn = club_matchers.get((snap, lid))
            if fn is None:
                fn = club_matchers[(snap, lid)] = build_team_matcher(roster)
            hit = fn(key)
            if hit:
                return lid, hit
        return None, None

    def resolve_nation(snap, key):
        roster = nation_keys.get(snap)
        if not roster:
            return None
        fn = nation_matchers.get(snap)
        if fn is None:
            fn = nation_matchers[snap] = build_team_matcher(roster)
        return fn(NATION_ALIASES.get(key, key))

    rows = []
    cols = ["match_id", "snapshot_date", "competition_code", "match_type",
            "home_key", "away_key"]
    for m in matches.select(cols).iter_rows(named=True):
        snap = m["snapshot_date"]
        for side, key in (("home", m["home_key"]), ("away", m["away_key"])):
            scope_id, resolved = None, None
            if m["match_type"] == "club":
                scope_id, resolved = resolve_club(snap, m["competition_code"], key)
            else:
                resolved = resolve_nation(snap, key)
                scope_id = _NATION_SCOPE if resolved else None
            rows.append({
                "match_id": m["match_id"], "side": side, "snapshot_date": snap,
                "scope_id": scope_id, "team_key": resolved,
            })

    return pl.DataFrame(
        rows,
        schema={
            "match_id": pl.Utf8, "side": pl.Utf8, "snapshot_date": pl.Date,
            "scope_id": pl.Int64, "team_key": pl.Utf8,
        },
    )


def _player_slot_exprs() -> list[pl.Expr]:
    """Expressões p/ pivotar os 11 titulares de cada lado em colunas wide."""
    exprs: list[pl.Expr] = []
    for side in ("home", "away"):
        for i in range(1, N_STARTERS + 1):
            p = f"{side}_p{i:02d}"
            sel = (pl.col("side") == side) & (pl.col("slot") == i)
            exprs.append(pl.col("player_name").filter(sel).first().alias(f"{p}_name"))
            exprs.append(pl.col("position").filter(sel).first().alias(f"{p}_position"))
            for a in FIFA_PLAYER_ATTRS:
                exprs.append(pl.col(a).filter(sel).first().alias(f"{p}_{a}"))
    return exprs


def _side_aggregate_exprs() -> list[pl.Expr]:
    """Agregados do XI por lado: contagem, médias, max/min, médias por setor e
    SCORES de time ponderados por posição (*1*/*2*).

    Scores de time (a partir de cada jogador do XI):
      • team_avg_score    — média do `overall` dos 11 titulares.
      • team_attack_score — média, entre os FWD, de (shooting+pace+dribbling)/3.
      • team_mid_score    — média, entre os MID, de (passing+dribbling)/2.
      • team_def_score    — média, entre os DEF, de (defending+physic)/2.
      • team_gk_score     — média do `overall` dos GK (atributos de campo são
                            nulos p/ goleiro no FIFA, então usamos overall).
    """
    exprs: list[pl.Expr] = []
    for side in ("home", "away"):
        sel = pl.col("side") == side
        exprs.append(pl.col("player_name").filter(sel).len().alias(f"{side}_xi_count"))
        for a in FIFA_PLAYER_ATTRS:
            exprs.append(pl.col(a).filter(sel).mean().alias(f"{side}_xi_{a}_mean"))
        exprs.append(pl.col("overall").filter(sel).max().alias(f"{side}_xi_overall_max"))
        exprs.append(pl.col("overall").filter(sel).min().alias(f"{side}_xi_overall_min"))
        for sec in ("GK", "DEF", "MID", "FWD"):
            ssec = sel & (pl.col("sector") == sec)
            exprs.append(
                pl.col("overall").filter(ssec).mean().alias(f"{side}_xi_overall_mean_{sec}")
            )
        exprs.append(pl.col("squad_source").filter(sel).first().alias(f"{side}_squad_source"))

        # --- *1* score médio do time + *2* scores ponderados por posição ---
        sel_fwd = sel & (pl.col("sector") == "FWD")
        sel_mid = sel & (pl.col("sector") == "MID")
        sel_def = sel & (pl.col("sector") == "DEF")
        sel_gk = sel & (pl.col("sector") == "GK")
        attack = (pl.col("shooting") + pl.col("pace") + pl.col("dribbling")) / 3.0
        midf = (pl.col("passing") + pl.col("dribbling")) / 2.0
        defe = (pl.col("defending") + pl.col("physic")) / 2.0
        exprs.append(pl.col("overall").filter(sel).mean().alias(f"{side}_team_avg_score"))
        exprs.append(attack.filter(sel_fwd).mean().alias(f"{side}_team_attack_score"))
        exprs.append(midf.filter(sel_mid).mean().alias(f"{side}_team_mid_score"))
        exprs.append(defe.filter(sel_def).mean().alias(f"{side}_team_def_score"))
        exprs.append(pl.col("overall").filter(sel_gk).mean().alias(f"{side}_team_gk_score"))
    return exprs


def _win_rate_features(matches: pl.DataFrame) -> pl.DataFrame:
    """Win streak normalizado = VITÓRIAS nas últimas 10 partidas / 10, por lado.

    Sem vazamento: só conta jogos ESTRITAMENTE anteriores (shift(1)) à partida de
    referência, por time, na ordem cronológica. Devolve (match_id,
    home_win_rate_last10, away_win_rate_last10).
    """
    home = matches.select(
        "match_id", "match_date",
        pl.lit("home").alias("side"),
        pl.col("home_team").alias("team"),
        (pl.col("target") == "W").cast(pl.Int64).alias("win"),
    )
    away = matches.select(
        "match_id", "match_date",
        pl.lit("away").alias("side"),
        pl.col("away_team").alias("team"),
        (pl.col("away_result") == "W").cast(pl.Int64).alias("win"),
    )
    long = pl.concat([home, away]).sort(["team", "match_date"])

    long = long.with_columns(
        # vitórias nas até 10 partidas anteriores (shift(1) exclui a atual).
        pl.col("win").shift(1).rolling_sum(window_size=10, min_periods=1).over("team").alias("_w"),
        # nº de partidas anteriores consideradas (posição na ordem, no máx. 10).
        pl.min_horizontal(pl.int_range(pl.len()).over("team"), 10).alias("_n"),
    ).with_columns(
        pl.when(pl.col("_n") > 0)
        .then(pl.col("_w").fill_null(0) / 10.0)  # normalizado por 10 (win streak)
        .otherwise(0.0)
        .alias("win_rate_last10")
    )

    home_wr = long.filter(pl.col("side") == "home").select(
        "match_id", pl.col("win_rate_last10").alias("home_win_rate_last10")
    )
    away_wr = long.filter(pl.col("side") == "away").select(
        "match_id", pl.col("win_rate_last10").alias("away_win_rate_last10")
    )
    return home_wr.join(away_wr, on="match_id", how="full", coalesce=True)


def _mirror(final: pl.DataFrame) -> pl.DataFrame:
    """Espelha cada partida (HOME↔AWAY) — data augmentation que dobra a tabela.

    Troca TODAS as colunas `home_*`↔`away_*` (jogadores, scores, win rate, flags,
    placar, time) e inverte os alvos não-prefixados: `result` H↔A, `target` W↔L
    (D fica D). `home_result`/`away_result` são trocados pela renomeação. O
    `match_id` recebe sufixo `:mirror` para permanecer único.
    """
    swap = {}
    for c in final.columns:
        if c.startswith("home_"):
            swap[c] = "away_" + c[len("home_"):]
        elif c.startswith("away_"):
            swap[c] = "home_" + c[len("away_"):]
    mirror = final.rename(swap).with_columns(
        pl.when(pl.col("result") == "H").then(pl.lit("A"))
        .when(pl.col("result") == "A").then(pl.lit("H"))
        .otherwise(pl.lit("D")).alias("result"),
        pl.when(pl.col("target") == "W").then(pl.lit("L"))
        .when(pl.col("target") == "L").then(pl.lit("W"))
        .otherwise(pl.lit("D")).alias("target"),
        (pl.col("match_id") + pl.lit(":mirror")).alias("match_id"),
    )
    return mirror.select(final.columns)


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    matches = input_tables["matches"]
    xi = input_tables["xi"]

    # scope_id no XI: league_id (clube) ou sentinela (seleção).
    xi = xi.with_columns(
        pl.when(pl.col("team_type") == "club")
        .then(pl.col("league_id"))
        .otherwise(pl.lit(_NATION_SCOPE))
        .cast(pl.Int64)
        .alias("scope_id")
    )

    # 1) Resolve o team_key do FIFA p/ cada (partida, lado).
    crosswalk = _resolve_crosswalk(matches, xi)

    # 2) Liga ao XI → uma linha por (partida, lado, jogador titular).
    resolved = crosswalk.filter(pl.col("team_key").is_not_null()).join(
        xi.select(
            "snapshot_date", "scope_id", "team_key", "slot", "position", "sector",
            "player_name", "squad_source", *FIFA_PLAYER_ATTRS,
        ),
        on=["snapshot_date", "scope_id", "team_key"],
        how="inner",
    )

    # 3) Pivota titulares + agregados p/ 1 linha por partida.
    players_wide = resolved.group_by("match_id").agg(
        _player_slot_exprs() + _side_aggregate_exprs()
    )

    # 4) Flags de casamento por lado (a partir do crosswalk).
    matched = (
        crosswalk.with_columns(pl.col("team_key").is_not_null().alias("_m"))
        .group_by("match_id")
        .agg(
            pl.col("_m").filter(pl.col("side") == "home").first().alias("home_matched"),
            pl.col("_m").filter(pl.col("side") == "away").first().alias("away_matched"),
        )
    )

    # 5) Base de identificação + alvos.
    base = matches.select(
        "match_id", "match_date", "snapshot_date", "competition_code",
        "competition_group", "match_type", "season",
        "home_team", "away_team", "home_goals", "away_goals",
        "result", "target", "home_result", "away_result",
    )

    # 6) Win streak (vitórias/10 nas últimas 10) por lado — *3*.
    win_rate = _win_rate_features(matches)

    final = (
        base.join(matched, on="match_id", how="left")
        .join(players_wide, on="match_id", how="left")
        .join(win_rate, on="match_id", how="left")
        .with_columns(
            pl.col("home_matched").fill_null(False),
            pl.col("away_matched").fill_null(False),
            pl.col("home_xi_count").fill_null(0),
            pl.col("away_xi_count").fill_null(0),
            pl.col("home_win_rate_last10").fill_null(0.0),
            pl.col("away_win_rate_last10").fill_null(0.0),
        )
    )

    # 7) Augmentation: anexa o espelho (HOME↔AWAY) → tabela com o dobro de linhas.
    final = pl.concat([final, _mirror(final)], how="vertical_relaxed")
    return final.sort(["match_date", "match_id"])
