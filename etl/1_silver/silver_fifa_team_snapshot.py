"""Silver (FIFA) — XI titular de cada time em cada snapshot do FIFA (formato longo).

Para cada (snapshot do sofifa, time) materializa os 11 "jogadores principais",
uma linha por jogador (`slot` 1..11), com posição, setor e atributos. É a base
canônica de "elenco vigente" que a gold pivota para 1 linha por partida.

Três fontes de XI:
  • CLUBE        — XI real por `club_position` (sem SUB/RES), por
                   (snapshot, league_id, club_key). Restrito ao POOL de
                   `fifa_league_ids` das competições (config), pois é só nesse
                   pool que a gold casa clubes (resolve Serie A ITA=31 vs BRA=7).
  • SELEÇÃO OFICIAL — XI por `nation_position` quando o FIFA licencia a seleção.
  • SELEÇÃO FALLBACK — melhor GK + melhores restantes por overall, para as
                   seleções/snapshots sem escalação oficial (`squad_source`
                   registra qual XI foi usado).

Saída (long): snapshot_date, team_type, league_id, team_key, team_name,
squad_source, slot, position, sector, player_name, + atributos FIFA.
"""
import polars as pl

from macros.fifa import (
    FIFA_PLAYER_ATTRS,
    competition_league_ids,
    nation_fallback_xi_long,
    starting_xi_long,
)

_OUT_COLS = [
    "snapshot_date", "team_type", "league_id", "team_key", "team_name",
    "squad_source", "slot", "position", "sector", "player_name", *FIFA_PLAYER_ATTRS,
]


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    players = input_tables["players"].rename({"fifa_update_date": "snapshot_date"})

    frames: list[pl.DataFrame] = []

    # ----------------------------- CLUBES ------------------------------------
    pool = sorted({lid for ids in competition_league_ids().values() for lid in ids})
    club_src = players.filter(
        pl.col("club_key").is_not_null() & (pl.col("club_key") != "")
        & pl.col("club_position").is_not_null()
        & (pl.col("league_id").is_in(pool) if pool else pl.lit(True))
    )
    club_keys = ["snapshot_date", "league_id", "club_key"]
    club_xi = starting_xi_long(club_src, pos_col="club_position", group_keys=club_keys)
    if not club_xi.is_empty():
        frames.append(
            club_xi.with_columns(
                pl.lit("club").alias("team_type"),
                pl.col("club_key").alias("team_key"),
                pl.col("club_name").alias("team_name"),
                pl.lit("club").alias("squad_source"),
            ).select(_OUT_COLS)
        )

    # ------------------------- SELEÇÕES: OFICIAL -----------------------------
    nation_src = players.filter(
        pl.col("nation_key").is_not_null() & (pl.col("nation_key") != "")
    )
    nat_keys = ["snapshot_date", "nation_key"]

    official_src = nation_src.filter(pl.col("nation_position").is_not_null())
    official_xi = starting_xi_long(official_src, pos_col="nation_position", group_keys=nat_keys)
    official_pairs = (
        official_xi.select("snapshot_date", "nation_key").unique()
        if not official_xi.is_empty() else None
    )
    if not official_xi.is_empty():
        frames.append(
            official_xi.with_columns(
                pl.lit("nation").alias("team_type"),
                pl.lit(None, dtype=pl.Int64).alias("league_id"),
                pl.col("nation_key").alias("team_key"),
                pl.col("nationality").alias("team_name"),
                pl.lit("nation_official").alias("squad_source"),
            ).select(_OUT_COLS)
        )

    # ------------------------- SELEÇÕES: FALLBACK ----------------------------
    # Nationalities/snapshots SEM XI oficial → melhor GK + melhores por overall.
    if official_pairs is not None:
        fb_src = nation_src.join(
            official_pairs.with_columns(pl.lit(True).alias("_has_official")),
            on=["snapshot_date", "nation_key"], how="left",
        ).filter(pl.col("_has_official").is_null()).drop("_has_official")
    else:
        fb_src = nation_src
    fb_xi = nation_fallback_xi_long(fb_src, group_keys=nat_keys)
    if not fb_xi.is_empty():
        frames.append(
            fb_xi.with_columns(
                pl.lit("nation").alias("team_type"),
                pl.lit(None, dtype=pl.Int64).alias("league_id"),
                pl.col("nation_key").alias("team_key"),
                pl.col("nationality").alias("team_name"),
                pl.lit("nationality_top").alias("squad_source"),
            ).select(_OUT_COLS)
        )

    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _OUT_COLS})

    out = pl.concat(frames, how="vertical_relaxed")
    return out.sort(["team_type", "snapshot_date", "team_key", "slot"])
