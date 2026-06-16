"""Bronze (FIFA) — resultados de partida limpos (schedule → dimensão de partida).

Lê o flatfile `fifa_schedule` (gerado a partir de
`data/raw/fifa_matches/schedule.parquet`) e produz uma linha por partida com:
  • `match_id` sintético e estável (competição|temporada|data|casa|fora);
  • `match_date` (Date), placar inteiro, e só jogos DISPUTADOS a partir da data
    mínima do `fifa_match_sources.yaml`;
  • `result` (H/D/A) e `target`/`home_result`/`away_result` (W/D/L na ótica de
    cada lado);
  • `home_key`/`away_key` normalizados (casamento de nome com o FIFA na gold);
  • metadados da competição (`competition_code`, `competition_group`, `match_type`).
"""
import polars as pl

from macros.fifa import fifa_min_match_date, normalize_team_name


def process_data(self, input_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    sched = input_tables["schedule"]

    # match_date pode vir como Date (parquet) ou string; normaliza para Date.
    if sched.schema.get("match_date") == pl.Utf8:
        sched = sched.with_columns(
            pl.col("match_date").str.to_datetime(strict=False).dt.date().alias("match_date")
        )
    else:
        sched = sched.with_columns(pl.col("match_date").cast(pl.Date, strict=False))

    sched = sched.with_columns(
        pl.col("home_goals").cast(pl.Float64, strict=False),
        pl.col("away_goals").cast(pl.Float64, strict=False),
    ).filter(
        pl.col("match_date").is_not_null()
        & (pl.col("match_date") >= pl.lit(fifa_min_match_date()).str.to_date())
        & pl.col("home_goals").is_not_null()
        & pl.col("away_goals").is_not_null()  # só jogos disputados
    )

    sched = sched.with_columns(
        pl.col("home_goals").cast(pl.Int64),
        pl.col("away_goals").cast(pl.Int64),
    ).with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals")).then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals")).then(pl.lit("A"))
        .otherwise(pl.lit("D")).alias("result"),
    ).with_columns(
        # ótica do mandante (igual à gold de futebol): W/D/L
        pl.when(pl.col("result") == "H").then(pl.lit("W"))
        .when(pl.col("result") == "A").then(pl.lit("L"))
        .otherwise(pl.lit("D")).alias("target"),
        pl.when(pl.col("result") == "A").then(pl.lit("W"))
        .when(pl.col("result") == "H").then(pl.lit("L"))
        .otherwise(pl.lit("D")).alias("away_result"),
    ).with_columns(
        pl.col("target").alias("home_result"),
    )

    # match_id sintético e estável (o schedule não traz id).
    sched = sched.with_columns(
        pl.concat_str(
            [
                pl.col("competition_code").cast(pl.Utf8),
                pl.col("season").cast(pl.Utf8),
                pl.col("match_date").cast(pl.Utf8),
                pl.col("home_team").cast(pl.Utf8),
                pl.col("away_team").cast(pl.Utf8),
            ],
            separator="|",
        ).alias("match_id"),
        pl.col("home_team").map_elements(normalize_team_name, return_dtype=pl.Utf8).alias("home_key"),
        pl.col("away_team").map_elements(normalize_team_name, return_dtype=pl.Utf8).alias("away_key"),
    )

    return sched.select(
        "match_id",
        "match_date",
        "competition_code",
        "competition_group",
        "match_type",
        "season",
        "home_team",
        "home_key",
        "away_team",
        "away_key",
        "home_goals",
        "away_goals",
        "result",
        "target",
        "home_result",
        "away_result",
    ).unique(subset=["match_id"])
