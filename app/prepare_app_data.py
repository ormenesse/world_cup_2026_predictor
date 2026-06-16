#!/usr/bin/env python3
"""Prepara os dados leves usados pelo app Streamlit (roda 1x, offline).

Gera em `app/app_data/`:
  • players_pool.parquet   — pool de jogadores do snapshot MAIS RECENTE do FIFA
    (dedup por nome, melhor overall), com `nationality` — usado p/ escolher
    jogadores DA SELEÇÃO e p/ a busca no app.
  • default_lineups.parquet — XI titular padrão de cada SELEÇÃO (de
    silver_fifa_team_snapshot, snapshot mais completo), carregado ao escolher o país.

Assim o app não precisa varrer a bronze de 10M+ linhas em tempo de execução.

Uso:  cd football_analysis && ../.venv/bin/python -m app.prepare_app_data
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                       # football_analysis
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macros.fifa import FIFA_PLAYER_ATTRS, sector_of  # noqa: E402

_BRONZE = _ROOT / "data" / "bronze_fifa_players" / "part-0.parquet"
_SILVER = _ROOT / "data" / "silver_fifa_team_snapshot" / "part-0.parquet"
_OUT = _HERE / "app_data"


def _primary_position_expr() -> pl.Expr:
    """1ª posição natural do jogador (campo `positions`, lista separada por vírgula)."""
    return (
        pl.col("positions").cast(pl.Utf8).str.split(",").list.first()
        .str.strip_chars().str.to_uppercase().alias("position")
    )


def build_players_pool() -> pl.DataFrame:
    df = pl.read_parquet(_BRONZE)
    latest = df["fifa_update_date"].max()
    df = df.filter(pl.col("fifa_update_date") == latest)
    df = df.with_columns(_primary_position_expr())
    # dedup por nome mantendo o de maior overall (jogador "mais relevante").
    df = df.sort("overall", descending=True).unique(subset=["player_name"], keep="first")
    df = df.with_columns(
        pl.col("position").map_elements(sector_of, return_dtype=pl.Utf8).alias("sector")
    )
    return df.select(
        "player_name", "position", "sector", "nationality", *FIFA_PLAYER_ATTRS
    ).sort(["sector", "overall"], descending=[False, True])


def build_default_lineups() -> pl.DataFrame:
    xi = pl.read_parquet(_SILVER).filter(pl.col("team_type") == "nation")
    # Escolhe, por seleção, o snapshot MAIS COMPLETO (mais jogadores) e, em empate,
    # o mais recente — para que o XI padrão tenha 11 sempre que possível.
    sizes = xi.group_by(["team_key", "snapshot_date"]).agg(pl.len().alias("n"))
    best = (
        sizes.sort(["n", "snapshot_date"], descending=[True, True])
        .group_by("team_key", maintain_order=True)
        .first()
        .select("team_key", "snapshot_date")
    )
    xi = xi.join(best, on=["team_key", "snapshot_date"], how="inner")
    return xi.select(
        "team_name", "slot", "player_name", "position", "sector", *FIFA_PLAYER_ATTRS,
    ).sort(["team_name", "slot"])


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    pool = build_players_pool()
    lineups = build_default_lineups()
    pool.write_parquet(_OUT / "players_pool.parquet")
    lineups.write_parquet(_OUT / "default_lineups.parquet")
    print(f"[app-data] players_pool: {pool.shape} | default_lineups: {lineups.shape}")
    print(f"[app-data] seleções com XI: {lineups['team_name'].n_unique()}")
    print(f"[app-data] salvo em: {_OUT}")


if __name__ == "__main__":
    main()
