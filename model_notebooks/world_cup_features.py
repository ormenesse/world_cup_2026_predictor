from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TEAM_NAME_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde Islands",
    "Curaçao": "Curacao",
}


@dataclass
class TeamFeatureStore:
    team_features: dict[str, pd.Series]
    template_row: pd.Series
    feature_columns: list[str]
    team_name_aliases: dict[str, str]
    fallback_features: pd.Series


def get_training_feature_columns(gold_df: pd.DataFrame) -> list[str]:
    """Rebuild the exact feature list used in the notebook training cell."""
    do_not_use = ["year_month", "month_index", "home_score", "away_score"]
    thresh = len(gold_df) * 0.20
    return [
        col
        for col in gold_df.dropna(axis=1, thresh=thresh)
        .select_dtypes(include="number")
        .columns.tolist()
        if col not in do_not_use and "mean" in col
    ]


def _to_team_perspective(
    match_row: pd.Series,
    side: str,
    allowed_side_columns: set[str],
) -> pd.Series:
    """Convert a historical row into side-agnostic team features."""
    prefix = f"{side}_"
    values = {}
    for col, value in match_row.items():
        if col.startswith(prefix) and col in allowed_side_columns:
            values[col[len(prefix) :]] = value
    return pd.Series(values, dtype="float64")


def build_team_feature_store(
    gold_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    team_name_aliases: dict[str, str] | None = None,
) -> TeamFeatureStore:
    """Build latest-available feature snapshots for each team."""
    if feature_columns is None:
        feature_columns = get_training_feature_columns(gold_df)

    team_name_aliases = {**TEAM_NAME_ALIASES, **(team_name_aliases or {})}

    hist = gold_df.sort_values("match_date").copy()
    feature_column_set = set(feature_columns)

    side_agnostic_columns = sorted(
        {
            col[len("home_") :]
            for col in feature_columns
            if col.startswith("home_")
        }
        | {
            col[len("away_") :]
            for col in feature_columns
            if col.startswith("away_")
        }
    )
    template_row = pd.Series(0.0, index=side_agnostic_columns, dtype="float64")

    team_features: dict[str, pd.Series] = {}

    for team_col, side in [("home_team", "home"), ("away_team", "away")]:
        latest_rows = hist.dropna(subset=[team_col]).groupby(team_col, sort=False).tail(1)
        for _, row in latest_rows.iterrows():
            team = str(row[team_col])
            snapshot = template_row.copy()
            snapshot.update(_to_team_perspective(row, side, feature_column_set))
            team_features[team] = snapshot

    return TeamFeatureStore(
        team_features=team_features,
        template_row=template_row,
        feature_columns=feature_columns,
        team_name_aliases=team_name_aliases,
        fallback_features=pd.DataFrame(team_features).T.median(axis=0).astype("float64"),
    )


def _resolve_team_name(team: str, store: TeamFeatureStore) -> str | None:
    resolved = store.team_name_aliases.get(team, team)
    return resolved if resolved in store.team_features else None


def prepare_match_row(
    home_team: str,
    away_team: str,
    store: TeamFeatureStore,
    fillna_value: float = 0.0,
    use_fallback_for_missing: bool = True,
) -> pd.DataFrame:
    """Create one prediction row aligned to the LightGBM training columns."""
    home_key = _resolve_team_name(home_team, store)
    away_key = _resolve_team_name(away_team, store)
    if not use_fallback_for_missing and (home_key is None or away_key is None):
        missing = [team for team, key in [(home_team, home_key), (away_team, away_key)] if key is None]
        raise KeyError(f"No historical features found for: {missing}")

    home_features = store.team_features[home_key] if home_key else store.fallback_features
    away_features = store.team_features[away_key] if away_key else store.fallback_features

    row = {}
    for col in store.feature_columns:
        if col.startswith("home_"):
            row[col] = home_features.get(col[len("home_") :], fillna_value)
        elif col.startswith("away_"):
            row[col] = away_features.get(col[len("away_") :], fillna_value)
        else:
            row[col] = fillna_value

    match_df = pd.DataFrame([row], columns=store.feature_columns)
    return match_df.fillna(fillna_value)


def prepare_world_cup_matches(
    fixtures_df: pd.DataFrame,
    store: TeamFeatureStore,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    match_number_col: str = "MatchNumber",
    use_fallback_for_missing: bool = True,
) -> pd.DataFrame:
    """Create one model-ready row per World Cup fixture."""
    rows = []
    metadata = []

    for _, fixture in fixtures_df.iterrows():
        home_team = fixture[home_col]
        away_team = fixture[away_col]

        if any(str(team).startswith(("1", "2", "3")) for team in [home_team, away_team]):
            continue
        if any(team in {"To be announced", None} for team in [home_team, away_team]):
            continue

        match_row = prepare_match_row(
            home_team=home_team,
            away_team=away_team,
            store=store,
            use_fallback_for_missing=use_fallback_for_missing,
        )
        rows.append(match_row.iloc[0])
        metadata.append(
            {
                "MatchNumber": fixture.get(match_number_col),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "RoundNumber": fixture.get("RoundNumber"),
                "Group": fixture.get("Group"),
                "DateUtc": fixture.get("DateUtc"),
                "home_team_resolved": store.team_name_aliases.get(home_team, home_team),
                "away_team_resolved": store.team_name_aliases.get(away_team, away_team),
                "home_team_has_history": _resolve_team_name(home_team, store) is not None,
                "away_team_has_history": _resolve_team_name(away_team, store) is not None,
            }
        )

    features_df = pd.DataFrame(rows).reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata)
    return pd.concat([metadata_df, features_df], axis=1)
