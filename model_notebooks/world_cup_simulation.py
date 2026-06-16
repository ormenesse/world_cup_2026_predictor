from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from football_analysis.model_notebooks.world_cup_features import prepare_match_row
except ModuleNotFoundError:
    from world_cup_features import prepare_match_row


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    result: str
    winner: str
    most_likely_result: str
    most_likely_winner: str
    proba_W: float
    proba_D: float
    proba_L: float
    # Placar previsto pelos modelos de gols (None se os modelos não forem passados).
    home_goals: int | None = None
    away_goals: int | None = None


def _model_proba(model, X):
    """Probabilidades do modelo, aceitando LGBMClassifier (predict_proba) OU
    Booster LightGBM (predict devolve probabilidades no multiclasse)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


def _apply_feature_overrides(x_match, home_overrides=None, away_overrides=None):
    """Sobrescreve características de feature por lado, ex.: forma (*3*).

    `*_overrides` é um dict {feature_side_agnostico: valor}, ex.:
    {"win_rate_last10": 0.9}. Aplica em `home_<feat>` / `away_<feat>` quando a
    coluna existe no row de previsão. Útil para simular cenários ("e se a
    seleção X chegasse embalada?") sem reprocessar a gold.
    """
    for side, overrides in (("home", home_overrides), ("away", away_overrides)):
        if not overrides:
            continue
        for feat, value in overrides.items():
            col = f"{side}_{feat}"
            if col in x_match.columns:
                x_match.loc[:, col] = value
    return x_match


def _match_feature_row(home_team, away_team, store, home_overrides=None, away_overrides=None):
    """Linha de features (alinhada a store.feature_columns) p/ uma partida."""
    x = prepare_match_row(home_team, away_team, store)
    x = _apply_feature_overrides(x, home_overrides, away_overrides)
    return x[store.feature_columns].astype("float32")


def predict_match_proba(
    home_team,
    away_team,
    model,
    store,
    label_encoder=None,
    home_overrides=None,
    away_overrides=None,
):
    x_match = _match_feature_row(home_team, away_team, store, home_overrides, away_overrides)
    proba = _model_proba(model, x_match)[0]
    classes = (list(label_encoder.classes_) if label_encoder is not None
               else list(getattr(model, "classes_", range(len(proba)))))
    return dict(zip(classes, proba))


def predict_goal_distributions(
    home_team,
    away_team,
    store,
    home_goals_model,
    away_goals_model,
    home_labels,
    away_labels,
    home_overrides=None,
    away_overrides=None,
):
    """Distribuição de gols (0..5+) de cada lado + placar mais provável e esperado.

    `home_goals_model`/`away_goals_model` são os 2 modelos de placar (Booster ou
    LGBMClassifier). `*_labels` são os rótulos das classes (ex.: ['0','1','2','3','4','5+']).
    Devolve dict com `home_dist`, `away_dist` (rótulo→prob), `home_goals`,
    `away_goals` (mais provável, int) e `home_expected`, `away_expected`.
    """
    X = _match_feature_row(home_team, away_team, store, home_overrides, away_overrides).values
    hp = np.asarray(_model_proba(home_goals_model, X)[0], dtype=float)
    ap = np.asarray(_model_proba(away_goals_model, X)[0], dtype=float)
    return {
        "home_dist": {lab: float(p) for lab, p in zip(home_labels, hp)},
        "away_dist": {lab: float(p) for lab, p in zip(away_labels, ap)},
        "home_goals": int(np.argmax(hp)),
        "away_goals": int(np.argmax(ap)),
        "home_expected": float(sum(i * p for i, p in enumerate(hp))),
        "away_expected": float(sum(i * p for i, p in enumerate(ap))),
    }


def predict_match_full(
    home_team,
    away_team,
    result_model,
    store,
    label_encoder=None,
    goal_models=None,
    home_overrides=None,
    away_overrides=None,
):
    """Previsão completa de uma partida: probabilidades de resultado (W/D/L) +,
    se `goal_models` for dado, as distribuições de gols e o placar provável.

    `goal_models` = (home_goals_model, away_goals_model, home_labels, away_labels).
    """
    out = {"proba": predict_match_proba(home_team, away_team, result_model, store,
                                        label_encoder, home_overrides, away_overrides)}
    if goal_models:
        hm, am, hl, al = goal_models
        out["goals"] = predict_goal_distributions(
            home_team, away_team, store, hm, am, hl, al, home_overrides, away_overrides
        )
    return out


def _resolve_winner(home_team, away_team, result, proba, rng=None, deterministic=True):
    if result == "W":
        return home_team
    if result == "L":
        return away_team

    home_win_prob = float(proba.get("W", 0.0))
    away_win_prob = float(proba.get("L", 0.0))

    if deterministic:
        return home_team if home_win_prob >= away_win_prob else away_team

    rng = rng or np.random.default_rng()
    if home_win_prob + away_win_prob == 0:
        penalty_home_prob = 0.5
    else:
        penalty_home_prob = home_win_prob / (home_win_prob + away_win_prob)
    return rng.choice([home_team, away_team], p=[penalty_home_prob, 1 - penalty_home_prob])


def sample_match(
    home_team,
    away_team,
    model,
    store,
    label_encoder=None,
    rng=None,
    outcome_mode="most_likely",
    team_feature_overrides=None,
    goal_models=None,
):
    rng = rng or np.random.default_rng()
    team_feature_overrides = team_feature_overrides or {}
    home_ov = team_feature_overrides.get(home_team)
    away_ov = team_feature_overrides.get(away_team)
    proba = predict_match_proba(
        home_team, away_team, model, store, label_encoder,
        home_overrides=home_ov, away_overrides=away_ov,
    )

    # Placar previsto pelos modelos de gols (opcional).
    home_goals = away_goals = None
    if goal_models:
        hm, am, hl, al = goal_models
        gd = predict_goal_distributions(
            home_team, away_team, store, hm, am, hl, al,
            home_overrides=home_ov, away_overrides=away_ov,
        )
        home_goals, away_goals = gd["home_goals"], gd["away_goals"]

    classes = np.array(list(proba.keys()))
    probs = np.array(list(proba.values()), dtype=float)
    probs = probs / probs.sum()

    most_likely_result = str(classes[np.argmax(probs)])
    most_likely_winner = _resolve_winner(
        home_team,
        away_team,
        most_likely_result,
        proba,
        deterministic=True,
    )

    if outcome_mode == "sample":
        result = str(rng.choice(classes, p=probs))
        winner = _resolve_winner(
            home_team,
            away_team,
            result,
            proba,
            rng=rng,
            deterministic=False,
        )
    else:
        result = most_likely_result
        winner = most_likely_winner

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        result=str(result),
        winner=winner,
        most_likely_result=most_likely_result,
        most_likely_winner=most_likely_winner,
        proba_W=float(proba.get("W", 0.0)),
        proba_D=float(proba.get("D", 0.0)),
        proba_L=float(proba.get("L", 0.0)),
        home_goals=home_goals,
        away_goals=away_goals,
    )


def init_group_table(group_fixtures):
    teams = sorted(set(group_fixtures["HomeTeam"]) | set(group_fixtures["AwayTeam"]))
    return pd.DataFrame(
        {
            "team": teams,
            "pts": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
    )


def update_group_table(table, match):
    if match.result == "W":
        home_goals, away_goals = 1, 0
        home_pts, away_pts = 3, 0
        home_w, away_w = 1, 0
        home_d, away_d = 0, 0
        home_l, away_l = 0, 1
    elif match.result == "L":
        home_goals, away_goals = 0, 1
        home_pts, away_pts = 0, 3
        home_w, away_w = 0, 1
        home_d, away_d = 0, 0
        home_l, away_l = 1, 0
    else:
        home_goals, away_goals = 0, 0
        home_pts, away_pts = 1, 1
        home_w, away_w = 0, 0
        home_d, away_d = 1, 1
        home_l, away_l = 0, 0

    for team, gf, ga, pts, wins, draws, losses in [
        (match.home_team, home_goals, away_goals, home_pts, home_w, home_d, home_l),
        (match.away_team, away_goals, home_goals, away_pts, away_w, away_d, away_l),
    ]:
        idx = table["team"] == team
        table.loc[idx, "pts"] += pts
        table.loc[idx, "gf"] += gf
        table.loc[idx, "ga"] += ga
        table.loc[idx, "gd"] = table.loc[idx, "gf"] - table.loc[idx, "ga"]
        table.loc[idx, "wins"] += wins
        table.loc[idx, "draws"] += draws
        table.loc[idx, "losses"] += losses

    return table


def simulate_group_stage(fixtures_df, model, store, label_encoder=None, rng=None, outcome_mode="most_likely", team_feature_overrides=None, goal_models=None):
    rng = rng or np.random.default_rng()
    group_fixtures = fixtures_df[fixtures_df["Group"].notna()].copy()

    standings_rows = []
    match_rows = []

    for group_name, group_df in group_fixtures.groupby("Group", sort=True):
        table = init_group_table(group_df)
        for _, row in group_df.sort_values("MatchNumber").iterrows():
            match = sample_match(
                row["HomeTeam"],
                row["AwayTeam"],
                model,
                store,
                label_encoder,
                rng,
                outcome_mode=outcome_mode,
                team_feature_overrides=team_feature_overrides,
                goal_models=goal_models,
            )
            update_group_table(table, match)
            match_rows.append(
                {
                    "MatchNumber": row["MatchNumber"],
                    "RoundNumber": row["RoundNumber"],
                    "Stage": group_name,
                    "HomeTeam": match.home_team,
                    "AwayTeam": match.away_team,
                    "result": match.result,
                    "winner": match.winner,
                    "most_likely_result": match.most_likely_result,
                    "most_likely_winner": match.most_likely_winner,
                    "proba_W": match.proba_W,
                    "proba_D": match.proba_D,
                    "proba_L": match.proba_L,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                }
            )

        table = table.sort_values(["pts", "gd", "gf", "wins"], ascending=[False, False, False, False]).reset_index(drop=True)
        table["Group"] = group_name
        table["rank"] = np.arange(1, len(table) + 1)
        standings_rows.append(table)

    standings = pd.concat(standings_rows, ignore_index=True)
    matches = pd.DataFrame(match_rows)
    return standings, matches


def rank_third_placed_teams(standings):
    third = standings[standings["rank"] == 3].copy()
    third = third.sort_values(["pts", "gd", "gf", "wins"], ascending=[False, False, False, False]).reset_index(drop=True)
    third["third_rank"] = np.arange(1, len(third) + 1)
    return third


def build_slot_mapping(standings):
    mapping = {}
    for _, row in standings.iterrows():
        group_letter = row["Group"].replace("Group ", "")
        mapping[f"{int(row['rank'])}{group_letter}"] = row["team"]

    third = rank_third_placed_teams(standings)
    top8 = third.head(8)
    qualified_groups = "".join(sorted(g.replace("Group ", "") for g in top8["Group"]))
    mapping[f"3{qualified_groups}"] = top8
    return mapping


def resolve_knockout_team(slot, slot_mapping):
    if slot in slot_mapping:
        value = slot_mapping[slot]
        if isinstance(value, pd.DataFrame):
            raise KeyError(f"Slot '{slot}' points to multiple teams; resolve with third-place pattern logic.")
        return value

    if re.fullmatch(r"[123][A-L]", str(slot)):
        raise KeyError(f"Slot '{slot}' not found in standings.")

    if re.fullmatch(r"3[A-L]{5,6}", str(slot)):
        qualified_groups = str(slot)[1:]
        third_table = slot_mapping.get(f"3{qualified_groups}")
        if third_table is None or third_table.empty:
            raise KeyError(f"No third-place qualification table for pattern '{slot}'.")
        return third_table.iloc[0]["team"]

    return slot


def assign_third_place_slots(round4, third_place_ranking):
    third_slots = pd.unique(
        round4[["HomeTeam", "AwayTeam"]].stack().loc[lambda s: s.astype(str).str.fullmatch(r"3[A-L]{5,6}")]
    ).tolist()
    top_thirds = third_place_ranking.copy()
    top_thirds["group_letter"] = top_thirds["Group"].str.replace("Group ", "", regex=False)

    slot_options = {}
    for slot in third_slots:
        eligible_groups = set(slot[1:])
        slot_options[slot] = top_thirds.loc[
            top_thirds["group_letter"].isin(eligible_groups), "team"
        ].tolist()

    ordered_slots = sorted(third_slots, key=lambda slot: (len(slot_options[slot]), slot))

    def backtrack(idx, assigned, used_teams):
        if idx == len(ordered_slots):
            return assigned

        slot = ordered_slots[idx]
        for team in slot_options[slot]:
            if team in used_teams:
                continue
            result = backtrack(
                idx + 1,
                {**assigned, slot: team},
                used_teams | {team},
            )
            if result is not None:
                return result
        return None

    assigned = backtrack(0, {}, set())
    if assigned is not None:
        return assigned

    # Fallback: if the placeholder patterns do not admit a perfect assignment,
    # keep the simulation running by filling unresolved slots with the best
    # remaining third-placed teams.
    assigned = {}
    remaining_teams = top_thirds["team"].tolist()
    for slot in ordered_slots:
        eligible_remaining = [team for team in slot_options[slot] if team in remaining_teams]
        chosen_team = eligible_remaining[0] if eligible_remaining else remaining_teams[0]
        assigned[slot] = chosen_team
        remaining_teams.remove(chosen_team)

    return assigned


def simulate_knockout_round(fixtures, model, store, label_encoder=None, rng=None, outcome_mode="most_likely", team_feature_overrides=None, goal_models=None):
    rng = rng or np.random.default_rng()
    rows = []
    winners = {}

    for _, row in fixtures.sort_values("MatchNumber").iterrows():
        match = sample_match(
            row["HomeTeam"],
            row["AwayTeam"],
            model,
            store,
            label_encoder,
            rng,
            outcome_mode=outcome_mode,
            team_feature_overrides=team_feature_overrides,
            goal_models=goal_models,
        )
        winners[row["MatchNumber"]] = match.winner
        rows.append(
            {
                "MatchNumber": row["MatchNumber"],
                "RoundNumber": row["RoundNumber"],
                "Stage": row.get("Stage", f"Round {row['RoundNumber']}"),
                "HomeTeam": match.home_team,
                "AwayTeam": match.away_team,
                "result": match.result,
                "winner": match.winner,
                "most_likely_result": match.most_likely_result,
                "most_likely_winner": match.most_likely_winner,
                "proba_W": match.proba_W,
                "proba_D": match.proba_D,
                "proba_L": match.proba_L,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
            }
        )

    return pd.DataFrame(rows), winners


def simulate_world_cup(
    fixtures_df,
    model,
    store,
    label_encoder=None,
    rng_seed=42,
    outcome_mode="most_likely",
    team_feature_overrides=None,
    goal_models=None,
):
    rng = np.random.default_rng(rng_seed)

    standings, group_matches = simulate_group_stage(
        fixtures_df,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )
    slot_mapping = build_slot_mapping(standings)
    third_place_ranking = rank_third_placed_teams(standings)

    round4 = fixtures_df[fixtures_df["RoundNumber"] == 4].copy()
    third_slot_mapping = assign_third_place_slots(round4, third_place_ranking.head(8))

    def resolve_slot(slot):
        if slot in third_slot_mapping:
            return third_slot_mapping[slot]
        return resolve_knockout_team(slot, slot_mapping)

    round4["HomeTeam"] = round4["HomeTeam"].map(resolve_slot)
    round4["AwayTeam"] = round4["AwayTeam"].map(resolve_slot)
    round4["Stage"] = "Round of 32"
    round4_matches, round4_winners = simulate_knockout_round(
        round4,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )

    round5 = fixtures_df[fixtures_df["RoundNumber"] == 5].copy().sort_values("MatchNumber").reset_index(drop=True)
    round5["HomeTeam"] = [round4_winners[m] for m in range(73, 81)]
    round5["AwayTeam"] = [round4_winners[m] for m in range(81, 89)]
    round5["Stage"] = "Round of 16"
    round5_matches, round5_winners = simulate_knockout_round(
        round5,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )

    round6 = fixtures_df[fixtures_df["RoundNumber"] == 6].copy().sort_values("MatchNumber").reset_index(drop=True)
    round6["HomeTeam"] = [round5_winners[89], round5_winners[91], round5_winners[93], round5_winners[95]]
    round6["AwayTeam"] = [round5_winners[90], round5_winners[92], round5_winners[94], round5_winners[96]]
    round6["Stage"] = "Quarterfinal"
    round6_matches, round6_winners = simulate_knockout_round(
        round6,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )

    round7 = fixtures_df[fixtures_df["RoundNumber"] == 7].copy().sort_values("MatchNumber").reset_index(drop=True)
    round7["HomeTeam"] = [round6_winners[97], round6_winners[99]]
    round7["AwayTeam"] = [round6_winners[98], round6_winners[100]]
    round7["Stage"] = "Semifinal"
    round7_matches, round7_winners = simulate_knockout_round(
        round7,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )

    round8 = fixtures_df[fixtures_df["RoundNumber"] == 8].copy().sort_values("MatchNumber").reset_index(drop=True)
    round8["Stage"] = ""
    semifinal_winners = [round7_winners[101], round7_winners[102]]
    semifinal_losers = [
        round7.loc[round7["MatchNumber"] == 101, "AwayTeam"].iloc[0]
        if round7_winners[101] == round7.loc[round7["MatchNumber"] == 101, "HomeTeam"].iloc[0]
        else round7.loc[round7["MatchNumber"] == 101, "HomeTeam"].iloc[0],
        round7.loc[round7["MatchNumber"] == 102, "AwayTeam"].iloc[0]
        if round7_winners[102] == round7.loc[round7["MatchNumber"] == 102, "HomeTeam"].iloc[0]
        else round7.loc[round7["MatchNumber"] == 102, "HomeTeam"].iloc[0],
    ]
    round8.iloc[0, round8.columns.get_loc("HomeTeam")] = semifinal_losers[0]
    round8.iloc[0, round8.columns.get_loc("AwayTeam")] = semifinal_losers[1]
    round8.iloc[0, round8.columns.get_loc("Stage")] = "Third Place"
    round8.iloc[1, round8.columns.get_loc("HomeTeam")] = semifinal_winners[0]
    round8.iloc[1, round8.columns.get_loc("AwayTeam")] = semifinal_winners[1]
    round8.iloc[1, round8.columns.get_loc("Stage")] = "Final"
    round8_matches, _ = simulate_knockout_round(
        round8,
        model,
        store,
        label_encoder,
        rng,
        outcome_mode=outcome_mode,
        team_feature_overrides=team_feature_overrides,
        goal_models=goal_models,
    )

    knockout_matches = pd.concat(
        [round4_matches, round5_matches, round6_matches, round7_matches, round8_matches],
        ignore_index=True,
    )
    champion = round8_matches.loc[round8_matches["Stage"] == "Final", "winner"].iloc[0]

    return {
        "standings": standings,
        "group_matches": group_matches,
        "third_place_ranking": third_place_ranking,
        "knockout_matches": knockout_matches,
        "champion": champion,
    }
