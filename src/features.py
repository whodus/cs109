import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INTERIM_DIR, LOOKBACK, MAX_MINUTE, MIN_MINUTE


def compute_game_state(goals_before_t: pd.DataFrame, team_name: str) -> str:
    """
    Compute game state for a team at a given moment.

    goals_before_t: DataFrame of goal events strictly before time t,
                    must have 'team_name' column.
    Returns: "leading", "tied", or "trailing"
    """
    team_goals = (goals_before_t["team_name"] == team_name).sum()
    opp_goals = (goals_before_t["team_name"] != team_name).sum()
    if team_goals > opp_goals:
        return "leading"
    elif team_goals < opp_goals:
        return "trailing"
    else:
        return "tied"


def _rolling_xg(shots_df: pd.DataFrame, match_id: int, team_name: str,
                 t_start: float, t_end: float) -> tuple:
    """Return (sum_xg, shot_count) for team in [t_start, t_end)."""
    mask = (
        (shots_df["match_id"] == match_id) &
        (shots_df["team_name"] == team_name) &
        (shots_df["event_time_min"] >= t_start) &
        (shots_df["event_time_min"] < t_end)
    )
    sub = shots_df.loc[mask]
    return float(sub["shot_xg"].sum()), int(len(sub))


_CORNER_PATTERNS = {"From Corner"}
_SET_PIECE_PATTERNS = {"From Corner", "From Free Kick"}
_BIG_CHANCE_XG = 0.30


def _supp_features(match_shots: pd.DataFrame, team_name: str,
                   t_start: float, t_end: float) -> dict:
    """
    Compute supplementary pressure features for one team in [t_start, t_end).

    Requires match_shots to have 'play_pattern_name' and 'shot_xg' columns.
    Returns dict of 6 feature values.
    """
    mask = (
        (match_shots["team_name"] == team_name) &
        (match_shots["event_time_min"] >= t_start) &
        (match_shots["event_time_min"] < t_end)
    )
    sub = match_shots.loc[mask]

    corner = int((sub["play_pattern_name"].isin(_CORNER_PATTERNS)).sum())
    set_piece = int((sub["play_pattern_name"].isin(_SET_PIECE_PATTERNS)).sum())
    big_chance = int((sub["shot_xg"] >= _BIG_CHANCE_XG).sum())

    return {
        "corner_shots_5": corner,
        "set_piece_shots_5": set_piece,
        "big_chances_5": big_chance,
        "recent_corner_pressure": 1 if corner > 0 else 0,
        "recent_set_piece_pressure": 1 if set_piece > 0 else 0,
        "recent_big_chance_pressure": 1 if big_chance > 0 else 0,
    }


def build_team_windows(shots_df: pd.DataFrame,
                       lookback: int = LOOKBACK,
                       max_minute: int = MAX_MINUTE) -> pd.DataFrame:
    """
    Build team-minute window feature table.

    For each match × team × t in range(MIN_MINUTE, max_minute+1):
      - rolling_xg_5: sum of team shot xG in [t-lookback, t)
      - shots_5: count of team shots in [t-lookback, t)
      - opp_rolling_xg_5, opp_shots_5: same for opponent
      - xg_diff_5 = rolling_xg_5 - opp_rolling_xg_5
      - shot_next_2: 1 if team has ≥1 shot in [t, t+2)
      - goal_next_5: 1 if team has ≥1 goal in [t, t+5)
      - game_state: leading / tied / trailing
      - half: 1 if t < 45 else 2
      - match_score_for, match_score_against
    """
    if shots_df.empty:
        return pd.DataFrame()

    # Work only with shot rows (is_shot == 1) — keep shot_xg valid
    shot_rows = shots_df[shots_df["is_shot"] == 1].copy()
    shot_rows["shot_xg"] = shot_rows["shot_xg"].fillna(0.0)

    # Supplementary features require play_pattern_name (absent in simulated data)
    has_supp = "play_pattern_name" in shot_rows.columns

    # Also keep goal rows for game_state computation
    goal_rows = shots_df[shots_df["is_goal"] == 1].copy()

    rows = []
    match_ids = shot_rows["match_id"].unique() if len(shot_rows) > 0 else shots_df["match_id"].unique()

    for match_id in match_ids:
        match_shots = shot_rows[shot_rows["match_id"] == match_id]
        match_goals = goal_rows[goal_rows["match_id"] == match_id]

        # Identify teams — use all events if shots_df includes non-shot rows
        teams_in_shots = list(match_shots["team_name"].unique())
        all_match_rows = shots_df[shots_df["match_id"] == match_id]
        teams_all = list(all_match_rows["team_name"].unique())
        # Use teams_all if we have at least 2 teams; else fall back
        teams = list(dict.fromkeys(teams_all))  # preserve order, deduplicate
        if len(teams) < 2:
            teams = teams_in_shots
        if len(teams) < 2:
            continue

        # Final match scores
        final_goals = match_goals
        for i, team_name in enumerate(teams):
            opp_name = teams[1 - i] if len(teams) == 2 else [t for t in teams if t != team_name][0]

            match_score_for = int((final_goals["team_name"] == team_name).sum())
            match_score_against = int((final_goals["team_name"] == opp_name).sum())

            for t in range(MIN_MINUTE, max_minute + 1):
                t_float = float(t)

                # Rolling xG and shots in lookback window
                rxg, shots5 = _rolling_xg(match_shots, match_id, team_name,
                                           t_float - lookback, t_float)
                opp_rxg, opp_shots5 = _rolling_xg(match_shots, match_id, opp_name,
                                                    t_float - lookback, t_float)

                # Outcome windows
                shot_next_2_mask = (
                    (match_shots["team_name"] == team_name) &
                    (match_shots["event_time_min"] >= t_float) &
                    (match_shots["event_time_min"] < t_float + 2)
                )
                shot_next_2 = 1 if shot_next_2_mask.any() else 0

                goal_next_5_mask = (
                    (match_goals["team_name"] == team_name) &
                    (match_goals["event_time_min"] >= t_float) &
                    (match_goals["event_time_min"] < t_float + 5)
                )
                goal_next_5 = 1 if goal_next_5_mask.any() else 0

                # Game state: goals strictly before t
                goals_before = match_goals[match_goals["event_time_min"] < t_float]
                game_state = compute_game_state(goals_before, team_name)

                row = {
                    "match_id": match_id,
                    "team_name": team_name,
                    "opponent_name": opp_name,
                    "t_minute": t,
                    "rolling_xg_5": rxg,
                    "shots_5": shots5,
                    "opp_rolling_xg_5": opp_rxg,
                    "opp_shots_5": opp_shots5,
                    "xg_diff_5": rxg - opp_rxg,
                    "shot_next_2": shot_next_2,
                    "goal_next_5": goal_next_5,
                    "game_state": game_state,
                    "half": 1 if t < 45 else 2,
                    "match_score_for": match_score_for,
                    "match_score_against": match_score_against,
                }

                if has_supp:
                    row.update(_supp_features(match_shots, team_name,
                                              t_float - lookback, t_float))

                rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Validation
    dupes = df.duplicated(subset=["match_id", "team_name", "t_minute"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (match_id, team_name, t_minute) rows"
    assert (df["rolling_xg_5"] >= 0).all(), "rolling_xg_5 has negative values"
    assert (df["shots_5"] >= 0).all(), "shots_5 has negative values"
    assert df["shot_next_2"].isin([0, 1]).all(), "shot_next_2 not in {0,1}"
    assert df["goal_next_5"].isin([0, 1]).all(), "goal_next_5 not in {0,1}"

    return df


def save_interim(shots_df: pd.DataFrame, windows_df: pd.DataFrame):
    """Save shots and team_windows to data/interim/."""
    os.makedirs(INTERIM_DIR, exist_ok=True)
    shots_path = os.path.join(INTERIM_DIR, "shots.csv")
    windows_path = os.path.join(INTERIM_DIR, "team_windows.csv")
    shots_df.to_csv(shots_path, index=False)
    windows_df.to_csv(windows_path, index=False)
    print(f"Saved shots ({len(shots_df)} rows) → {shots_path}")
    print(f"Saved team_windows ({len(windows_df)} rows) → {windows_path}")
