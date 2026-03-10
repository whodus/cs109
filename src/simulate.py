import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INTERIM_DIR, MATCH_END
from features import build_team_windows


def simulate_team_shots(observed_team_shots: pd.DataFrame,
                        match_end: int = MATCH_END,
                        rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Homogeneous Poisson null model for one team in one match.

    - Keeps the same number of shots N.
    - Redistributes them uniformly over [0, match_end).
    - Re-assigns xG by sampling with replacement from observed xG values.
    - Draws is_goal ~ Bernoulli(xg).

    Returns DataFrame with columns: match_id, team_name, event_time_min, shot_xg, is_goal
    Skips (returns empty DataFrame) if team has zero shots.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(observed_team_shots) == 0:
        return pd.DataFrame(columns=["match_id", "team_name", "event_time_min", "shot_xg", "is_goal"])

    n = len(observed_team_shots)
    match_id = observed_team_shots["match_id"].iloc[0]
    team_name = observed_team_shots["team_name"].iloc[0]

    # Uniform times
    times = rng.uniform(0, match_end, size=n)

    # Sample xG with replacement from observed
    observed_xg = observed_team_shots["shot_xg"].fillna(0.0).values
    xg_vals = rng.choice(observed_xg, size=n, replace=True)

    # Bernoulli goals
    goals = rng.binomial(1, xg_vals).astype(int)

    return pd.DataFrame({
        "match_id": match_id,
        "team_name": team_name,
        "event_time_min": times,
        "shot_xg": xg_vals,
        "is_goal": goals,
        "is_shot": 1,
    })


def simulate_match_shots(observed_match_shots: pd.DataFrame,
                         rng: np.random.Generator = None) -> pd.DataFrame:
    """Apply simulate_team_shots per team for one match."""
    if rng is None:
        rng = np.random.default_rng()

    parts = []
    for team_name, team_df in observed_match_shots.groupby("team_name"):
        sim = simulate_team_shots(team_df, rng=rng)
        if not sim.empty:
            parts.append(sim)

    if not parts:
        return pd.DataFrame(columns=["match_id", "team_name", "event_time_min", "shot_xg", "is_goal", "is_shot"])
    return pd.concat(parts, ignore_index=True)


def simulate_competition_shots(shots_df: pd.DataFrame,
                                rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Apply null-model simulation across all matches.

    Returns sim_shots_df with same schema as shots_df (subset of columns).
    """
    if rng is None:
        rng = np.random.default_rng()

    shot_only = shots_df[shots_df["is_shot"] == 1].copy()
    shot_only["shot_xg"] = shot_only["shot_xg"].fillna(0.0)

    parts = []
    for match_id, match_df in shot_only.groupby("match_id"):
        sim = simulate_match_shots(match_df, rng=rng)
        if not sim.empty:
            parts.append(sim)

    if not parts:
        return pd.DataFrame(columns=["match_id", "team_name", "event_time_min", "shot_xg", "is_goal", "is_shot"])
    return pd.concat(parts, ignore_index=True)


def run_simulation(shots_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Run null-model simulation and build sim_team_windows_df.

    Saves to data/interim/simulated_team_windows.csv.
    Returns sim_team_windows_df.
    """
    rng = np.random.default_rng(seed)
    sim_shots_df = simulate_competition_shots(shots_df, rng=rng)
    sim_team_windows_df = build_team_windows(sim_shots_df)

    os.makedirs(INTERIM_DIR, exist_ok=True)
    out_path = os.path.join(INTERIM_DIR, "simulated_team_windows.csv")
    sim_team_windows_df.to_csv(out_path, index=False)
    print(f"Saved simulated_team_windows ({len(sim_team_windows_df)} rows) → {out_path}")

    return sim_team_windows_df
