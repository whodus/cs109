import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import N_BOOT, BOOT_SEED, PROCESSED_DIR
from simulate import simulate_competition_shots
from features import build_team_windows
from utils import compute_delta


def bootstrap_effects(team_windows_df: pd.DataFrame,
                      shots_df: pd.DataFrame,
                      n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    Match-level bootstrap: resample matches with replacement, n_boot iterations.

    For each iteration:
      1. Sample match IDs with replacement.
      2. Subset team_windows_df and shots_df.
      3. Re-run simulation on sampled shots → sim_tw.
      4. Compute quintile deltas on real and sim data.
      5. Record delta_shot_real, delta_goal_real, delta_shot_sim, delta_goal_sim,
         gap_shot = delta_shot_real - delta_shot_sim, gap_goal.

    Returns DataFrame with columns:
        iter, delta_shot_real, delta_goal_real, delta_shot_sim, delta_goal_sim,
        gap_shot, gap_goal
    """
    rng = np.random.default_rng(seed)
    all_match_ids = np.array(team_windows_df["match_id"].unique())
    records = []

    for i in range(n_boot):
        # Sample match IDs with replacement
        sampled_ids = rng.choice(all_match_ids, size=len(all_match_ids), replace=True)

        # Subset real data (allow duplicates by iterating over sampled IDs)
        tw_parts = []
        sh_parts = []
        for mid in sampled_ids:
            tw_parts.append(team_windows_df[team_windows_df["match_id"] == mid])
            sh_parts.append(shots_df[shots_df["match_id"] == mid])

        tw_boot = pd.concat(tw_parts, ignore_index=True)
        sh_boot = pd.concat(sh_parts, ignore_index=True)

        # Re-run simulation on sampled shots
        iter_rng = np.random.default_rng(rng.integers(0, 2**32))
        sim_shots = simulate_competition_shots(sh_boot, rng=iter_rng)
        sim_tw = build_team_windows(sim_shots)

        # Compute deltas
        delta_shot_real = compute_delta(tw_boot, "shot_next_2")
        delta_goal_real = compute_delta(tw_boot, "goal_next_5")
        delta_shot_sim = compute_delta(sim_tw, "shot_next_2") if not sim_tw.empty else np.nan
        delta_goal_sim = compute_delta(sim_tw, "goal_next_5") if not sim_tw.empty else np.nan

        gap_shot = delta_shot_real - delta_shot_sim
        gap_goal = delta_goal_real - delta_goal_sim

        records.append({
            "iter": i,
            "delta_shot_real": delta_shot_real,
            "delta_goal_real": delta_goal_real,
            "delta_shot_sim": delta_shot_sim,
            "delta_goal_sim": delta_goal_sim,
            "gap_shot": gap_shot,
            "gap_goal": gap_goal,
        })

        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_boot}")

    bootstrap_df = pd.DataFrame(records)

    # Save results
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "bootstrap_results.json")
    bootstrap_df.to_json(out_path, orient="records", indent=2)
    print(f"Saved bootstrap_results ({len(bootstrap_df)} rows) → {out_path}")

    return bootstrap_df
