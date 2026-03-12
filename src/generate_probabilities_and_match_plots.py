import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FIGURES_DIR, MATCHES_FILE, RAW_DIR, TABLES_DIR
from features import build_team_windows
from load_data import load_competition_events

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_dir, "data"))
from statsbomb_data import load_match_ids  # noqa: E402


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def _quintile_table(df: pd.DataFrame, factor: str, outcome: str) -> pd.DataFrame:
    tmp = df[[factor, outcome]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[factor], q=5, labels=False, duplicates="drop")
    grp = tmp.groupby("bin")[outcome].agg(["mean", "count"]).reset_index()
    grp["bin"] = grp["bin"].astype(int) + 1
    grp = grp.rename(columns={"mean": "P_outcome", "count": "n_rows"})
    return grp


def _count_bin_table(df: pd.DataFrame, factor: str, outcome: str) -> pd.DataFrame:
    tmp = df[[factor, outcome]].dropna().copy()
    tmp["bin"] = tmp[factor].clip(upper=2).astype(int)
    grp = tmp.groupby("bin")[outcome].agg(["mean", "count"]).reset_index()
    grp["bin"] = grp["bin"].map({0: "0", 1: "1", 2: "2+"})
    grp = grp.rename(columns={"mean": "P_outcome", "count": "n_rows"})
    return grp


def _delta_from_table(tbl: pd.DataFrame) -> float:
    if len(tbl) < 2:
        return float("nan")
    low = tbl.iloc[0]["P_outcome"]
    high = tbl.iloc[-1]["P_outcome"]
    return float(high - low)


def compute_probability_tables(windows_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["shot_next_2", "shot_next_5"]
    factors_quintile = ["rolling_xg_5", "shots_5", "xg_diff_5"]
    factors_count = ["set_piece_shots_5", "big_chances_5"]

    summaries = []

    for outcome in outcomes:
        for factor in factors_quintile:
            tbl = _quintile_table(windows_df, factor, outcome)
            out_path = os.path.join(
                TABLES_DIR, f"prob_{outcome}_by_{factor}_quintile.csv"
            )
            tbl.to_csv(out_path, index=False)
            summaries.append(
                {
                    "factor": factor,
                    "outcome": outcome,
                    "binning": "quintile",
                    "delta_high_minus_low": _delta_from_table(tbl),
                }
            )

        for factor in factors_count:
            if factor not in windows_df.columns:
                continue
            tbl = _count_bin_table(windows_df, factor, outcome)
            out_path = os.path.join(
                TABLES_DIR, f"prob_{outcome}_by_{factor}_bins.csv"
            )
            tbl.to_csv(out_path, index=False)
            summaries.append(
                {
                    "factor": factor,
                    "outcome": outcome,
                    "binning": "0/1/2+",
                    "delta_high_minus_low": _delta_from_table(tbl),
                }
            )

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(TABLES_DIR, "probability_deltas_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def plot_match_momentum_timeline(
    windows_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    match_id: int,
    smooth_window: int = 3,
    save_path: str = None,
):
    import matplotlib.pyplot as plt

    match_windows = windows_df[windows_df["match_id"] == match_id].copy()
    match_shots = shots_df[
        (shots_df["match_id"] == match_id) & (shots_df["is_shot"] == 1)
    ].copy()
    if match_windows.empty or match_shots.empty:
        print(f"Match {match_id}: no data; skipping plot.")
        return

    # Momentum score
    for col in ["rolling_xg_5", "shots_5", "big_chances_5", "set_piece_shots_5"]:
        if col not in match_windows.columns:
            raise ValueError(f"Missing required column: {col}")

    match_windows["momentum"] = (
        match_windows["rolling_xg_5"]
        + 0.1 * match_windows["shots_5"]
        + 0.3 * match_windows["big_chances_5"]
        + 0.2 * match_windows["set_piece_shots_5"]
    )

    teams = list(match_windows["team_name"].unique())
    if len(teams) < 2:
        print(f"Match {match_id}: fewer than 2 teams; skipping plot.")
        return

    # Single-axis plot: momentum lines with shots/goals overlaid
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e"]
    team_color = {t: colors[i % 2] for i, t in enumerate(teams)}

    for team in teams:
        team_df = match_windows[match_windows["team_name"] == team].sort_values(
            "t_minute"
        )
        smoothed = _smooth_series(team_df["momentum"].values, smooth_window)
        ax.plot(
            team_df["t_minute"].values,
            smoothed,
            label=f"{team} momentum",
            color=team_color[team],
            lw=2,
        )

    # Place shot/goal markers near the bottom of the plot
    ymin, ymax = ax.get_ylim()
    base = ymin + (ymax - ymin) * 0.05
    for team in teams:
        team_shots = match_shots[match_shots["team_name"] == team]
        ax.scatter(
            team_shots["event_time_min"],
            np.full(len(team_shots), base),
            s=team_shots["shot_xg"].fillna(0.05) * 500,
            color=team_color[team],
            alpha=0.6,
            marker="o",
            label=f"{team} shot",
        )
        goals = team_shots[team_shots["is_goal"] == 1]
        ax.scatter(
            goals["event_time_min"],
            np.full(len(goals), base),
            s=goals["shot_xg"].fillna(0.1) * 700,
            color=team_color[team],
            marker="*",
            edgecolors="black",
            zorder=5,
            label=f"{team} goal",
        )

    ax.set_xlabel("Minute")
    ax.set_ylabel("Momentum (weighted sum)")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", ncol=2)

    plt.tight_layout()
    out = save_path or os.path.join(FIGURES_DIR, f"match_{match_id}_momentum.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute probability tables and render match momentum timelines."
    )
    parser.add_argument(
        "--match-ids",
        type=int,
        nargs="+",
        default=[3869685, 3857300, 3857259],
        help="Match IDs to plot (space-separated)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Smoothing window (minutes) for momentum line",
    )

    args = parser.parse_args()

    _ensure_dirs()

    match_ids = load_match_ids(MATCHES_FILE)
    shots_df = load_competition_events(RAW_DIR, match_ids)
    if shots_df.empty:
        print("No events loaded. Aborting.")
        return

    windows_df = build_team_windows(shots_df)

    summary_df = compute_probability_tables(windows_df)
    print("Saved probability tables and summary.")
    print(summary_df.head())

    for mid in args.match_ids:
        plot_match_momentum_timeline(
            windows_df,
            shots_df,
            match_id=mid,
            smooth_window=args.smooth_window,
            save_path=os.path.join(FIGURES_DIR, f"match_{mid}_momentum.png"),
        )


if __name__ == "__main__":
    main()
