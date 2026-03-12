import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bootstrap import bootstrap_effects
from config import FIGURES_DIR, MATCHES_FILE, RAW_DIR
from features import build_team_windows
from load_data import load_competition_events
from plots import (
    plot_bootstrap_intervals,
    plot_match_timeline,
    plot_momentum_bins,
    plot_real_vs_sim,
)
from simulate import run_simulation

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_dir, "data"))
from statsbomb_data import load_match_ids  # noqa: E402


def _figure_path(name: str, suffix: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if suffix:
        return os.path.join(FIGURES_DIR, f"{name}_{suffix}.png")
    return os.path.join(FIGURES_DIR, f"{name}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Render project visuals with optional text-free styling."
    )
    parser.add_argument("--match-id", type=int, default=None, help="Match ID for timeline figure")
    parser.add_argument("--with-text", action="store_true", help="Include titles/labels/legends")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap-based figures")
    parser.add_argument("--n-boot", type=int, default=None, help="Override bootstrap iterations")
    parser.add_argument("--suffix", type=str, default="clean", help="Suffix for output filenames")

    args = parser.parse_args()

    match_ids = load_match_ids(MATCHES_FILE)
    shots_df = load_competition_events(RAW_DIR, match_ids)
    if shots_df.empty:
        print("No events loaded. Aborting.")
        return

    windows_df = build_team_windows(shots_df)

    # Choose a match ID if not provided
    match_id = args.match_id
    if match_id is None:
        match_id = int(shots_df["match_id"].iloc[0])

    plot_match_timeline(
        shots_df,
        match_id,
        save_path=_figure_path(f"fig1_timeline_{match_id}", args.suffix),
        with_text=args.with_text,
    )

    plot_momentum_bins(
        windows_df,
        save_path=_figure_path("fig2_momentum_bins", args.suffix),
        with_text=args.with_text,
    )

    # Simulation + bootstrap-based figures
    sim_windows_df = run_simulation(shots_df)

    if not args.skip_bootstrap:
        if args.n_boot is None:
            bootstrap_df = bootstrap_effects(windows_df, shots_df)
        else:
            bootstrap_df = bootstrap_effects(windows_df, shots_df, n_boot=args.n_boot)

        plot_real_vs_sim(
            windows_df,
            sim_windows_df,
            bootstrap_df,
            save_path=_figure_path("fig4_real_vs_sim", args.suffix),
            with_text=args.with_text,
        )

        plot_bootstrap_intervals(
            bootstrap_df,
            save_path=_figure_path("fig5_bootstrap_intervals", args.suffix),
            with_text=args.with_text,
        )
    else:
        print("Skipped bootstrap; fig4 and fig5 not generated.")


if __name__ == "__main__":
    main()
