import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bootstrap import bootstrap_effects
from config import FIGURES_DIR, MATCHES_FILE, RAW_DIR
from features import build_team_windows
from load_data import load_competition_events
from plots import (
    plot_effect_size_bars,
    plot_real_vs_sim_match_timeline,
    plot_real_vs_sim_momentum_bins,
    plot_simulated_momentum_line,
)
from simulate import simulate_match_shots, simulate_competition_shots

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_dir, "data"))
from statsbomb_data import load_match_ids  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Render null-model visuals: timelines, momentum bins, effect sizes."
    )
    parser.add_argument("--match-id", type=int, default=3869685, help="Match ID")
    parser.add_argument("--smooth-window", type=int, default=3, help="Smoothing window")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap")
    parser.add_argument("--n-boot", type=int, default=None, help="Bootstrap iterations")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    match_ids = load_match_ids(MATCHES_FILE)
    shots_df = load_competition_events(RAW_DIR, match_ids)
    if shots_df.empty:
        print("No events loaded. Aborting.")
        return

    # Build windows on real data
    windows_df = build_team_windows(shots_df)

    # Real match data
    real_shots = shots_df[(shots_df["match_id"] == args.match_id) & (shots_df["is_shot"] == 1)].copy()
    real_windows = windows_df[windows_df["match_id"] == args.match_id].copy()

    # Simulated match data (same match, random timing)
    sim_match_shots = simulate_match_shots(real_shots)
    sim_match_windows = build_team_windows(sim_match_shots)

    # Simulated competition for bin curves / effect sizes
    sim_comp_shots = simulate_competition_shots(shots_df)
    sim_comp_windows = build_team_windows(sim_comp_shots)

    plot_real_vs_sim_match_timeline(
        real_windows,
        real_shots,
        sim_match_windows,
        sim_match_shots,
        match_id=args.match_id,
        smooth_window=args.smooth_window,
        save_path=os.path.join(FIGURES_DIR, f"fig_null_real_vs_sim_{args.match_id}.png"),
    )

    plot_simulated_momentum_line(
        sim_match_windows,
        match_id=args.match_id,
        smooth_window=args.smooth_window,
        save_path=os.path.join(FIGURES_DIR, f"fig_null_sim_momentum_{args.match_id}.png"),
    )

    plot_real_vs_sim_momentum_bins(
        windows_df,
        sim_comp_windows,
        save_path=os.path.join(FIGURES_DIR, "fig_null_bins_shot_next_2.png"),
    )

    bootstrap_df = None
    if not args.skip_bootstrap:
        if args.n_boot is None:
            bootstrap_df = bootstrap_effects(windows_df, shots_df)
        else:
            bootstrap_df = bootstrap_effects(windows_df, shots_df, n_boot=args.n_boot)

    plot_effect_size_bars(
        windows_df,
        sim_comp_windows,
        bootstrap_df=bootstrap_df,
        save_path=os.path.join(FIGURES_DIR, "fig_null_effect_sizes.png"),
    )


if __name__ == "__main__":
    main()
