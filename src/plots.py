import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES_DIR
from utils import make_quintile_bins

# Consistent style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _strip_text(ax):
    """Remove titles, labels, tick labels, and legends for clean visuals."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def plot_match_timeline(shots_df: pd.DataFrame,
                        match_id: int,
                        save_path: str = None,
                        with_text: bool = True):
    """
    Fig 1: Timeline of shots for one match.

    x-axis = event_time_min, dot size ∝ shot_xg, goals marked with star.
    """
    _ensure_figures_dir()
    match = shots_df[(shots_df["match_id"] == match_id) & (shots_df["is_shot"] == 1)].copy()
    if match.empty:
        print(f"No shots found for match {match_id}")
        return

    teams = match["team_name"].unique()
    colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    team_color = {t: list(colors.values())[i % 2] for i, t in enumerate(teams)}

    fig, ax = plt.subplots(figsize=(12, 4))

    for team, grp in match.groupby("team_name"):
        c = team_color[team]
        # Non-goals
        non_goals = grp[grp["is_goal"] == 0]
        ax.scatter(non_goals["event_time_min"],
                   [team] * len(non_goals),
                   s=non_goals["shot_xg"].fillna(0.05) * 800,
                   color=c, alpha=0.6, label=f"{team}")
        # Goals
        goals = grp[grp["is_goal"] == 1]
        ax.scatter(goals["event_time_min"],
                   [team] * len(goals),
                   s=goals["shot_xg"].fillna(0.1) * 800,
                   color=c, marker="*", zorder=5, edgecolors="black",
                   label=f"{team} goal")

    ax.set_xlabel("Minute")
    ax.set_title(f"Shot Timeline — Match {match_id}")
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="upper right")
    if not with_text:
        _strip_text(ax)
    plt.tight_layout()

    path = save_path or os.path.join(FIGURES_DIR, f"fig1_timeline_{match_id}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_momentum_bins(windows_df: pd.DataFrame,
                       save_path: str = None,
                       with_text: bool = True):
    """
    Fig 2/3: Bar charts of P(shot_next_2) and P(goal_next_5) by rolling_xg_5 quintile.
    """
    _ensure_figures_dir()
    df = windows_df.copy()
    df["quintile"] = make_quintile_bins(df["rolling_xg_5"])
    df["quintile"] = df["quintile"].astype(float)

    grouped = df.groupby("quintile").agg(
        P_shot=("shot_next_2", "mean"),
        P_goal=("goal_next_5", "mean"),
        n=("shot_next_2", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title, color in [
        (axes[0], "P_shot", "P(shot in next 2 min) by xG Quintile", "#4e79a7"),
        (axes[1], "P_goal", "P(goal in next 5 min) by xG Quintile", "#f28e2b"),
    ]:
        ax.bar(grouped["quintile"].astype(int), grouped[col], color=color, alpha=0.8,
               edgecolor="black")
        ax.set_xlabel("rolling_xg_5 Quintile (1=lowest, 5=highest)")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_xticks(grouped["quintile"].astype(int))
        if not with_text:
            _strip_text(ax)

    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "fig2_momentum_bins.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_real_vs_sim(real_windows: pd.DataFrame,
                     sim_windows: pd.DataFrame,
                     bootstrap_df: pd.DataFrame,
                     save_path: str = None,
                     with_text: bool = True):
    """
    Fig 4: Bar chart of delta_shot and delta_goal for real vs. sim,
    with 95% CI error bars derived from bootstrap_df.
    """
    _ensure_figures_dir()

    from utils import compute_delta

    delta_shot_real = compute_delta(real_windows, "shot_next_2")
    delta_goal_real = compute_delta(real_windows, "goal_next_5")
    delta_shot_sim = compute_delta(sim_windows, "shot_next_2")
    delta_goal_sim = compute_delta(sim_windows, "goal_next_5")

    # 95% CI from bootstrap
    def ci95(col):
        vals = bootstrap_df[col].dropna()
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        return hi - lo  # total width / 2 for symmetric display
    # Use half-width for error bars
    def ci95_hw(col):
        vals = bootstrap_df[col].dropna()
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        mean = vals.mean()
        return mean - lo, hi - mean

    shot_lo, shot_hi = ci95_hw("delta_shot_real")
    goal_lo, goal_hi = ci95_hw("delta_goal_real")
    sim_shot_lo, sim_shot_hi = ci95_hw("delta_shot_sim")
    sim_goal_lo, sim_goal_hi = ci95_hw("delta_goal_sim")

    x = np.array([0, 1])
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, outcome, real_val, sim_val, real_err, sim_err, title in [
        (axes[0], "Shot Next 2 min",
         delta_shot_real, delta_shot_sim,
         [[shot_lo], [shot_hi]], [[sim_shot_lo], [sim_shot_hi]],
         "Δ P(shot next 2 min): Real vs Null Model"),
        (axes[1], "Goal Next 5 min",
         delta_goal_real, delta_goal_sim,
         [[goal_lo], [goal_hi]], [[sim_goal_lo], [sim_goal_hi]],
         "Δ P(goal next 5 min): Real vs Null Model"),
    ]:
        bars = ax.bar(["Real", "Null Model"], [real_val, sim_val],
                      color=["#4e79a7", "#f28e2b"], alpha=0.8, edgecolor="black")
        ax.errorbar(["Real"], [real_val], yerr=[[real_err[0][0]], [real_err[1][0]]],
                    fmt="none", color="black", capsize=6)
        ax.errorbar(["Null Model"], [sim_val], yerr=[[sim_err[0][0]], [sim_err[1][0]]],
                    fmt="none", color="black", capsize=6)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Δ Probability (Q5 - Q1)")
        ax.set_title(title)
        if not with_text:
            _strip_text(ax)

    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "fig4_real_vs_sim.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_bootstrap_intervals(bootstrap_df: pd.DataFrame,
                             save_path: str = None,
                             with_text: bool = True):
    """
    Fig 5 (optional): Distribution of gap_shot and gap_goal across bootstrap iterations.
    """
    _ensure_figures_dir()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title, color in [
        (axes[0], "gap_shot", "Bootstrap Distribution: gap_shot (Real - Null)", "#4e79a7"),
        (axes[1], "gap_goal", "Bootstrap Distribution: gap_goal (Real - Null)", "#f28e2b"),
    ]:
        vals = bootstrap_df[col].dropna()
        ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="black")
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        ax.axvline(lo, color="red", ls="--", label=f"2.5%: {lo:.3f}")
        ax.axvline(hi, color="red", ls="-", label=f"97.5%: {hi:.3f}")
        ax.axvline(0, color="gray", ls=":", lw=1.5)
        ax.set_xlabel("Gap (Real Δ − Null Δ)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        if not with_text:
            _strip_text(ax)

    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "fig5_bootstrap_intervals.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def _momentum_series(windows_df: pd.DataFrame) -> pd.Series:
    """
    Default momentum metric for null-model visuals.
    Uses rolling_xg_5 so it works for both real and simulated data.
    """
    return windows_df["rolling_xg_5"]


def _rolling_mean(values: pd.Series, window: int = 3) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def plot_real_vs_sim_match_timeline(real_windows: pd.DataFrame,
                                    real_shots: pd.DataFrame,
                                    sim_windows: pd.DataFrame,
                                    sim_shots: pd.DataFrame,
                                    match_id: int,
                                    save_path: str = None,
                                    smooth_window: int = 3):
    """
    Fig N1: Real vs simulated match timeline (same match_id).
    Two panels: Real (top) and Simulated (bottom).
    Each panel shows shot dots, goals, and a rolling momentum line.
    """
    _ensure_figures_dir()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    panels = [
        ("Real match", real_windows, real_shots, axes[0]),
        ("Simulated match", sim_windows, sim_shots, axes[1]),
    ]

    for title, win_df, shots_df, ax in panels:
        if win_df.empty or shots_df.empty:
            ax.set_title(f"{title} (no data)")
            continue

        teams = list(win_df["team_name"].unique())
        colors = ["#1f77b4", "#ff7f0e"]
        team_color = {t: colors[i % 2] for i, t in enumerate(teams)}

        # Momentum lines
        for team in teams:
            team_win = win_df[win_df["team_name"] == team].sort_values("t_minute")
            mom = _momentum_series(team_win)
            mom_s = _rolling_mean(mom, window=smooth_window)
            ax.plot(team_win["t_minute"], mom_s, color=team_color[team], lw=2)

        # Shot and goal markers near bottom
        ymin, ymax = ax.get_ylim()
        base = ymin + (ymax - ymin) * 0.05
        for team in teams:
            team_shots = shots_df[shots_df["team_name"] == team]
            ax.scatter(
                team_shots["event_time_min"],
                np.full(len(team_shots), base),
                s=team_shots["shot_xg"].fillna(0.05) * 500,
                color=team_color[team],
                alpha=0.6,
                marker="o",
                label=f"{team} shot" if title == "Real match" else None,
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
                label=f"{team} goal" if title == "Real match" else None,
            )

        ax.set_ylabel("Momentum (rolling xG)")
        ax.set_title(title)

    # Legend from top panel only, dedup
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l:
            seen.setdefault(l, h)
    axes[0].legend(seen.values(), seen.keys(), loc="upper left", ncol=2)

    axes[1].set_xlabel("Minute")
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, f"fig_null_real_vs_sim_{match_id}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_simulated_momentum_line(sim_windows: pd.DataFrame,
                                 match_id: int,
                                 save_path: str = None,
                                 smooth_window: int = 3):
    """
    Fig N2: Simulated match momentum line over time.
    """
    _ensure_figures_dir()
    if sim_windows.empty:
        print(f"Simulated windows empty for match {match_id}")
        return

    teams = list(sim_windows["team_name"].unique())
    colors = ["#1f77b4", "#ff7f0e"]
    team_color = {t: colors[i % 2] for i, t in enumerate(teams)}

    fig, ax = plt.subplots(figsize=(12, 4))
    for team in teams:
        team_win = sim_windows[sim_windows["team_name"] == team].sort_values("t_minute")
        mom = _momentum_series(team_win)
        mom_s = _rolling_mean(mom, window=smooth_window)
        ax.plot(team_win["t_minute"], mom_s, color=team_color[team], lw=2, label=team)

    ax.set_xlabel("Minute")
    ax.set_ylabel("Momentum (rolling xG)")
    ax.set_title("Simulated match momentum")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, f"fig_null_sim_momentum_{match_id}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_real_vs_sim_momentum_bins(real_windows: pd.DataFrame,
                                   sim_windows: pd.DataFrame,
                                   save_path: str = None):
    """
    Fig N3: Real vs simulated momentum-bin curve.
    x-axis: momentum bin (quintiles), y-axis: P(shot_next_2).
    """
    _ensure_figures_dir()

    real = real_windows.copy()
    sim = sim_windows.copy()

    # Define bins from real data and apply to both
    real["momentum"] = _momentum_series(real)
    sim["momentum"] = _momentum_series(sim)

    bins = pd.qcut(real["momentum"], q=5, labels=False, duplicates="drop")
    # Use bin edges from qcut on real
    _, bin_edges = pd.qcut(real["momentum"], q=5, retbins=True, duplicates="drop")
    real["bin"] = pd.cut(real["momentum"], bins=bin_edges, labels=False, include_lowest=True)
    sim["bin"] = pd.cut(sim["momentum"], bins=bin_edges, labels=False, include_lowest=True)

    real_grp = real.groupby("bin")["shot_next_2"].mean().reset_index()
    sim_grp = sim.groupby("bin")["shot_next_2"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(real_grp["bin"] + 1, real_grp["shot_next_2"], marker="o", label="Real")
    ax.plot(sim_grp["bin"] + 1, sim_grp["shot_next_2"], marker="o", label="Simulated")
    ax.set_xlabel("Momentum quintile (1=low, 5=high)")
    ax.set_ylabel("P(shot in next 2 min)")
    ax.set_title("Real vs Simulated Momentum Bins")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "fig_null_bins_shot_next_2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_effect_size_bars(real_windows: pd.DataFrame,
                          sim_windows: pd.DataFrame,
                          bootstrap_df: pd.DataFrame = None,
                          save_path: str = None):
    """
    Fig N4: Bar chart of effect sizes (Δshot and Δgoal) for real vs simulated.
    Uses bootstrap_df for error bars if provided.
    """
    _ensure_figures_dir()

    from utils import compute_delta

    delta_shot_real = compute_delta(real_windows, "shot_next_2")
    delta_goal_real = compute_delta(real_windows, "goal_next_5")
    delta_shot_sim = compute_delta(sim_windows, "shot_next_2")
    delta_goal_sim = compute_delta(sim_windows, "goal_next_5")

    labels = ["Δ Shot (Real)", "Δ Shot (Sim)", "Δ Goal (Real)", "Δ Goal (Sim)"]
    values = [delta_shot_real, delta_shot_sim, delta_goal_real, delta_goal_sim]

    yerr = None
    if bootstrap_df is not None and not bootstrap_df.empty:
        def ci_hw(col):
            vals = bootstrap_df[col].dropna()
            lo = np.percentile(vals, 2.5)
            hi = np.percentile(vals, 97.5)
            mean = vals.mean()
            return mean - lo, hi - mean

        shot_lo, shot_hi = ci_hw("delta_shot_real")
        goal_lo, goal_hi = ci_hw("delta_goal_real")
        sim_shot_lo, sim_shot_hi = ci_hw("delta_shot_sim")
        sim_goal_lo, sim_goal_hi = ci_hw("delta_goal_sim")
        yerr = [
            [shot_lo, sim_shot_lo, goal_lo, sim_goal_lo],
            [shot_hi, sim_shot_hi, goal_hi, sim_goal_hi],
        ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#4e79a7", "#f28e2b"],
           alpha=0.85, edgecolor="black", yerr=yerr, capsize=6)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Δ Probability (Q5 - Q1)")
    ax.set_title("Effect Sizes: Real vs Simulated")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "fig_null_effect_sizes.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Supplementary pressure figures ───────────────────────────────────────────

def _binary_bar(ax, windows_df, pressure_col, outcome_col, labels, title, colors):
    """Helper: bar chart of P(outcome) split by a binary pressure column."""
    grp = windows_df.groupby(pressure_col)[outcome_col].mean()
    ax.bar(labels, [grp.get(0, 0), grp.get(1, 0)],
           color=colors, alpha=0.85, edgecolor="black")
    ax.set_ylabel(f"P({outcome_col})")
    ax.set_title(title)
    ax.set_ylim(0, max(grp.max() * 1.3, 0.05))
    for i, v in enumerate([grp.get(0, 0), grp.get(1, 0)]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10)


def plot_supp_corner_pressure(windows_df: pd.DataFrame, save_path: str = None):
    """
    Supp Fig 1: P(shot_next_2) split by recent_corner_pressure (0 vs 1).
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    _binary_bar(ax, windows_df,
                pressure_col="recent_corner_pressure",
                outcome_col="shot_next_2",
                labels=["No recent corner shot", "Recent corner shot"],
                title="P(shot next 2 min) by Recent Corner Pressure",
                colors=["#aec7e8", "#1f77b4"])
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "suppfig1_corner_pressure.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_supp_set_piece_pressure(windows_df: pd.DataFrame, save_path: str = None):
    """
    Supp Fig 2: P(goal_next_5) split by recent_set_piece_pressure (0 vs 1).
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    _binary_bar(ax, windows_df,
                pressure_col="recent_set_piece_pressure",
                outcome_col="goal_next_5",
                labels=["No recent set-piece shot", "Recent set-piece shot"],
                title="P(goal next 5 min) by Recent Set-Piece Pressure",
                colors=["#ffbb78", "#ff7f0e"])
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "suppfig2_set_piece_pressure.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_supp_big_chance_bins(windows_df: pd.DataFrame, save_path: str = None):
    """
    Supp Fig 3: P(goal_next_5) by big_chances_5 bin (0 / 1 / 2+).
    """
    _ensure_figures_dir()
    df = windows_df.copy()
    df["big_chance_bin"] = df["big_chances_5"].clip(upper=2).astype(int)
    bin_labels = {0: "0", 1: "1", 2: "2+"}

    grp = df.groupby("big_chance_bin")["goal_next_5"].mean()
    counts = df.groupby("big_chance_bin")["goal_next_5"].count()

    x_vals = sorted(grp.index)
    labels = [bin_labels[v] for v in x_vals]
    heights = [grp[v] for v in x_vals]
    ns = [counts[v] for v in x_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, heights, color=["#d62728", "#e377c2", "#9467bd"][:len(x_vals)],
                  alpha=0.85, edgecolor="black")
    ax.set_xlabel("Big chances (xG ≥ 0.30) in last 5 min")
    ax.set_ylabel("P(goal next 5 min)")
    ax.set_title("P(goal next 5 min) by Big Chance Count")
    for i, (h, n) in enumerate(zip(heights, ns)):
        ax.text(i, h + 0.002, f"{h:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "suppfig3_big_chance_bins.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved → {path}")
