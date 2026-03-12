"""
make_presentation_figs.py  —  Part 2
Figures 1–6 + figure_guide.md
Run after make_presentation.py defines helpers (or run standalone).
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT, 'src'))

# Re-import everything (standalone safe)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

DATA_INTERIM   = os.path.join(PROJECT, 'data', 'interim')
DATA_PROCESSED = os.path.join(PROJECT, 'data', 'processed')
OUT_DIR        = os.path.join(PROJECT, 'outputs', 'figures', 'presentation')
os.makedirs(OUT_DIR, exist_ok=True)

C_BLUE = "#1d4ed8"; C_RED = "#b91c1c"; C_GOLD = "#d97706"
C_DARK = "#0f172a"; C_NULL = "#64748b"; C_BG = "#ffffff"
C_SUBTEXT = "#64748b"; C_MUTED = "#cbd5e1"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 12, 'figure.facecolor': C_BG, 'axes.facecolor': C_BG,
    'axes.edgecolor': '#e2e8f0', 'axes.linewidth': 0.8,
    'xtick.color': '#475569', 'ytick.color': '#475569',
    'text.color': C_DARK, 'axes.labelcolor': '#334155',
    'grid.color': '#f1f5f9', 'grid.linewidth': 0.8,
    'xtick.labelsize': 11, 'ytick.labelsize': 10,
})
DPI = 200

def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=DPI if ext == 'png' else None,
                    bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f"  ✓  {name}")

def despine(ax, keep=('bottom', 'left')):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(s in keep)

print("Loading data…")
shots   = pd.read_csv(os.path.join(DATA_INTERIM,    'shots.csv'))
windows = pd.read_csv(os.path.join(DATA_INTERIM,    'team_windows.csv'))
boot    = pd.read_json(os.path.join(DATA_PROCESSED, 'bootstrap_results.json'))

MATCH_META = {
    3869685: ("Argentina", "France",       "3 – 3", "World Cup Final", "Dec 18, 2022"),
    3857300: ("Argentina", "Saudi Arabia", "1 – 2", "Group Stage",     "Nov 22, 2022"),
    3857259: ("Cameroon",  "Serbia",       "3 – 3", "Group Stage",     "Nov 28, 2022"),
    3857284: ("Germany",   "Japan",        "1 – 2", "Group Stage",     "Nov 23, 2022"),
    3869519: ("Argentina", "Croatia",      "3 – 0", "Semi-Final",      "Dec 13, 2022"),
}

def rolling_pressure(shots_df, match_id, team, window=5, dt=0.2):
    s = shots_df[(shots_df['match_id'] == match_id) & (shots_df['team_name'] == team)]
    t = np.arange(0, 95, dt)
    xg = np.array([
        s.loc[(s['event_time_min'] >= ti - window) & (s['event_time_min'] < ti),
              'shot_xg'].sum()
        for ti in t
    ])
    return t, gaussian_filter1d(xg, sigma=3)

def draw_timeline(ax, match_id, team_a, team_b, ca=C_BLUE, cb=C_RED, annotate_goals=True):
    ms = shots[shots['match_id'] == match_id].copy()
    sa = ms[ms['team_name'] == team_a]
    sb = ms[ms['team_name'] == team_b]
    t, pa = rolling_pressure(ms, match_id, team_a)
    _,  pb = rolling_pressure(ms, match_id, team_b)

    ax.fill_between(t,  pa, alpha=0.13, color=ca)
    ax.fill_between(t, -pb, alpha=0.13, color=cb)
    ax.plot(t,  pa, lw=2.2, color=ca, alpha=0.9)
    ax.plot(t, -pb, lw=2.2, color=cb, alpha=0.9)
    ax.axhline(0, color=C_MUTED, lw=0.8)

    for s_team, sign, c in [(sa, +1, ca), (sb, -1, cb)]:
        ng = s_team[s_team['is_goal'] == 0]
        sz = ng['shot_xg'].fillna(0.02) * 450 + 10
        ax.scatter(ng['event_time_min'], np.full(len(ng), sign * 0.006),
                   s=sz, color=c, alpha=0.30, linewidths=0, zorder=3)

    ax.axvline(45, color=C_MUTED, lw=1, ls=(0, (5, 4)), zorder=1)
    ax.set_xlim(0, 93)
    despine(ax, keep=('bottom',))
    ax.set_yticks([])
    ax.set_xlabel("Match Minute", fontsize=12, color='#475569', labelpad=6)

    ga = sa[sa['is_goal'] == 1]
    gb = sb[sb['is_goal'] == 1]

    if annotate_goals:
        for df_g, sign, c, arr in [(ga, +1, ca, pa), (gb, -1, cb, pb)]:
            for _, row in df_g.iterrows():
                idx = np.argmin(np.abs(t - row['event_time_min']))
                yp  = sign * arr[idx]
                ax.scatter(row['event_time_min'], yp + sign * 0.008,
                           s=280, color=C_GOLD, marker='*', zorder=6,
                           edgecolors='white', linewidths=0.8)
                offset = 0.030 if sign > 0 else -0.030
                va = 'bottom' if sign > 0 else 'top'
                ax.text(row['event_time_min'], yp + sign * offset,
                        f"{int(row['minute'])}'",
                        ha='center', va=va, fontsize=9.5, color=c, fontweight='bold')

    return ga, gb, t, pa, pb

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  —  Three hero match timelines
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 1: Hero timelines ──")
HERO_IDS = [3869685, 3857300, 3857259]

for i, mid in enumerate(HERO_IDS, 1):
    team_a, team_b, score, stage, date = MATCH_META[mid]
    ca = C_BLUE; cb = C_RED

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ga, gb, t, pa, pb = draw_timeline(ax, mid, team_a, team_b, ca, cb)

    ylim = ax.get_ylim()
    yspan = ylim[1] - ylim[0]

    # Team labels on left margin
    ax.text(-1, ylim[1] * 0.55, team_a, fontsize=15, fontweight='bold',
            color=ca, ha='right', va='center')
    ax.text(-1, ylim[0] * 0.55, team_b, fontsize=15, fontweight='bold',
            color=cb, ha='right', va='center')

    # Half-time label
    ax.text(45.6, ylim[1] * 0.93, "HT", fontsize=8.5, color='#94a3b8', va='top')

    # Legend
    leg = [
        Line2D([0],[0], color=ca, lw=2.2, label=f"{team_a} attacking pressure"),
        Line2D([0],[0], color=cb, lw=2.2, label=f"{team_b} attacking pressure"),
        Line2D([0],[0], color='none', marker='*', markerfacecolor=C_GOLD,
               markersize=12, label="Goal scored"),
    ]
    ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=10,
              labelcolor='#475569', handlelength=1.5)

    fig.suptitle(f"{team_a}  {score}  {team_b}",
                 fontsize=22, fontweight='bold', color=C_DARK, y=1.01)
    ax.set_title(f"{stage}  ·  {date}", fontsize=12, color=C_SUBTEXT,
                 style='italic', pad=5)
    fig.text(0.01, -0.03,
             "Shaded area = 5-min rolling attacking pressure (xG)  ·  "
             "Dot size = shot xG  ·  ★ = goal",
             fontsize=8.5, color='#94a3b8')

    plt.tight_layout()
    save_fig(fig, f"hero_timeline_match_{i}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1b  —  Wide cover (Argentina vs France)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 1b: Cover timeline ──")
mid = 3869685
team_a, team_b, score, stage, date = MATCH_META[mid]

fig, ax = plt.subplots(figsize=(20, 4.5))
ga, gb, t, pa, pb = draw_timeline(ax, mid, team_a, team_b)
ylim = ax.get_ylim()

ax.text(-0.5, ylim[1] * 0.60, team_a, fontsize=16, fontweight='bold',
        color=C_BLUE, ha='right')
ax.text(-0.5, ylim[0] * 0.60, team_b, fontsize=16, fontweight='bold',
        color=C_RED, ha='right')
ax.text(45.6, ylim[1] * 0.92, "HT", fontsize=8.5, color='#94a3b8')

fig.suptitle("Argentina  3 – 3  France", fontsize=26, fontweight='bold',
             color=C_DARK, y=1.02)
ax.set_title("FIFA World Cup Final  ·  December 18, 2022",
             fontsize=13, color=C_SUBTEXT, style='italic', pad=5)
fig.text(0.01, -0.04,
         "5-minute rolling attacking pressure (xG)  ·  ★ = goal",
         fontsize=9, color='#94a3b8')

plt.tight_layout()
save_fig(fig, "hero_timeline_cover")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  —  Momentum illusion example
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 2: Momentum illusion ──")

# Find goal with highest pre-goal xG buildup (≥2 shots in 8-min window before)
best_score, best = -1.0, None
for mid_s, grp in shots.groupby('match_id'):
    goals = grp[grp['is_goal'] == 1]
    for _, row in goals.iterrows():
        tg = row['event_time_min']
        if tg < 20 or tg > 78:
            continue
        team = row['team_name']
        pre = grp[(grp['team_name'] == team) &
                  (grp['event_time_min'] >= tg - 8) &
                  (grp['event_time_min'] < tg - 0.5)]
        val = float(pre['shot_xg'].sum())
        if val > best_score and len(pre) >= 2:
            best_score = val
            best = (mid_s, team, tg, row.get('player_name', ''), val)

mid_ill, team_hero, t_goal, scorer, _ = best
ms_ill = shots[shots['match_id'] == mid_ill]
teams_ill = ms_ill['team_name'].unique().tolist()
opp_ill = next(t for t in teams_ill if t != team_hero)
meta_ill = MATCH_META.get(mid_ill, (team_hero, opp_ill, '?', 'Group Stage', ''))

fig, ax = plt.subplots(figsize=(15, 5.5))
_, _, t_arr, pa_arr, pb_arr = draw_timeline(
    ax, mid_ill, team_hero, opp_ill, annotate_goals=False)

# Zoom to the interesting window
t_start = max(2, t_goal - 22)
t_end   = min(91, t_goal + 13)
ax.set_xlim(t_start, t_end)

ylim = ax.get_ylim()

# Goal star at momentum-curve height
goal_idx = np.argmin(np.abs(t_arr - t_goal))
goal_yp  = pa_arr[goal_idx]
ax.scatter(t_goal, goal_yp + 0.008, s=380, color=C_GOLD, marker='*',
           zorder=7, edgecolors='white', linewidths=1.0)

# Shaded buildup band
ax.axvspan(t_goal - 8, t_goal - 0.3, alpha=0.07, color=C_BLUE, zorder=0)

# Annotation: "Pressure builds"
mid_buildup = t_goal - 4.5
ax.annotate("Pressure builds →",
            xy=(mid_buildup, ylim[1] * 0.62),
            fontsize=12, color=C_BLUE, fontweight='bold', ha='center',
            annotation_clip=False)

# Annotation: "Goal scored here"
ann_x = min(t_goal + 4, t_end - 1.5)
ax.annotate("★  Goal scored here",
            xy=(t_goal, goal_yp + 0.008),
            xytext=(ann_x, ylim[1] * 0.80),
            fontsize=11, color=C_GOLD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=1.6,
                            connectionstyle='arc3,rad=-0.2'),
            ha='left', annotation_clip=False)

# Annotation: "Looks like momentum"
ax.text((t_start + t_goal - 8) / 2, ylim[0] * 0.72,
        '"Looks like momentum"',
        fontsize=11, color='#94a3b8', style='italic', ha='center')

# Team labels
ax.text(t_start + 0.4, ylim[1] * 0.88, team_hero,
        fontsize=13, fontweight='bold', color=C_BLUE, va='top')
ax.text(t_start + 0.4, ylim[0] * 0.88, opp_ill,
        fontsize=13, fontweight='bold', color=C_RED, va='bottom')

fig.suptitle("What Fans See as Momentum",
             fontsize=20, fontweight='bold', color=C_DARK, y=1.01)
ax.set_title(f"{team_hero} vs {opp_ill}  ·  {meta_ill[3]}",
             fontsize=12, color=C_SUBTEXT, style='italic', pad=5)
fig.text(0.01, -0.03,
         "Shaded band = buildup period  ·  Rolling xG line  ·  ★ = goal",
         fontsize=8.5, color='#94a3b8')

plt.tight_layout()
save_fig(fig, "momentum_illusion_example")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  —  Descriptive pressure effects
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 3: Descriptive results ──")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Pressure Raises Near-Future Scoring Odds",
             fontsize=20, fontweight='bold', color=C_DARK, y=1.01)
fig.text(0.5, 0.96, "Descriptive probabilities across all 64 World Cup matches",
         ha='center', fontsize=12, color=C_SUBTEXT, style='italic')

panels = [
    ('recent_big_chance_pressure', 'shot_next_2',
     'Big Chance in Last 5 Min\n(shot with xG ≥ 0.30)',
     'Shot in Next 2 Minutes', C_BLUE),
    ('recent_big_chance_pressure', 'goal_next_5',
     'Big Chance in Last 5 Min\n(shot with xG ≥ 0.30)',
     'Goal in Next 5 Minutes', C_RED),
]

for ax, (pred, outcome, pred_label, out_label, color) in zip(axes, panels):
    grp = windows.groupby(pred)[outcome].mean()
    ns  = windows[pred].value_counts().sort_index()
    vals = [grp.get(0, 0), grp.get(1, 0)]
    ns_vals = [ns.get(0, 0), ns.get(1, 0)]
    xlabels = ['No Recent\nBig Chance', 'Recent\nBig Chance']
    colors  = ['#e2e8f0', color]

    bars = ax.bar(xlabels, vals, color=colors, alpha=0.88,
                  edgecolor='white', linewidth=2.0, width=0.5)

    for bar, val, n in zip(bars, vals, ns_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val * 100:.1f}%",
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                color='#334155')
        ax.text(bar.get_x() + bar.get_width() / 2, -0.006,
                f"n = {n:,}", ha='center', va='top', fontsize=9, color='#94a3b8')

    diff = vals[1] - vals[0]
    sign = '+' if diff >= 0 else ''
    ax.text(0.97, 0.97, f"Δ = {sign}{diff * 100:.1f} pp",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, color=color, fontweight='bold')

    ax.set_title(pred_label, fontsize=13, fontweight='bold', color=C_DARK, pad=10)
    ax.set_ylabel(f"P({out_label})", fontsize=12, color='#475569')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_ylim(0, max(vals) * 1.40)
    despine(ax)
    ax.grid(axis='y', color='#f1f5f9', linewidth=1.2, zorder=0)
    ax.tick_params(axis='x', labelsize=12)

plt.tight_layout(rect=[0, 0.01, 1, 0.94])
save_fig(fig, "descriptive_pressure_effects")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4  —  Main result: dumbbell plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 4: Main result (dumbbell) ──")

fig, ax = plt.subplots(figsize=(12, 5.0))

rows = [
    ("Shot in Next 2 Min",  'delta_shot_real', 'delta_shot_sim', 1.0),
    ("Goal in Next 5 Min",  'delta_goal_real', 'delta_goal_sim', 0.0),
]

for label, rc, sc, y in rows:
    rv = boot[rc].dropna(); sv = boot[sc].dropna()
    rm, sm = rv.mean(), sv.mean()
    r_lo, r_hi = np.percentile(rv, [2.5, 97.5])
    s_lo, s_hi = np.percentile(sv, [2.5, 97.5])

    # Connecting line
    ax.plot([rm, sm], [y, y], color='#cbd5e1', lw=2.5, solid_capstyle='round', zorder=1)

    # CI bands
    ax.fill_betweenx([y - 0.07, y + 0.07], r_lo, r_hi,
                     color=C_BLUE, alpha=0.18, zorder=2)
    ax.fill_betweenx([y - 0.07, y + 0.07], s_lo, s_hi,
                     color=C_NULL, alpha=0.18, zorder=2)

    # Dots
    ax.scatter(rm, y, s=200, color=C_BLUE, zorder=4, edgecolors='white', linewidths=2)
    ax.scatter(sm, y, s=200, color=C_NULL, zorder=4, edgecolors='white', linewidths=2)

    # Value labels above dots
    ax.text(rm, y + 0.13, f"{rm * 100:.1f}%", ha='center', fontsize=11,
            color=C_BLUE, fontweight='bold')
    ax.text(sm, y + 0.13, f"{sm * 100:.1f}%", ha='center', fontsize=11,
            color=C_NULL, fontweight='bold')

    # Row label (left)
    ax.text(-0.008, y, label, ha='right', va='center',
            fontsize=13, fontweight='bold', color=C_DARK)

# Legend
leg_y = -0.52
for lx, c, lbl in [(0.05, C_BLUE, "Real match data"),
                    (0.18, C_NULL, "Null model  (random timing)")]:
    ax.scatter(lx, leg_y, s=150, color=c, zorder=5, edgecolors='white', linewidths=2)
    ax.text(lx + 0.012, leg_y, lbl, va='center', fontsize=11, color='#475569')

ax.set_xlim(-0.015, 0.34)
ax.set_ylim(-0.72, 1.45)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
ax.set_xlabel(
    "Probability Difference  (highest pressure quintile − lowest pressure quintile)",
    fontsize=11.5, color='#475569', labelpad=10)
despine(ax, keep=('bottom',))
ax.set_yticks([])
ax.grid(axis='x', color='#f1f5f9', linewidth=1.2, zorder=0)
ax.axvline(0, color='#e2e8f0', lw=1)

fig.suptitle('The Null Model Produces Stronger "Momentum" Than Real Soccer',
             fontsize=18, fontweight='bold', color=C_DARK, y=1.02)
ax.set_title(
    "Randomly timed shots create as much apparent clustering as actual match play.",
    fontsize=12, color=C_SUBTEXT, style='italic', pad=8)

plt.tight_layout()
save_fig(fig, "main_result_real_vs_null")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5  —  Hot-hand fallacy connection
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 5: Hot-hand connection ──")

fig = plt.figure(figsize=(14, 5.8))
gs  = GridSpec(1, 2, figure=fig, wspace=0.06)
ax_l = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])

for ax in (ax_l, ax_r):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    despine(ax, keep=())

# ─ LEFT: What fans believe ─
ax_l.set_facecolor('#f0f4ff')
ax_l.text(5, 9.2, "What Fans Believe", ha='center', va='top',
          fontsize=15, fontweight='bold', color=C_BLUE)
ax_l.text(5, 8.35, '"Pressure leads to goals"',
          ha='center', va='top', fontsize=11, color='#475569', style='italic')

chain = [
    (5, 6.8, "High attacking\npressure", C_BLUE),
    (5, 4.8, "Shots cluster\nnear goal", C_BLUE),
    (5, 2.8, "Goal  ★", C_GOLD),
]
for x, y, txt, c in chain:
    patch = mpatches.FancyBboxPatch((x - 1.7, y - 0.55), 3.4, 1.1,
                                    boxstyle="round,pad=0.12",
                                    fc=c, ec='none', alpha=0.14)
    ax_l.add_patch(patch)
    ax_l.text(x, y, txt, ha='center', va='center', fontsize=11,
              fontweight='bold', color=c)

for y_f, y_t in [(6.22, 5.37), (4.22, 3.37)]:
    ax_l.annotate('', xy=(5, y_t), xytext=(5, y_f),
                  arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.8))

ax_l.text(5, 1.5, "Intuitively compelling, but…",
          ha='center', fontsize=10, color='#94a3b8', style='italic')

# ─ RIGHT: What the data shows ─
ax_r.set_facecolor('#f8f8f8')
ax_r.text(5, 9.2, "What the Data Shows", ha='center', va='top',
          fontsize=15, fontweight='bold', color=C_DARK)
ax_r.text(5, 8.35, '"Random timing creates the same clustering"',
          ha='center', va='top', fontsize=11, color='#475569', style='italic')

r_shot = boot['delta_shot_real'].mean()
s_shot = boot['delta_shot_sim'].mean()
r_goal = boot['delta_goal_real'].mean()
s_goal = boot['delta_goal_sim'].mean()
ratio  = s_shot / max(r_shot, 0.001)

box_data = [
    (6.3, C_BLUE, "Real soccer",
     f"Shot effect:  +{r_shot*100:.1f} pp\nGoal effect:  +{r_goal*100:.1f} pp"),
    (3.4, C_NULL, "Null model  (random timing)",
     f"Shot effect:  +{s_shot*100:.1f} pp\nGoal effect:  +{s_goal*100:.1f} pp"),
]
for y, c, title, body in box_data:
    patch = mpatches.FancyBboxPatch((1.0, y - 1.0), 8.0, 2.0,
                                    boxstyle="round,pad=0.15",
                                    fc=c, ec='none', alpha=0.10)
    ax_r.add_patch(patch)
    ax_r.text(1.7, y + 0.35, title,
              ha='left', va='center', fontsize=11, fontweight='bold', color=c)
    ax_r.text(1.7, y - 0.30, body,
              ha='left', va='center', fontsize=10, color='#475569')

ax_r.text(5, 1.85, f"Null model effect is",
          ha='center', fontsize=11, color='#334155')
ax_r.text(5, 1.15, f"≈ {ratio:.0f}× LARGER than real soccer",
          ha='center', fontsize=14, fontweight='bold', color=C_RED)

# Central divider
fig.add_artist(plt.Line2D([0.5, 0.5], [0.06, 0.92],
                           transform=fig.transFigure,
                           color='#e2e8f0', lw=1.5))

fig.suptitle("The Hot-Hand Fallacy in Soccer",
             fontsize=20, fontweight='bold', color=C_DARK, y=1.01)
fig.text(0.5, 0.965,
         "What fans call momentum may be a statistical illusion",
         ha='center', fontsize=12, color=C_SUBTEXT, style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "hot_hand_fallacy_connection")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6  —  Supplementary pressure panels
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fig 6: Supplementary panels ──")

supp = [
    ('recent_corner_pressure',    'Shot in Next 2 Min', 'shot_next_2',
     'Corner in Last 5 Min',      "#7c3aed"),
    ('recent_set_piece_pressure', 'Goal in Next 5 Min', 'goal_next_5',
     'Set-Piece in Last 5 Min',   "#0891b2"),
    ('recent_big_chance_pressure','Goal in Next 5 Min', 'goal_next_5',
     'Big Chance in Last 5 Min\n(xG ≥ 0.30)', C_RED),
]
all_ok = all(c in windows.columns for c, *_ in supp)

if all_ok:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    fig.suptitle("Supplementary: Specific Pressure Types",
                 fontsize=17, fontweight='bold', color=C_DARK, y=1.01)

    for ax, (pred, out_label, outcome, pred_label, color) in zip(axes, supp):
        grp  = windows.groupby(pred)[outcome].mean()
        ns   = windows[pred].value_counts().sort_index()
        vals = [grp.get(0, 0), grp.get(1, 0)]
        ns_v = [ns.get(0, 0), ns.get(1, 0)]

        bars = ax.bar(['No\nPressure', 'Recent\nPressure'], vals,
                      color=['#e2e8f0', color], alpha=0.88,
                      edgecolor='white', linewidth=2.0, width=0.5)

        for bar, val, n in zip(bars, vals, ns_v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val * 100:.1f}%",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#334155')
            ax.text(bar.get_x() + bar.get_width() / 2, -0.006,
                    f"n={n:,}", ha='center', va='top',
                    fontsize=8.5, color='#94a3b8')

        diff = vals[1] - vals[0]
        sign = '+' if diff >= 0 else ''
        ax.text(0.97, 0.97, f"Δ = {sign}{diff * 100:.1f} pp",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color=color, fontweight='bold')

        ax.set_title(pred_label, fontsize=12, fontweight='bold', color=C_DARK, pad=10)
        ax.set_ylabel(f"P({out_label})", fontsize=11, color='#475569')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.set_ylim(0, max(vals) * 1.42)
        despine(ax)
        ax.grid(axis='y', color='#f1f5f9', linewidth=1.2, zorder=0)
        ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_fig(fig, "supplementary_pressure_panels")
else:
    print("  ⚠  Supplementary cols missing — skipped")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GUIDE
# ─────────────────────────────────────────────────────────────────────────────
r_shot_pct = boot['delta_shot_real'].mean() * 100
s_shot_pct = boot['delta_shot_sim'].mean()  * 100
r_goal_pct = boot['delta_goal_real'].mean() * 100
s_goal_pct = boot['delta_goal_sim'].mean()  * 100

guide = f"""# Figure Guide — CS109 Soccer Momentum Presentation

All figures in `outputs/figures/presentation/`
Format: high-res PNG (200 dpi) + vector PDF

---

## hero_timeline_match_1  —  Argentina 3–3 France (Final)
**Purpose:** Open with the most iconic match of the tournament.
**Talking point:** "The World Cup Final had six goals and dramatic swings — let's measure what was actually happening inside the match."
**Why chosen:** Maximum cultural resonance; six goals means many momentum swings to visualise.

## hero_timeline_match_2  —  Argentina 1–2 Saudi Arabia
**Purpose:** Show the tournament's biggest upset through a pressure lens.
**Talking point:** "Saudi Arabia were massive underdogs. Did they seize momentum, or did shots just cluster by chance?"
**Why chosen:** Most famous upset of the tournament; great narrative hook.

## hero_timeline_match_3  —  Cameroon 3–3 Serbia
**Purpose:** High-scoring back-and-forth match with many lead changes.
**Talking point:** "Six goals, multiple lead changes — a perfect stress test for momentum."
**Why chosen:** Highest group-stage goal count; visually dramatic pressure swings.

## hero_timeline_cover  —  Argentina vs France (wide, cinematic)
**Purpose:** Presentation cover or opening slide.
**Talking point:** "Today we ask: is soccer momentum real, or a statistical illusion?"
**Why chosen:** Wide format feels cinematic; minimal labels keep attention on the waves.

## momentum_illusion_example
**Purpose:** The single most intuitive figure — shows why fans believe in momentum.
**Talking point:** "Here's a real moment: pressure builds, then a goal. But is this pattern meaningful or expected by chance?"
**Why chosen:** Auto-selected as the match window with the highest pre-goal xG buildup (≥2 shots in the 8 minutes before a goal).

## descriptive_pressure_effects
**Purpose:** Clean descriptive evidence that pressure increases near-future outcomes.
**Talking point:** "At first glance, momentum looks real — teams with a recent big chance are more likely to score again soon."
**Why chosen:** Binary comparison is immediately readable; audience reaction: 'momentum is real!'

## main_result_real_vs_null  ← KEY SLIDE
**Purpose:** The central finding of the project.
**Talking point:** "Here's the surprise. The null model — randomly timed shots — produces {s_shot_pct:.0f}% momentum-like clustering vs {r_shot_pct:.0f}% in real soccer. What fans call momentum may be a statistical illusion."
**Why chosen:** Dumbbell format makes the real vs null contrast legible in one glance.

## hot_hand_fallacy_connection
**Purpose:** Conceptual closer linking the project to the hot-hand fallacy.
**Talking point:** "Just like basketball's hot hand, soccer momentum may be a pattern our brains impose on random clustering."
**Why chosen:** Gives the presentation a satisfying conceptual conclusion.

## supplementary_pressure_panels
**Purpose:** Supporting evidence for three specific pressure types.
**Talking point:** "Even specific catalysts — corners, free kicks, big chances — show only modest real effects."
**Why chosen:** Reinforces the main finding without cluttering the main slides.
"""

with open(os.path.join(OUT_DIR, 'figure_guide.md'), 'w') as f:
    f.write(guide)
print("  ✓  figure_guide.md")

print(f"\n✅  All done — figures saved to {OUT_DIR}\n")
