import logging
import os

import numpy as np
import pandas as pd


def setup_logger(name, log_file):
    """Create a logger that writes to both a file and the console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def make_quintile_bins(series):
    """
    Divide series into 5 bins (quintiles).

    Falls back to custom label-based bins when pd.qcut fails due to
    too many duplicate values (common when most observations are 0).
    """
    try:
        return pd.qcut(series, q=5, labels=False, duplicates="drop")
    except ValueError:
        # Custom fallback: zero / low / medium / high / very_high
        max_val = series.max()
        if max_val == 0:
            return pd.Series(0, index=series.index)
        cuts = pd.cut(
            series,
            bins=[-np.inf, 0, max_val * 0.25, max_val * 0.5, max_val * 0.75, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        )
        return cuts.astype(float)


def compute_delta(windows_df, outcome_col):
    """
    Compute P(outcome | Q5) - P(outcome | Q1) based on rolling_xg_5 quintile bins.

    Returns float (np.nan if quintile bins are missing).
    """
    df = windows_df.copy()
    df["_bin"] = make_quintile_bins(df["rolling_xg_5"])

    q1 = df[df["_bin"] == df["_bin"].min()]
    q5 = df[df["_bin"] == df["_bin"].max()]

    if len(q1) == 0 or len(q5) == 0:
        return np.nan

    p_q1 = q1[outcome_col].mean()
    p_q5 = q5[outcome_col].mean()
    return float(p_q5 - p_q1)


def summarize_momentum_table(windows_df):
    """
    Build Table A: mean outcomes by rolling_xg_5 quintile bin.

    Returns a DataFrame with columns:
        quintile_bin, n_rows, rolling_xg_5_mean,
        P_shot_next_2, P_goal_next_5
    """
    df = windows_df.copy()
    df["quintile_bin"] = make_quintile_bins(df["rolling_xg_5"])

    grouped = (
        df.groupby("quintile_bin")
        .agg(
            n_rows=("rolling_xg_5", "count"),
            rolling_xg_5_mean=("rolling_xg_5", "mean"),
            P_shot_next_2=("shot_next_2", "mean"),
            P_goal_next_5=("goal_next_5", "mean"),
        )
        .reset_index()
    )
    return grouped
