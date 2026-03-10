import json
import os
import sys

import pandas as pd

# Allow running this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_events import events_to_dataframe

# Reuse load_match_ids from data/statsbomb_data.py
_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sys.path.insert(0, os.path.abspath(_data_dir))
from statsbomb_data import load_match_ids  # noqa: E402


def load_match_events(path: str) -> list:
    """Load a single match event JSON file and inject match_id into each event."""
    match_id = int(os.path.splitext(os.path.basename(path))[0])
    with open(path, "r") as f:
        events = json.load(f)
    for e in events:
        e["match_id"] = match_id
    return events


def load_competition_events(dir_path: str, match_ids: list) -> pd.DataFrame:
    """
    Load events for all matches in match_ids from dir_path/<match_id>.json.

    Logs validation counts: matches, events, shots, goals, unique event types.
    """
    if not os.path.isdir(dir_path):
        print(f"ERROR: Raw data directory not found: {dir_path}")
        return pd.DataFrame()

    all_dfs = []
    missing = []

    for mid in match_ids:
        path = os.path.join(dir_path, f"{mid}.json")
        if not os.path.exists(path):
            missing.append(mid)
            continue
        events = load_match_events(path)
        df = events_to_dataframe(events)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("WARNING: data/raw/ appears to be empty or no matching files found.")
        return pd.DataFrame()

    if missing:
        print(f"WARNING: {len(missing)} match files not found: {missing[:5]}{'...' if len(missing)>5 else ''}")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Validation logging
    n_matches = full_df["match_id"].nunique()
    n_events = len(full_df)
    n_shots = full_df["is_shot"].sum()
    n_goals = full_df["is_goal"].sum()
    unique_types = sorted(full_df["type_name"].unique())

    print(f"Loaded {n_matches} matches, {n_events} events")
    print(f"  Shots: {n_shots}  |  Goals: {n_goals}")
    print(f"  Unique event types ({len(unique_types)}): {unique_types}")

    return full_df
