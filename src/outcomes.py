import pandas as pd


def has_shot_in_window(shots_df: pd.DataFrame, match_id: int, team_name: str,
                       start: float, end: float) -> int:
    """Return 1 if team has at least one shot in [start, end), else 0."""
    mask = (
        (shots_df["match_id"] == match_id) &
        (shots_df["team_name"] == team_name) &
        (shots_df["event_time_min"] >= start) &
        (shots_df["event_time_min"] < end)
    )
    return 1 if mask.any() else 0


def has_goal_in_window(shots_df: pd.DataFrame, match_id: int, team_name: str,
                       start: float, end: float) -> int:
    """Return 1 if team has at least one goal in [start, end), else 0."""
    mask = (
        (shots_df["match_id"] == match_id) &
        (shots_df["team_name"] == team_name) &
        (shots_df["event_time_min"] >= start) &
        (shots_df["event_time_min"] < end) &
        (shots_df["is_goal"] == 1)
    )
    return 1 if mask.any() else 0
