import pandas as pd


def parse_event(event: dict, match_id: int) -> dict:
    """Flatten one StatsBomb event dict into a plain dict of required columns."""
    minute = event.get("minute", 0)
    second = event.get("second", 0)

    type_info = event.get("type", {})
    team_info = event.get("team", {})
    possession_team = event.get("possession_team", {})
    play_pattern = event.get("play_pattern", {})
    player_info = event.get("player", {})
    location = event.get("location") or []

    shot_info = event.get("shot", {}) or {}
    shot_outcome = shot_info.get("outcome", {}) or {}
    shot_xg = shot_info.get("statsbomb_xg", None)

    type_name = type_info.get("name", "")
    is_shot = 1 if type_name == "Shot" else 0

    # Rule 1: shot_xg only for shot events
    if is_shot:
        shot_xg_val = float(shot_xg) if shot_xg is not None else 0.0
    else:
        shot_xg_val = None

    # Rule 2: shot_outcome_name only for shots
    shot_outcome_name = shot_outcome.get("name", None) if is_shot else None

    # Rule 3: is_goal = 1 iff Shot and outcome == "Goal"
    is_goal = 1 if (is_shot and shot_outcome_name == "Goal") else 0

    return {
        "match_id": match_id,
        "event_id": event.get("id", ""),
        "index": event.get("index", None),
        "period": event.get("period", None),
        "minute": minute,
        "second": second,
        "event_time_min": minute + second / 60.0,
        "type_name": type_name,
        "team_name": team_info.get("name", ""),
        "team_id": team_info.get("id", None),
        "possession": event.get("possession", None),
        "possession_team_name": possession_team.get("name", ""),
        "play_pattern_name": play_pattern.get("name", ""),
        "player_name": player_info.get("name", ""),
        "under_pressure": event.get("under_pressure", None),
        "location_x": location[0] if len(location) > 0 else None,
        "location_y": location[1] if len(location) > 1 else None,
        "shot_xg": shot_xg_val,
        "shot_outcome_name": shot_outcome_name,
        "is_goal": is_goal,
        "is_shot": is_shot,
    }


def events_to_dataframe(events: list) -> pd.DataFrame:
    """Convert a list of raw event dicts (already injected with match_id) to a DataFrame."""
    if not events:
        return pd.DataFrame()

    match_id = events[0].get("match_id", 0)
    rows = [parse_event(e, match_id) for e in events]
    return pd.DataFrame(rows)
