import json
import os
import time
import requests
import argparse

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"

def load_match_ids(matches_file):
    with open(matches_file, "r") as f:
        matches = json.load(f)

    match_ids = []
    for m in matches:
        if "match_id" in m:
            match_ids.append(m["match_id"])

    match_ids = sorted(set(match_ids))
    return match_ids


def download_event(match_id, output_dir, overwrite=False):
    url = f"{BASE_URL}/{match_id}.json"
    output_path = os.path.join(output_dir, f"{match_id}.json")

    if os.path.exists(output_path) and not overwrite:
        print(f"Skipping {match_id} (already exists)")
        return

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        with open(output_path, "w") as f:
            f.write(r.text)

        print(f"Downloaded {match_id}")

    except Exception as e:
        print(f"Failed to download {match_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download StatsBomb event files for matches")
    parser.add_argument("--matches-file", required=True, help="Path to matches JSON file (e.g., 106.json)")
    parser.add_argument("--output-dir", default="data/raw", help="Directory to save event files")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if they exist")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause between downloads (seconds)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    match_ids = load_match_ids(args.matches_file)

    print(f"Found {len(match_ids)} matches")

    for i, match_id in enumerate(match_ids, start=1):
        print(f"[{i}/{len(match_ids)}] Processing match {match_id}")
        download_event(match_id, args.output_dir, overwrite=args.overwrite)
        time.sleep(args.pause)

    print("Done!")


if __name__ == "__main__":
    main()