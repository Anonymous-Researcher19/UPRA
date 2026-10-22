from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upra.benchmark import assert_split_integrity, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark counts, chronology and leakage constraints")
    parser.add_argument("--input", required=True)
    parser.add_argument("--paper-counts", action="store_true", help="Require the exact Appendix-A counts")
    args = parser.parse_args()
    profiles = load_jsonl(args.input)
    assert_split_integrity(profiles)
    if any(not 30 <= len(profile["history"]) <= 50 for profile in profiles):
        raise ValueError("every profile must contain 30-50 history tracks")
    if any(len(profile["targets"]) != 3 for profile in profiles):
        raise ValueError("every profile must contain exactly three targets")
    for profile in profiles:
        history_time = [track["source_timestamp"] for track in profile["history"]]
        target_time = [track["source_timestamp"] for track in profile["targets"]]
        if history_time != sorted(history_time) or target_time != sorted(target_time):
            raise ValueError(f"non-monotonic proxy chronology for user {profile['user_id']}")
        if max(history_time) >= min(target_time):
            raise ValueError(f"target does not occur after history for user {profile['user_id']}")
    summary = {
        "users": len(profiles),
        "history": sum(len(profile["history"]) for profile in profiles),
        "targets": sum(len(profile["targets"]) for profile in profiles),
        "splits": dict(Counter(profile["split"] for profile in profiles)),
    }
    if args.paper_counts:
        expected = {"users": 1404, "history": 43324, "targets": 4212, "splits": {"train": 983, "validation": 140, "test": 281}}
        if summary != expected:
            raise ValueError(f"paper-count mismatch: expected {expected}, observed {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
