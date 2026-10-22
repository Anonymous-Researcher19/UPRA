from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upra.benchmark import assert_split_integrity, construct_proxy_benchmark, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the paper's Music4All/MTG-Jamendo proxy benchmark")
    parser.add_argument("--music4all", required=True, help="JSONL: user_id,timestamp,tags")
    parser.add_argument("--mtg-jamendo", required=True, help="JSONL: track_id,path,tags")
    parser.add_argument("--output", default="data/processed/benchmark.jsonl")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--users", type=int, default=1404)
    args = parser.parse_args()
    result = construct_proxy_benchmark(
        load_jsonl(args.music4all), load_jsonl(args.mtg_jamendo),
        seed=args.seed, users=args.users,
    )
    assert_split_integrity(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(result)} user profiles to {output}")


if __name__ == "__main__":
    main()
