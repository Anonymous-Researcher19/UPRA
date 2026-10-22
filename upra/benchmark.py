from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .text import normalize_tag


def tag_compatibility(source: Iterable[str], candidate: Iterable[str]) -> float:
    """Paper Eq. (17): Jaccard compatibility of normalized tag sets."""
    left = {normalize_tag(tag) for tag in source if normalize_tag(tag)}
    right = {normalize_tag(tag) for tag in candidate if normalize_tag(tag)}
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def construct_proxy_benchmark(
    interactions: list[dict], mtg_tracks: list[dict], seed: int = 2023,
    users: int = 1404, min_tracks: int = 33, max_history: int = 50,
) -> list[dict]:
    """Create matched, temporally ordered profiles and the 7:1:2 user split."""
    rng = random.Random(seed)
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, track in enumerate(mtg_tracks):
        for tag in track["tags"]:
            inverted[normalize_tag(tag)].add(index)
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in interactions:
        by_user[str(row["user_id"])].append(row)
    eligible = []
    for user_id, rows in by_user.items():
        rows.sort(key=lambda row: row["timestamp"])
        used: set[int] = set()
        matched = []
        for row in rows:
            candidate_ids: set[int] = set()
            for tag in row["tags"]:
                candidate_ids.update(inverted.get(normalize_tag(tag), set()))
            scored = [
                (tag_compatibility(row["tags"], mtg_tracks[index]["tags"]), index)
                for index in sorted(candidate_ids) if index not in used
            ]
            if not scored:
                continue
            score = max(item[0] for item in scored)
            tied = [index for candidate_score, index in scored if candidate_score == score]
            index = rng.choice(tied)
            if score <= 0:
                continue
            used.add(index)
            matched.append({**mtg_tracks[index], "source_timestamp": row["timestamp"], "match_score": score})
        if len(matched) >= min_tracks:
            eligible.append((user_id, matched))
    if len(eligible) < users:
        raise ValueError(f"Only {len(eligible)} eligible users; requested {users}")
    rng.shuffle(eligible)
    n_train, n_valid = round(users * 0.7), round(users * 0.1)
    capacities = {"train": n_train, "validation": n_valid, "test": users - n_train - n_valid}
    owners: dict[str, str] = {}
    chosen: list[tuple[str, list[dict], str]] = []
    # Assign complete profiles greedily while prohibiting proxy-track leakage across splits.
    # Profiles sharing a proxy may coexist only inside the same split.
    for user_id, tracks in eligible:
        retained = tracks[:max_history + 3]
        track_ids = {str(track["track_id"]) for track in retained}
        choices = [
            split for split, remaining in capacities.items()
            if remaining > 0 and all(owners.get(track_id, split) == split for track_id in track_ids)
        ]
        if not choices:
            continue
        split = max(choices, key=lambda name: capacities[name] / {"train": n_train, "validation": n_valid, "test": users - n_train - n_valid}[name])
        chosen.append((user_id, retained, split))
        capacities[split] -= 1
        for track_id in track_ids:
            owners[track_id] = split
        if not any(capacities.values()):
            break
    if any(capacities.values()):
        raise ValueError(
            "Could not satisfy user- and proxy-track-disjoint split capacities; "
            f"unfilled={capacities}. Supply more eligible users or inspect duplicate proxy mappings."
        )
    result = []
    for user_id, retained, split in chosen:
        result.append({
            "user_id": user_id,
            "split": split,
            "history": retained[:-3],
            "targets": retained[-3:],
        })
    return result


def assert_split_integrity(profiles: list[dict]) -> None:
    """Reject both user leakage and proxy-track leakage across data splits."""
    user_owner: dict[str, str] = {}
    track_owner: dict[str, str] = {}
    for profile in profiles:
        split, user_id = profile["split"], str(profile["user_id"])
        if user_id in user_owner and user_owner[user_id] != split:
            raise ValueError(f"user leakage: {user_id}")
        user_owner[user_id] = split
        for track in profile["history"] + profile["targets"]:
            track_id = str(track["track_id"])
            if track_id in track_owner and track_owner[track_id] != split:
                raise ValueError(f"proxy-track leakage: {track_id}")
            track_owner[track_id] = split
