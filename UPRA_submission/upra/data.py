from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .retrieval import hybrid_intent_retrieval


def _tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


class FeatureDataset(Dataset):
    """Precomputed CLAP feature records; audio and model weights stay external."""

    def __init__(self, path: str | Path, split: str):
        path = Path(path)
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != 1:
                raise ValueError(f"Unsupported feature package schema in {path}")
            source_records = payload["records"]
        else:
            with path.open("r", encoding="utf-8") as handle:
                source_records = [json.loads(line) for line in handle if line.strip()]
        records = []
        for row in source_records:
            validate_feature_record(row)
            if row["split"] == split:
                records.append(row)
        if not records:
            raise ValueError(f"No records for split={split!r} in {path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _pad_tracks(rows: list[list[dict[str, Any]]], dim: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n_max = max(len(row) for row in rows)
    l_max = max(len(track["chunks"]) for row in rows for track in row)
    b = len(rows)
    text = torch.zeros(b, n_max, dim)
    chunks = torch.zeros(b, n_max, l_max, dim)
    track_mask = torch.zeros(b, n_max, dtype=torch.bool)
    chunk_mask = torch.zeros(b, n_max, l_max, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, track in enumerate(row):
            if "text_embed" in track:
                text[i, j] = _tensor(track["text_embed"])
            values = _tensor(track["chunks"])
            chunks[i, j, : len(values)] = values
            track_mask[i, j] = True
            chunk_mask[i, j, : len(values)] = True
    return text, chunks, track_mask, chunk_mask


def collate_feature_records(
    records: list[dict[str, Any]], top_k: int = 5, lexical_weight: float = 0.5
) -> dict[str, Any]:
    dim = len(records[0]["query_embed"])
    history_text, history_chunks, history_mask, history_chunk_mask = _pad_tracks(
        [record["history"] for record in records], dim
    )
    selected: list[list[dict[str, Any]]] = []
    retrieval_traces = []
    for record in records:
        preselected = record.get("retrieved_candidates")
        candidates = preselected or record["candidates"]
        if preselected:
            selected.append(preselected[:top_k])
            retrieval_traces.append(record.get("retrieval", {"precomputed": True}))
            continue
        candidate_audio = torch.stack([
            _tensor(candidate["audio_embed"])
            if candidate.get("audio_embed") is not None
            else _tensor(candidate["chunks"]).mean(0)
            for candidate in candidates
        ])
        result = hybrid_intent_retrieval(
            _tensor(record["query_embed"]), candidate_audio,
            record["query_tokens"], [candidate["tags"] for candidate in candidates],
            top_k=top_k, lexical_weight=lexical_weight,
        )
        selected.append([candidates[index] for index in result.indices.tolist()])
        retrieval_traces.append({
            "candidate_ids": [candidate["track_id"] for candidate in candidates],
            "selected_indices": result.indices.tolist(),
            "scores": result.scores.tolist(),
            "implicit": result.implicit.tolist(),
            "explicit": result.explicit.tolist(),
        })
    _, retrieved_chunks, retrieved_mask, retrieved_chunk_mask = _pad_tracks(selected, dim)
    return {
        "user_ids": [record["user_id"] for record in records],
        "target_ids": [record.get("target_id") for record in records],
        "query_texts": [record["query"] for record in records],
        "history_profile_texts": [record.get("history_profile", "") for record in records],
        "target_paths": [record.get("target_path") for record in records],
        "history_paths": [[track.get("path") for track in record["history"]] for record in records],
        "query": torch.stack([_tensor(record["query_embed"]) for record in records]),
        "consolidated_history_text": torch.stack([
            _tensor(record["history_profile_embed"]) for record in records
        ]),
        "target_audio": torch.stack([_tensor(record["target_audio_embed"]) for record in records]),
        "history_text": history_text,
        "history_chunks": history_chunks,
        "history_mask": history_mask,
        "history_chunk_mask": history_chunk_mask,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunk_mask": retrieved_chunk_mask & retrieved_mask[..., None],
        "retrieval": retrieval_traces,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def assert_user_disjoint(records: Iterable[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in records:
        user, split = row["user_id"], row["split"]
        if user in seen and seen[user] != split:
            raise ValueError(f"User {user!r} occurs in both {seen[user]!r} and {split!r}")
        seen[user] = split


def validate_feature_record(row: dict[str, Any]) -> None:
    required = {
        "schema_version", "user_id", "split", "query", "query_tokens", "query_embed",
        "history_profile_embed", "target_audio_embed", "history",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Feature record is missing fields: {sorted(missing)}")
    if row["schema_version"] != 1:
        raise ValueError(f"Unsupported schema_version={row['schema_version']!r}")
    candidates = row.get("retrieved_candidates") or row.get("candidates")
    if not candidates:
        raise ValueError("Feature record requires candidates or retrieved_candidates")
    dim = len(row["query_embed"])
    vectors = [row["history_profile_embed"], row["target_audio_embed"]]
    for track in row["history"]:
        vectors.append(track["text_embed"])
        vectors.extend(track["chunks"])
    for track in candidates:
        vectors.extend(track["chunks"])
    if not all(len(vector) == dim for vector in vectors):
        raise ValueError(f"Inconsistent embedding dimensions for user {row['user_id']!r}")
