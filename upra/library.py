from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .text import semantic_tokens


def _tokens(tags: Sequence[str]) -> list[str]:
    return semantic_tokens(tags)


def _minmax(values: Tensor) -> Tensor:
    span = values.max() - values.min()
    return (values - values.min()) / span if float(span) > 1e-8 else torch.zeros_like(values)


@dataclass(frozen=True)
class IndexedRetrievalResult:
    candidates: list[dict[str, Any]]
    trace: dict[str, Any]


class RetrievalLibrary:
    """Memory-efficient HIR index for the fixed external repository plus one user's history."""

    def __init__(self, features: dict[str, dict[str, Any]], device: str = "cpu"):
        if not features:
            raise ValueError("external retrieval library is empty")
        self.ids = sorted(features)
        self.features = features
        self.device = torch.device(device)
        audio = [torch.as_tensor(features[key]["audio_embed"], dtype=torch.float32) for key in self.ids]
        self.audio = F.normalize(torch.stack(audio).to(self.device), dim=-1)
        self.tags = [_tokens(features[key]["tags"]) for key in self.ids]
        self.document_frequency = Counter(token for tags in self.tags for token in set(tags))
        self.inverted: dict[str, Tensor] = {}
        positions: dict[str, list[int]] = defaultdict(list)
        for index, tags in enumerate(self.tags):
            for token in set(tags):
                positions[token].append(index)
        for token, indices in positions.items():
            self.inverted[token] = torch.tensor(indices, dtype=torch.long, device=self.device)
    def _query_idf(
        self, query_tokens: Sequence[str], extras: list[dict[str, Any]]
    ) -> tuple[dict[str, float], float]:
        query = set(_tokens(query_tokens))
        extra_sets = [set(_tokens(item["tags"])) for item in extras]
        n_docs = len(self.ids) + len(extras)
        weights = {}
        for token in query:
            frequency = self.document_frequency.get(token, 0)
            frequency += sum(token in tags for tags in extra_sets)
            weights[token] = math.log((n_docs + 1) / (frequency + 1)) + 1.0
        return weights, sum(weights.values()) + 1e-8

    def _explicit_external(self, weights: dict[str, float], denominator: float) -> Tensor:
        numerator = torch.zeros(len(self.ids), dtype=torch.float32, device=self.device)
        for token, weight in weights.items():
            indices = self.inverted.get(token)
            if indices is not None:
                numerator[indices] += weight
        return numerator / denominator

    def _explicit_extra(
        self, weights: dict[str, float], denominator: float, extras: list[dict[str, Any]]
    ) -> Tensor:
        values = []
        for item in extras:
            tags = set(_tokens(item["tags"]))
            numerator = sum(weight for token, weight in weights.items() if token in tags)
            values.append(numerator / denominator)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def query(
        self, query_embedding: Tensor, query_tokens: Sequence[str], extras: list[dict[str, Any]],
        top_k: int = 5, lexical_weight: float = 0.5,
    ) -> IndexedRetrievalResult:
        query = F.normalize(torch.as_tensor(query_embedding, dtype=torch.float32, device=self.device), dim=-1)
        implicit = self.audio @ query
        weights, denominator = self._query_idf(query_tokens, extras)
        explicit = self._explicit_external(weights, denominator)
        sources = [("external", key) for key in self.ids]
        if extras:
            extra_audio = F.normalize(torch.stack([
                torch.as_tensor(item["audio_embed"], dtype=torch.float32) for item in extras
            ]).to(self.device), dim=-1)
            implicit = torch.cat([implicit, extra_audio @ query])
            explicit = torch.cat([
                explicit,
                self._explicit_extra(weights, denominator, extras),
            ])
            sources.extend(("history", str(index)) for index in range(len(extras)))
        combined = lexical_weight * _minmax(explicit) + (1.0 - lexical_weight) * _minmax(implicit)
        indices = torch.argsort(combined, descending=True, stable=True)[: min(top_k, len(combined))]
        candidates = []
        selected_sources = []
        for index in indices.tolist():
            source, key = sources[index]
            item = dict(self.features[key]) if source == "external" else dict(extras[int(key)])
            item.setdefault("track_id", key if source == "external" else extras[int(key)]["track_id"])
            candidates.append(item)
            selected_sources.append(source)
        return IndexedRetrievalResult(candidates, {
            "selected_global_indices": indices.cpu().tolist(),
            "scores": combined[indices].cpu().tolist(),
            "implicit": implicit[indices].cpu().tolist(),
            "explicit": explicit[indices].cpu().tolist(),
            "sources": selected_sources,
            "library_size": len(self.ids),
            "history_candidates": len(extras),
        })
