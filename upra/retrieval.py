from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .text import semantic_tokens


@dataclass(frozen=True)
class RetrievalResult:
    indices: Tensor
    scores: Tensor
    implicit: Tensor
    explicit: Tensor


def _minmax(values: Tensor, eps: float = 1e-8) -> Tensor:
    lo, hi = values.min(), values.max()
    span = hi - lo
    return torch.where(span > eps, (values - lo) / span, torch.zeros_like(values))


def explicit_scores(query_tokens: Sequence[str], candidate_tokens: Sequence[Sequence[str]]) -> Tensor:
    """Paper Eq. (5), with IDF estimated within the complete candidate pool."""
    docs = [set(semantic_tokens(tags)) for tags in candidate_tokens]
    n_docs = len(docs)
    document_frequency = Counter(token for doc in docs for token in doc)
    query = set(semantic_tokens(query_tokens))
    query_weight = sum(
        math.log((n_docs + 1) / (document_frequency[token] + 1)) + 1
        for token in query
    )
    values = []
    for doc in docs:
        overlap = query & doc
        numerator = sum(math.log((n_docs + 1) / (document_frequency[t] + 1)) + 1 for t in overlap)
        values.append(numerator / (query_weight + 1e-8))
    return torch.tensor(values, dtype=torch.float32)


@torch.no_grad()
def hybrid_intent_retrieval(
    query_embedding: Tensor,
    candidate_audio: Tensor,
    query_tokens: Sequence[str],
    candidate_tokens: Sequence[Sequence[str]],
    top_k: int = 5,
    lexical_weight: float = 0.5,
) -> RetrievalResult:
    """Paper Eqs. (4)-(7); HIR is intentionally non-trainable."""
    if candidate_audio.ndim != 2 or len(candidate_audio) == 0:
        raise ValueError("candidate_audio must contain at least one [D] embedding")
    implicit = F.cosine_similarity(candidate_audio, query_embedding[None, :], dim=-1)
    explicit = explicit_scores(query_tokens, candidate_tokens).to(candidate_audio.device)
    combined = lexical_weight * _minmax(explicit) + (1.0 - lexical_weight) * _minmax(implicit)
    k = min(int(top_k), len(combined))
    # Stable ordering makes tied scores reproducible across CPU/GPU implementations.
    indices = torch.argsort(combined, descending=True, stable=True)[:k]
    scores = combined[indices]
    return RetrievalResult(indices, scores, implicit, explicit)


@torch.no_grad()
def profile_only_retrieval(profile: Tensor, candidate_audio: Tensor, top_k: int = 5) -> RetrievalResult:
    """Paper's w/o-HIR ablation: rank the full pool only by similarity to P_u."""
    scores = F.cosine_similarity(candidate_audio, profile[None, :], dim=-1)
    indices = torch.argsort(scores, descending=True, stable=True)[: min(top_k, len(scores))]
    zeros = torch.zeros_like(scores)
    return RetrievalResult(indices, scores[indices], scores, zeros)
