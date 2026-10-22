from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _masked_minmax(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalize each row over valid chunks only."""
    lower = values.masked_fill(~mask, torch.inf).min(dim=1, keepdim=True).values
    upper = values.masked_fill(~mask, -torch.inf).max(dim=1, keepdim=True).values
    normalized = (values - lower) / (upper - lower + eps)
    return torch.where(mask, normalized, torch.zeros_like(normalized))


class IntentGuidedPreferenceDistiller(nn.Module):
    """Paper Eqs. (8)-(12): score, normalize, threshold and weight chunks."""

    def __init__(self, dim: int, beta: float = 0.5):
        super().__init__()
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.beta = float(beta)

    def forward(
        self, query: Tensor, profile: Tensor, chunks: Tensor, chunk_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # chunks [B,K,L,D], chunk_mask [B,K,L]
        b, k, length, dim = chunks.shape
        flat = chunks.reshape(b, k * length, dim)
        mask = chunk_mask.reshape(b, k * length)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each sample must contain at least one valid retrieved chunk")
        # Eq. (8): scaled query-chunk interaction.
        projected_q = self.query_projection(query)
        projected_c = self.key_projection(flat)
        interaction = torch.einsum("bd,bnd->bn", projected_q, projected_c) / math.sqrt(dim)
        # Eq. (9): independently normalize and multiply query relevance and
        # profile compatibility.
        preference = F.cosine_similarity(profile[:, None, :], flat, dim=-1)
        joint = _masked_minmax(interaction, mask) * _masked_minmax(preference, mask)
        # Eq. (10): one epsilon-stabilized distribution across all retrieved chunks.
        eps = 1e-8
        stabilized = torch.where(mask, joint + eps, torch.zeros_like(joint))
        distribution = stabilized / stabilized.sum(dim=1, keepdim=True).clamp_min(eps)
        # Eq. (11): dynamic retention threshold over valid distribution entries.
        count = mask.sum(dim=1).clamp_min(1)
        mean = distribution.sum(dim=1) / count
        centered = torch.where(mask, distribution - mean[:, None], torch.zeros_like(distribution))
        std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0.0))
        theta = mean + self.beta * std
        max_value = distribution.masked_fill(~mask, -1.0).max(dim=1).values
        cutoff = max_value - theta
        keep = mask & (distribution > cutoff[:, None])
        # Numerical/tie safeguard: retain the highest valid chunk when strict > selects none.
        empty = ~keep.any(dim=1)
        if empty.any():
            best = distribution.argmax(dim=1)
            keep[empty, best[empty]] = True
        # Eq. (12): retained evidence is weighted by its global distribution mass.
        fine = flat * distribution[..., None]
        return fine, keep, distribution
