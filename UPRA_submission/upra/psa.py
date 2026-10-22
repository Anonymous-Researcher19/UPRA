from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PreferenceSemanticAligner(nn.Module):
    """Paper Eqs. (13)-(15): preference fusion, semantic context and alignment."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(
        self, query: Tensor, profile: Tensor, fine_chunks: Tensor, fine_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Eq. (13). True entries in key_padding_mask are ignored.
        preference, attention = self.cross_attention(
            profile[:, None, :], fine_chunks, fine_chunks,
            key_padding_mask=~fine_mask, need_weights=True,
        )
        preference = preference[:, 0, :]
        # Eq. (14).
        personalized = self.projection(torch.cat([query, preference], dim=-1))
        # Eq. (15).
        align_loss = (1.0 - F.cosine_similarity(preference, profile, dim=-1)).mean()
        return personalized, align_loss, attention[:, 0, :]
