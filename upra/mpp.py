from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .utils import masked_mean, masked_softmax


class MultimodalPreferenceProfiler(nn.Module):
    """Paper Eqs. (1)-(3): global semantic/acoustic user profiling."""

    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * dim, dim)

    def forward(
        self,
        history_text: Tensor,
        history_chunks: Tensor,
        history_mask: Tensor,
        chunk_mask: Tensor,
        consolidated_history_text: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Eq. (1): one acoustic representation per historical track.
        track_audio = masked_mean(history_chunks, chunk_mask, dim=2)
        # Eq. (2): semantic representativeness and normalized weights.
        scores = F.cosine_similarity(
            history_text, consolidated_history_text[:, None, :], dim=-1
        )
        alpha = masked_softmax(scores, history_mask, dim=1)
        weighted_audio = torch.sum(alpha[..., None] * track_audio, dim=1)
        # Eq. (3): concatenate semantic summary and weighted acoustic history.
        profile = self.projection(
            torch.cat([consolidated_history_text, weighted_audio], dim=-1)
        )
        return profile, alpha, track_audio

