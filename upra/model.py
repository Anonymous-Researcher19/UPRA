from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
import torch

from .ipd import IntentGuidedPreferenceDistiller
from .mpp import MultimodalPreferenceProfiler
from .psa import PreferenceSemanticAligner
from .utils import masked_mean, masked_softmax


@dataclass
class UPRAOutput:
    personalized: Tensor
    profile: Tensor
    alignment_loss: Tensor
    history_weights: Tensor
    retained_chunks: Tensor
    chunk_distribution: Tensor
    attention: Tensor


class UPRA(nn.Module):
    """Trainable MPP, IPD and PSA modules; HIR runs before this forward pass."""

    def __init__(self, dim: int = 512, num_heads: int = 8, beta: float = 0.5):
        super().__init__()
        self.mpp = MultimodalPreferenceProfiler(dim)
        self.ipd = IntentGuidedPreferenceDistiller(dim, beta=beta)
        self.psa = PreferenceSemanticAligner(dim, num_heads=num_heads)

    def forward(
        self,
        query: Tensor,
        consolidated_history_text: Tensor,
        history_text: Tensor,
        history_chunks: Tensor,
        history_mask: Tensor,
        history_chunk_mask: Tensor,
        retrieved_chunks: Tensor,
        retrieved_chunk_mask: Tensor,
        variant: str = "full",
    ) -> UPRAOutput:
        allowed = {"full", "no_mpp", "no_ipd", "ipd_track", "no_psa"}
        if variant not in allowed:
            raise ValueError(f"Unknown UPRA variant {variant!r}; expected one of {sorted(allowed)}")
        profile, history_weights, track_audio = self.mpp(
            history_text, history_chunks, history_mask, history_chunk_mask,
            consolidated_history_text,
        )
        if variant == "no_mpp":
            profile = masked_mean(track_audio, history_mask, dim=1)
            history_weights = history_mask.to(track_audio.dtype)
            history_weights = history_weights / history_weights.sum(1, keepdim=True).clamp_min(1.0)
        if variant == "ipd_track":
            track_mask = retrieved_chunk_mask.any(dim=2)
            track_evidence = masked_mean(retrieved_chunks, retrieved_chunk_mask, dim=2)[:, :, None, :]
            fine, keep, distribution = self.ipd(query, profile, track_evidence, track_mask[:, :, None])
        elif variant == "no_ipd":
            b, k, length, dim = retrieved_chunks.shape
            fine = retrieved_chunks.reshape(b, k * length, dim)
            keep = retrieved_chunk_mask.reshape(b, k * length)
            distribution = masked_softmax(torch.zeros_like(keep, dtype=fine.dtype), keep, dim=1)
        else:
            fine, keep, distribution = self.ipd(
                query, profile, retrieved_chunks, retrieved_chunk_mask
            )
        if variant == "no_psa":
            flat = retrieved_chunks.reshape(retrieved_chunks.shape[0], -1, retrieved_chunks.shape[-1])
            selected_weights = distribution * keep.to(distribution.dtype)
            personalized = torch.sum(selected_weights[..., None] * flat, dim=1)
            align_loss = personalized.new_zeros(())
            attention = selected_weights
        else:
            personalized, align_loss, attention = self.psa(query, profile, fine, keep)
        return UPRAOutput(
            personalized, profile, align_loss, history_weights, keep,
            distribution, attention,
        )
