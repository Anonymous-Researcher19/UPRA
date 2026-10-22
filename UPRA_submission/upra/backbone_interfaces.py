from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .backbones import PreferenceConditioningAdapter


@dataclass
class BackboneConditioningInterface:
    """Tensor-level interface shared by the three frozen paper backbones."""

    adapter: PreferenceConditioningAdapter
    eta: float = 0.6

    def append_preference(self, native_sequence: Tensor, personalized: Tensor) -> Tensor:
        if native_sequence.ndim != 3:
            raise ValueError("native conditioning must have shape [batch, sequence, width]")
        return self.adapter(native_sequence, personalized, self.eta)


class AudioLDM2Conditioning(BackboneConditioningInterface):
    """Append to the sequence consumed by AudioLDM2 cross-attention."""

    def condition_cross_attention(self, cross_attention_sequence: Tensor, personalized: Tensor) -> Tensor:
        return self.append_preference(cross_attention_sequence, personalized)


class MusicGenConditioning(BackboneConditioningInterface):
    """Append to MusicGen's text-conditioning sequence before autoregressive decoding."""

    def condition_autoregressive_decoder(self, text_sequence: Tensor, personalized: Tensor) -> Tensor:
        return self.append_preference(text_sequence, personalized)


class ACEStepConditioning(BackboneConditioningInterface):
    """Append to the ACE-Step 1.5 diffusion-Transformer conditioning sequence."""

    def condition_diffusion_transformer(self, diffusion_sequence: Tensor, personalized: Tensor) -> Tensor:
        return self.append_preference(diffusion_sequence, personalized)
