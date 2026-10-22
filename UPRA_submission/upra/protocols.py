from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class FrozenBackboneProtocol(Protocol):
    """Contract implemented by an upstream AudioLDM2, MusicGen, or ACE-Step wrapper."""

    native_conditioning_dim: int

    def native_conditioning(self, batch: dict[str, Any]) -> Tensor:
        """Encode batch['query_texts'] and return the native sequence [B,S,D]."""

    def native_loss(self, conditioned: Tensor, batch: dict[str, Any]) -> Tensor:
        """Use batch['target_paths'] to return the native diffusion/token/flow loss."""


@runtime_checkable
class GenerationBackboneProtocol(FrozenBackboneProtocol, Protocol):
    def native_conditioning_texts(self, prompts: list[str]) -> Tensor:
        """Encode prompts with the unchanged native text pathway."""

    def generate(
        self, conditioned: Tensor, batch: dict[str, Any], seeds: list[int],
        output_paths: list[str], generation_config: dict[str, Any],
    ) -> None:
        """Generate and write one waveform for every output path."""

    def generate_re_audio_ldm(
        self, batch: dict[str, Any], seeds: list[int], output_paths: list[str],
        generation_config: dict[str, Any],
    ) -> None:
        """Optional official Re-AudioLDM baseline entry point."""


@runtime_checkable
class EvaluationBackendProtocol(Protocol):
    """Adapter around the official frozen metric checkpoints listed in Appendix B.2."""

    def clap_audio(self, paths: list[str]) -> Tensor: ...
    def clap_text(self, texts: list[str]) -> Tensor: ...
    def passt_embedding(self, paths: list[str]) -> Tensor: ...
    def passt_posterior(self, paths: list[str]) -> Tensor: ...
    def vggish(self, paths: list[str]) -> Tensor: ...
    def audiobox(self, paths: list[str]) -> dict[str, Tensor]: ...
