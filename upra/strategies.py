from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .model import UPRAOutput
from .utils import masked_mean


@dataclass(frozen=True)
class StrategyCondition:
    prompts: list[str]
    preference: Tensor | None
    eta: float


def prepare_strategy(
    name: str, batch: dict, output: UPRAOutput | None, eta: float
) -> StrategyCondition:
    """Construct the four locally implemented main-table conditioning strategies."""
    if name == "text_only":
        return StrategyCondition(batch["query_texts"], None, 0.0)
    if name == "llm_profile":
        prompts = [
            f"{query} User preference profile: {profile}"
            for query, profile in zip(batch["query_texts"], batch["history_profile_texts"])
        ]
        return StrategyCondition(prompts, None, 0.0)
    if name == "vanilla_rag":
        chunks = batch["retrieved_chunks"].flatten(1, 2)
        mask = batch["retrieved_chunk_mask"].flatten(1, 2)
        return StrategyCondition(batch["query_texts"], masked_mean(chunks, mask, dim=1), eta)
    if name == "upra":
        if output is None:
            raise ValueError("UPRA output is required for the upra strategy")
        return StrategyCondition(batch["query_texts"], output.personalized, eta)
    if name == "re_audioldm":
        raise ValueError("re_audioldm must be dispatched to the official baseline wrapper")
    raise ValueError(f"Unknown strategy: {name}")

