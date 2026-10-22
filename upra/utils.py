from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    return (x * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    return torch.where(mask, weights, torch.zeros_like(weights))
