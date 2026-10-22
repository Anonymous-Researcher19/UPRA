from __future__ import annotations

from torch import Tensor, nn


class PreferenceConditioningAdapter(nn.Module):
    """Paper Eqs. (19)-(20): map and append one personalized token."""

    def __init__(self, preference_dim: int, native_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(preference_dim, native_dim),
            nn.GELU(),
            nn.Linear(native_dim, native_dim),
        )

    def forward(self, native_conditioning: Tensor, personalized: Tensor, eta: float) -> Tensor:
        if float(eta) == 0.0:
            return native_conditioning
        token = float(eta) * self.projection(personalized)
        return __import__("torch").cat([native_conditioning, token[:, None, :]], dim=1)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def total_objective(native_generation_loss: Tensor, alignment_loss: Tensor, lambda_align: float) -> Tensor:
    """Paper Eqs. (16)/(21). The native loss is supplied by the chosen backbone."""
    return native_generation_loss + float(lambda_align) * alignment_loss

