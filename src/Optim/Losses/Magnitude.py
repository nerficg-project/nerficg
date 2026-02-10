"""Losses/Magnitude.py: Mean 1-norm over given dim."""

import torch


def magnitude_loss(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if input is None:
        return torch.tensor(0.0, requires_grad=False)
    return torch.norm(input, dim=dim, keepdim=True, p=1).mean()
