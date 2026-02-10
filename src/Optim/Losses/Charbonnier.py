"""Losses/Charbonnier.py: Charbonnier loss as in Mip-NeRF360."""

import torch


def charbonnier_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Computes the Charbonnier loss as in Mip-NeRF360."""
    return (input - target).pow(2).add(eps).sqrt().mean()
