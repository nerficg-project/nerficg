"""Losses/BackgroundEntropy.py: Entropy loss encouraging the (alpha) value to be 0 or 1."""

import torch


def background_entropy(input: torch.Tensor, symmetrical: bool = False) -> torch.Tensor:
    x = input.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(x * torch.log(x) + (1 - x) * torch.log(1 - x)).mean() if symmetrical else (-x * torch.log(x)).mean()
