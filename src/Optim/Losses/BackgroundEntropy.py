# -- coding: utf-8 --

"""Optim/LossesBackgroundEntropy.py: Background entropy loss (encouraging alpha channel to be 0 or 1)."""

from typing import Any

import torch
import torchmetrics


def backgroundEntropy(input: torch.Tensor, symmetrical=False) -> torch.Tensor:
    """functional"""
    x = input.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(x * torch.log(x) + (1 - x) * torch.log(1 - x)).mean() if symmetrical else (-x * torch.log(x)).mean()


class BackgroundEntropyLoss(torchmetrics.Metric):
    """torchmetrics implementation"""
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, symmetrical: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("running_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.symmetrical = symmetrical

    def update(self, preds: torch.Tensor) -> None:
        """Update state with current alpha predictions."""
        x = preds.clamp(min=1e-6, max=1.0 - 1e-6)
        if self.symmetrical:
            y = -(x * torch.log(x) + (1 - x) * torch.log(1 - x)).sum()
        else:
            y = (-x * torch.log(x)).sum()
        self.running_sum += y
        self.total += preds.numel()

    def compute(self) -> torch.Tensor:
        """Computes background entropy over state."""
        return (self.running_sum / self.total) if self.total > 0 else (torch.tensor(0.0))
