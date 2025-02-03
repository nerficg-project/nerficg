# -- coding: utf-8 --

"""Optim/LossesMagnitude.py: Mean 1-norm over given dim."""

from typing import Any

import torch
import torchmetrics


def magnitudeLoss(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """functional"""
    if input is None:
        return torch.tensor(0.0, requires_grad=False)
    return torch.norm(input, dim=dim, keepdim=True, p=1).mean()


class MagnitudeLoss(torchmetrics.Metric):
    """torchmetrics implementation"""
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("running_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.dim = dim

    def update(self, preds: torch.Tensor) -> None:
        """torchmetrics update override"""
        x = torch.norm(preds, dim=self.dim, keepdim=True, p=1)
        self.running_sum += x.sum()
        self.total += x.numel()

    def compute(self) -> torch.Tensor:
        """torchmetrics compute override"""
        return (self.running_sum / self.total) if self.total > 0 else (torch.tensor(0.0))
