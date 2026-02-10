"""Losses/utils.py: Utilities for loss implementations."""

from dataclasses import dataclass, field
from typing import Any, Callable

import Framework
import torch
import torchmetrics


@dataclass
class QualityMetricItem:
    """Used to store quality metrics in BaseLoss, only for training evaluation in wandb"""
    name: str
    metric_func: Callable

    _running_sum: list[torch.Tensor] = field(init=False)
    _num_iters: list[int] = field(init=False)

    def __post_init__(self):
        self.reset()

    def _call_metric(self, kwargs: Any) -> torch.Tensor:
        return self.metric_func(**kwargs)

    def reset(self) -> None:
        self._running_sum = [torch.tensor(0.0, requires_grad=False), torch.tensor(0.0, requires_grad=False)]
        self._num_iters = [0, 0]

    def get_average(self):
        return [self._running_sum[i].item() / self._num_iters[i] if self._num_iters[i] > 0 else 0.0 for i in range(2)]

    def apply(self, train: bool, accumulate: bool, kwargs: Any) -> torch.Tensor:
        loss_val = self._call_metric(kwargs)
        if accumulate:
            idx = 0 if train else 1
            self._running_sum[idx] += loss_val.detach()
            self._num_iters[idx] += 1
        return loss_val


@dataclass
class LossMetricItem(QualityMetricItem):
    """Used to store individual loss terms in BaseLoss"""
    weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.initial_weight = max(0.0, self.weight) if self.weight is not None else 0.0
        self.weight = self.initial_weight
        if isinstance(self.metric_func, torchmetrics.Metric) and not self.metric_func.is_differentiable:
            raise Framework.LossError(f'requested loss metric {self.name} (instance of {self.metric_func.__class__.__name__}) is not differentiable')

    def _call_metric(self, kwargs: Any) -> torch.Tensor:
        if self.weight > 0.0:
            return self.metric_func(**kwargs) * self.weight
        return torch.tensor(0.0)
