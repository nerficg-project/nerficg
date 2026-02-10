"""Losses/Base.py: Base Loss class for accumulation and logging."""

from typing import Any, Callable

import torch

import Framework
from Optim.Losses.utils import QualityMetricItem, LossMetricItem


class BaseLoss(torch.nn.Module):
    """Simple configurable loss container for accumulation and wandb logging"""

    def __init__(
        self,
        loss_metrics: list[LossMetricItem] | None = None,
        quality_metrics: list[QualityMetricItem] | None = None
    ) -> None:
        super().__init__()
        self.loss_metrics: list[LossMetricItem] = loss_metrics or []
        self.quality_metrics: list[QualityMetricItem] = quality_metrics or []
        self.activate_logging: bool = Framework.config.TRAINING.WANDB.ACTIVATE

    def add_loss_metric(self, name: str, metric: Callable, weight: float = None) -> None:
        self.loss_metrics.append(LossMetricItem(
            name=name,
            metric_func=metric,
            weight=weight
        ))

    def add_quality_metric(self, name: str, metric: Callable) -> None:
        self.quality_metrics.append(QualityMetricItem(
            name=name,
            metric_func=metric,
        ))

    def reset(self) -> None:
        for item in self.loss_metrics + self.quality_metrics:
            item.reset()

    def log(self, iteration: int, log_validation: bool) -> None:
        if self.activate_logging:
            for item in self.loss_metrics + self.quality_metrics:
                val_train, val_eval = item.get_average()
                data = {'train': val_train}
                if log_validation:
                    data['eval'] = val_eval
                Framework.wandb.log({f'{item.name}': data}, step=iteration)

    def forward(self, configurations: dict[str, dict[str, Any]]) -> torch.Tensor:
        try:
            if self.activate_logging:
                with torch.no_grad():
                    for loss in self.quality_metrics:
                        loss.apply(train=self.training, accumulate=True, kwargs=configurations[loss.name])
            return sum(loss.apply(train=self.training, accumulate=True, kwargs=configurations[loss.name]) for loss in self.loss_metrics)
        except NameError:
            raise Framework.LossError(f'missing argument configuration for loss "{loss.name}"')
        except TypeError:
            raise Framework.LossError(f'invalid argument configuration for loss "{loss.name}"')
        except Exception as e:
            raise Framework.LossError(f'unexpected error occurred in loss "{loss.name}": {e}')
