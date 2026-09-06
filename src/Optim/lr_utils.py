"""Optim/lr_utils.py: Provides utility classes/functions for learning rate scheduling."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LRDecayPolicy(object):
    """Allows for flexible definition of a decay policy for a learning rate."""
    lr_init: float = 1.0
    lr_final: float = 1.0
    lr_delay_steps: int = 0
    lr_delay_mult: float = 1.0
    max_steps: int = 1_000_000

    # taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py#L78
    def __call__(self, iteration: int) -> float:
        """Calculates learning rate for the given iteration."""
        if iteration < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # disable this parameter
            return 0.0
        if self.lr_delay_steps > 0 and iteration < self.lr_delay_steps:
            # reverse cosine delay
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(iteration / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(iteration / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return float(delay_rate * log_lerp)


@dataclass(frozen=True)
class CosineDecayPolicy(object):
    """Cosine decay from lr_init to lr_final over [start_iteration, end_iteration); before the decay
    the learning rate is constant at lr_warmup (defaulting to lr_init), after it the learning rate is zero."""
    lr_init: float = 1.0
    lr_final: float = 0.0
    start_iteration: int = 0
    end_iteration: int = 1_000_000
    lr_warmup: float | None = None

    def __call__(self, iteration: int) -> float:
        """Calculates learning rate for the given iteration."""
        if iteration < self.start_iteration:
            return self.lr_init if self.lr_warmup is None else self.lr_warmup
        if iteration >= self.end_iteration:
            return 0.0
        t = (iteration - self.start_iteration) / (self.end_iteration - self.start_iteration)
        return self.lr_final + 0.5 * (self.lr_init - self.lr_final) * (1.0 + math.cos(math.pi * t))
