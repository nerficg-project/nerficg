"""Optim/lr_utils.py: Provides utility classes/functions for learning rate scheduling."""

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

