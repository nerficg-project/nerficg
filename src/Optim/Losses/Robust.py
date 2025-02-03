# -- coding: utf-8 --

"""Optim/Robust.py: Robust loss function described in https://arxiv.org/abs/1701.03077."""

import torch


class RobustLoss:
    """General and Adaptive Robust Loss Function."""

    def __init__(
            self,
            alpha: float,
            c: float,
            min_alpha: float = -1000.0  # resembles -inf
    ) -> None:
        """Initialize loss function."""
        c_reciprocal = 1.0 / c
        if alpha == 2.0:
            self.loss_function = lambda x, y: (x - y).mul(c_reciprocal).pow(2).mul(0.5).mean()
        elif alpha == 0.0:
            self.loss_function = lambda x, y: (x - y).mul(c_reciprocal).pow(2).mul(0.5).add(1.0).log().mean()
        elif alpha < min_alpha:
            self.loss_function = lambda x, y: (1.0 - (x - y).mul(c_reciprocal).pow(2).mul(-0.5).exp()).mean()
        else:
            factor = abs(alpha - 2.0) / alpha
            exponent = alpha / 2.0
            scale = 1.0 / abs(alpha - 2.0)
            self.loss_function = lambda x, y: (x - y).mul(c_reciprocal).pow(2).mul(scale).add(1.0).pow(exponent).sub(1.0).mul(factor).mean()

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        return self.loss_function(input, target)
