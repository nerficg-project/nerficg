"""Losses/Robust.py: Robust loss function described in https://arxiv.org/abs/1701.03077."""

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
        if c <= 0.0:
            raise ValueError(f'Invalid scale parameter c: expected c > 0.0, got {c}')
        if alpha == 2.0:
            scale = 1 / (2 * c ** 2)
            self.loss_function = lambda x, y: x.sub(y).square().mul(scale)
        elif alpha == 0.0:
            scale = 1 / (2 * c ** 2)
            self.loss_function = lambda x, y: x.sub(y).square().mul(scale).log1p()
        elif alpha <= min_alpha:
            scale = -1 / (2 * c ** 2)
            self.loss_function = lambda x, y: x.sub(y).square().mul(scale).expm1().neg()
        else:
            factor = abs(alpha - 2) / alpha
            exponent = alpha / 2
            scale = 1 / (c ** 2 * abs(alpha - 2))
            self.loss_function = lambda x, y: x.sub(y).square().mul(scale).log1p().mul(exponent).expm1().mul(factor)

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        return self.loss_function(input, target).mean()
