"""Losses/DSSIM.py: Loss based on the structural (dis-)similarity index measure (DSSIM = 1 - SSIM)."""

from typing import Sequence, Literal
from functools import partial

import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from Thirdparty.FusedSSIM import fused_ssim


def fused_dssim(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate loss."""
    # add batch dimensions if necessary
    if input.dim() == 3:
        input = input[None]
    if target.dim() == 3:
        target = target[None]
    return 1.0 - fused_ssim(input, target)


class DSSIMLoss:
    """SSIM-based perceptual loss called DSSIM (computed as DSSIM = 1 - SSIM)."""

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: float | Sequence[float] = 1.5,
        kernel_size: int | Sequence[int] = 11,
        reduction: Literal['elementwise_mean', 'sum', 'none', None] = 'elementwise_mean',
        data_range: float | tuple[float, float] | None = 1.0,  # torchmetrics default is None
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
    ) -> None:
        """Initialize loss function."""
        self.return_full_image = return_full_image
        self.loss_function = partial(
            structural_similarity_index_measure,
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
            data_range=data_range,
            k1=k1,
            k2=k2,
            return_full_image=return_full_image,
            return_contrast_sensitivity=False,  # not configurable for now
        )

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        if self.return_full_image:
            return 1.0 - self.loss_function(preds=input[None], target=target[None])[1][0]
        return 1.0 - self.loss_function(preds=input[None], target=target[None])
