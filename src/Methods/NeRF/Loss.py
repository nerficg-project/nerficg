# -- coding: utf-8 --

"""NeRF/Loss.py: Loss implementation for the NeRF method."""

import torch
import torchmetrics

from Cameras.utils import RayPropertySlice
from Optim.Losses.Base import BaseLoss


class NeRFLoss(BaseLoss):
    """Defines a class for all sub-losses of the NeRF method."""

    def __init__(self, lambda_color: float, lambda_alpha: float) -> None:
        super().__init__()
        self.addLossMetric('L2_Color', torch.nn.functional.mse_loss, lambda_color)
        self.addLossMetric('L2_Alpha', torch.nn.functional.mse_loss, lambda_alpha)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, outputs: dict[str, torch.Tensor | None], rays: torch.Tensor) -> torch.Tensor:
        """Defines loss calculation."""
        return super().forward({
            'L2_Color': {'input': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb]},
            'L2_Alpha': {'input': outputs['alpha'], 'target': rays[:, RayPropertySlice.alpha]},
            'PSNR': {'preds': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb], 'data_range': 1.0},
        })
