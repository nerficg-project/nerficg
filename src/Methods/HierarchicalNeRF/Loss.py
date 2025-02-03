# -- coding: utf-8 --

"""HierarchicalNeRF/Loss.py: Loss implementation for the hierarchical NeRF method."""

import torch
import torchmetrics

from Cameras.utils import RayPropertySlice
from Optim.Losses.Base import BaseLoss


class HierarchicalNeRFLoss(BaseLoss):
    """Defines a class for all sub-losses of the hierarchical NeRF method."""

    def __init__(self, lambda_color: float, lambda_alpha: float) -> None:
        super().__init__()
        self.addLossMetric('L2_Color', torch.nn.functional.mse_loss, lambda_color)
        self.addLossMetric('L2_Color_Coarse', torch.nn.functional.mse_loss, lambda_color)
        self.addLossMetric('L2_Alpha', torch.nn.functional.mse_loss, lambda_alpha)
        self.addLossMetric('L2_Alpha_Coarse', torch.nn.functional.mse_loss, lambda_alpha)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)
        self.addQualityMetric('PSNR_coarse', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, outputs: dict[str, torch.Tensor | None], rays: torch.Tensor) -> torch.Tensor:
        """Defines loss calculation."""
        return super().forward({
            'L2_Color': {'input': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb]},
            'L2_Color_Coarse': {'input': outputs['rgb_coarse'], 'target': rays[:, RayPropertySlice.rgb]},
            'L2_Alpha': {'input': outputs['alpha'], 'target': rays[:, RayPropertySlice.alpha]},
            'L2_Alpha_Coarse': {'input': outputs['alpha_coarse'], 'target': rays[:, RayPropertySlice.alpha]},
            'PSNR': {'preds': outputs['rgb'], 'target': rays[:, RayPropertySlice.rgb], 'data_range': 1.0},
            'PSNR_coarse': {'preds': outputs['rgb_coarse'], 'target': rays[:, RayPropertySlice.rgb], 'data_range': 1.0}
        })
