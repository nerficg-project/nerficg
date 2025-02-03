# -- coding: utf-8 --

"""InstantNGP/Loss.py: Loss calculation for InstantNGP."""

import torch
import torchmetrics

from Cameras.utils import RayPropertySlice
from Methods.InstantNGP import InstantNGPModel
from Optim.Losses.Base import BaseLoss


class InstantNGPLoss(BaseLoss):
    def __init__(self, model: 'InstantNGPModel') -> None:
        super().__init__()
        self.addLossMetric('MSE_Color', torch.nn.functional.mse_loss, 1.0)
        self.addLossMetric('Weight_Decay_MLP', model.weight_decay_mlp, 1.0e-6 / 2.0)
        self.addQualityMetric('PSNR', torchmetrics.functional.peak_signal_noise_ratio)

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def forward(self, outputs: dict[str, torch.Tensor | None], rays: torch.Tensor, bg_color: torch.Tensor) -> torch.Tensor:
        color_gt = rays[:, RayPropertySlice.rgb] + (1.0 - rays[:, RayPropertySlice.alpha]) * bg_color
        return super().forward({
            'MSE_Color': {'input': outputs['rgb'], 'target': color_gt},
            'Weight_Decay_MLP': {},
            'PSNR': {'preds': outputs['rgb'], 'target': color_gt, 'data_range': 1.0}
        })
