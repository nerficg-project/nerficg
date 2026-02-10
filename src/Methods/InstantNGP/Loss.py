"""InstantNGP/Loss.py: Loss calculation for InstantNGP."""

import torch
import torchmetrics

from Datasets.utils import apply_background_color, RayBatch
from Methods.InstantNGP import InstantNGPModel
from Optim.Losses.Base import BaseLoss


class InstantNGPLoss(BaseLoss):
    def __init__(self, model: 'InstantNGPModel') -> None:
        super().__init__()
        self.add_loss_metric('MSE_Color', torch.nn.functional.mse_loss, 1.0)
        self.add_loss_metric('Weight_Decay_MLP', model.weight_decay_mlp, 1.0e-6 / 2.0)
        self.add_quality_metric('PSNR', torchmetrics.functional.peak_signal_noise_ratio)

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def forward(self, outputs: dict[str, torch.Tensor | None], rays: RayBatch, bg_color: torch.Tensor) -> torch.Tensor:
        # FIXME: integrate bg color into ray generation
        color_gt = rays.rgb if rays.alpha is None else apply_background_color(rays.rgb, rays.alpha, bg_color, is_chw=False)
        return super().forward({
            'MSE_Color': {'input': outputs['rgb'], 'target': color_gt},
            'Weight_Decay_MLP': {},
            'PSNR': {'preds': outputs['rgb'], 'target': color_gt, 'data_range': 1.0}
        })
