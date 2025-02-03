# -- coding: utf-8 --

"""GaussianSplatting/Loss.py: GaussianSplatting training objective function."""

import torch
import torchmetrics

from Cameras.utils import CameraProperties
from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import DSSIMLoss


class GaussianSplattingLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList) -> None:
        super().__init__()
        self.addLossMetric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.addLossMetric('DSSIM_Color', DSSIMLoss(), loss_config.LAMBDA_DSSIM)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, outputs: dict[str, torch.Tensor], camera_properties: CameraProperties) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': outputs['rgb'], 'target': camera_properties.rgb},
            'DSSIM_Color': {'input': outputs['rgb'], 'target': camera_properties.rgb},
            'PSNR': {'preds': outputs['rgb'], 'target': camera_properties.rgb, 'data_range': 1.0}
        })
