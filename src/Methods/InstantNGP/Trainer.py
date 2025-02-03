# -- coding: utf-8 --
"""InstantNGP/Trainer.py: Implementation of the trainer for the InstantNGP method."""

import torch

import Framework
from Logging import Logger
from Cameras.NDC import NDCCamera
from Datasets.Base import BaseDataset
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback
from Methods.InstantNGP.Loss import InstantNGPLoss
from Methods.InstantNGP.utils import next_multiple, logOccupancyGrids
from Optim.Samplers.DatasetSamplers import RayPoolSampler
from Optim.Samplers.ImageSamplers import RandomImageSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=50000,
    TARGET_BATCH_SIZE=262144,
    WARMUP_STEPS=256,
    DENSITY_GRID_UPDATE_INTERVAL=16,
    LEARNING_RATE=1.0e-2,
    LEARNING_RATE_DECAY_START=20000,
    LEARNING_RATE_DECAY_INTERVAL=10000,
    LEARNING_RATE_DECAY_BASE=0.33,
    ADAM_EPS=1e-15,
    USE_APEX=False,
    WANDB=Framework.ConfigParameterList(
        RENDER_OCCUPANCY_GRIDS=False,
    )
)
class InstantNGPTrainer(GuiTrainer):
    """Defines the trainer for the InstantNGP method."""

    def __init__(self, **kwargs) -> None:
        super(InstantNGPTrainer, self).__init__(**kwargs)
        self.loss = InstantNGPLoss(self.model)
        try:
            if self.USE_APEX:
                from Thirdparty.Apex import FusedAdam  # slightly faster than the PyTorch implementation
                self.optimizer = FusedAdam(self.model.parameters(), lr=self.LEARNING_RATE, eps=self.ADAM_EPS, betas=(0.9, 0.99), adam_w_mode=False)
            else:
                raise Exception
        except Exception:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, eps=self.ADAM_EPS, betas=(0.9, 0.99), fused=True)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[iteration for iteration in range(
                self.LEARNING_RATE_DECAY_START,
                self.NUM_ITERATIONS,
                self.LEARNING_RATE_DECAY_INTERVAL
            )],
            gamma=self.LEARNING_RATE_DECAY_BASE
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=128.0, growth_interval=self.NUM_ITERATIONS + 1)
        self.rays_per_batch = 2 ** 12
        self.measured_batch_size = 0
        self.sampler_train = None
        self.sampler_val = None

    @preTrainingCallback(priority=10000)
    @torch.no_grad()
    def removeBackgroundColor(self, _, dataset: 'BaseDataset') -> None:
        if (dataset.camera.background_color == 0.0).all():
            return
        # remove background color from training samples to allow training with random background colors
        for cam_properties in Logger.logProgressBar(dataset.data['train'], desc='Removing background color', leave=False):
            if cam_properties.alpha is not None:
                cam_properties.rgb.sub_((1.0 - cam_properties.alpha) * dataset.camera.background_color[:, None, None]).clamp_(0.0, 1.0)
        # recompute rays if necessary
        if dataset.PRECOMPUTE_RAYS:
            dataset.ray_collection['train'] = None
            dataset.precomputeRays(['train'])

    @preTrainingCallback(priority=1000)
    @torch.no_grad()
    def initSampler(self, _, dataset: 'BaseDataset') -> None:
        self.sampler_train = RayPoolSampler(dataset=dataset.train(), img_sampler_cls=RandomImageSampler)
        if self.RUN_VALIDATION:
            self.sampler_val = RayPoolSampler(dataset=dataset.eval(), img_sampler_cls=RandomImageSampler)

    @preTrainingCallback(priority=100)
    @torch.no_grad()
    def carveDensityGrid(self, _, dataset: 'BaseDataset') -> None:
        if not isinstance(dataset.camera, NDCCamera):
            self.renderer.carveDensityGrid(dataset.train(), subtractive=False, use_alpha=False)

    @trainingCallback(priority=1000, iteration_stride='DENSITY_GRID_UPDATE_INTERVAL')
    @torch.no_grad()
    def updateDensityGrid(self, iteration: int, _) -> None:
        self.renderer.updateDensityGrid(warmup=iteration < self.WARMUP_STEPS)

    @trainingCallback(priority=999, start_iteration='DENSITY_GRID_UPDATE_INTERVAL', iteration_stride='DENSITY_GRID_UPDATE_INTERVAL')
    @torch.no_grad()
    def updateBatchSize(self, *_) -> None:
        self.measured_batch_size /= self.DENSITY_GRID_UPDATE_INTERVAL
        self.rays_per_batch = min(next_multiple(self.rays_per_batch * self.TARGET_BATCH_SIZE / self.measured_batch_size, 256), self.TARGET_BATCH_SIZE)
        self.measured_batch_size = 0

    @trainingCallback(priority=50)
    def processTrainingSample(self, _, dataset: 'BaseDataset') -> None:
        """Performs a single training step."""
        # prepare training iteration
        self.model.train()
        self.loss.train()
        dataset.train()
        # sample ray batch
        ray_batch: torch.Tensor = self.sampler_train.get(dataset=dataset, ray_batch_size=self.rays_per_batch)['ray_batch']
        with torch.amp.autocast('cuda', enabled=True):
            # render and update
            bg_color = torch.rand(3)
            output = self.renderer.renderRays(
                rays=ray_batch,
                camera=dataset.camera,
                custom_bg_color=bg_color,
                train_mode=True)
            loss = self.loss(output, ray_batch, bg_color)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.lr_scheduler.step()
        self.measured_batch_size += output['rm_samples'].item()

    @trainingCallback(active='RUN_VALIDATION', priority=100)
    @torch.no_grad()
    def processValidationSample(self, _, dataset: 'BaseDataset') -> None:
        """Performs a single validation step."""
        self.model.eval()
        self.loss.eval()
        dataset.eval()
        # sample ray batch
        ray_batch: torch.Tensor = self.sampler_val.get(dataset=dataset, ray_batch_size=self.rays_per_batch)['ray_batch']
        with torch.amp.autocast('cuda', enabled=True):
            output = self.renderer.renderRays(
                rays=ray_batch,
                camera=dataset.camera,
                train_mode=True)
            self.loss(output, ray_batch, None)

    @trainingCallback(active='WANDB.ACTIVATE', priority=500, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Logs all losses and visualizes training and validation samples using Weights & Biases."""
        super().logWandB(iteration, dataset)
        # visualize scene and occupancy grid as point clouds
        if self.WANDB.RENDER_OCCUPANCY_GRIDS:
            logOccupancyGrids(self.renderer, iteration, dataset, 'occupancy grid')
        # commit current step
        Framework.wandb.log(data={}, commit=True)
