"""InstantNGP/Trainer.py: Trainer for the InstantNGP method."""

import torch

import Framework
from Datasets.Base import BaseDataset
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback
from Methods.InstantNGP.Loss import InstantNGPLoss
from Methods.InstantNGP.utils import next_multiple, log_occupancy_grids
from Optim.Samplers.DatasetSamplers import RayPoolSampler
from Optim.Samplers.ImageSamplers import SequentialRandomImageSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=50000,
    TARGET_BATCH_SIZE=262144,
    WARMUP_STEPS=256,
    OCCUPANCY_GRID_UPDATE_INTERVAL=16,
    LR=1e-2,
    LR_DECAY_START=20000,
    LR_DECAY_INTERVAL=10000,
    LR_DECAY_BASE=0.33,
    WANDB=Framework.ConfigParameterList(
        VISUALIZE_OCCUPANCY_GRIDS=False,
    )
)
class InstantNGPTrainer(GuiTrainer):
    """Defines the trainer for the InstantNGP method."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        try:
            from Thirdparty.Apex import FusedAdam
            # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(self.model.parameters(), lr=self.LR, eps=1e-15, betas=(0.9, 0.99), adam_w_mode=False)
        except Framework.ExtensionError:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, eps=1e-15, betas=(0.9, 0.99), fused=True)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[iteration for iteration in range(self.LR_DECAY_START, self.NUM_ITERATIONS, self.LR_DECAY_INTERVAL)],
            gamma=self.LR_DECAY_BASE,
        )
        self.grad_scaler = torch.amp.GradScaler(init_scale=128.0, growth_interval=self.NUM_ITERATIONS + 1)
        self.loss = InstantNGPLoss(self.model)
        self.rays_per_batch = 2 ** 12
        self.measured_batch_size = 0
        self.sampler_train = None
        self.sampler_val = None

    @pre_training_callback(priority=1000)
    @torch.no_grad()
    def init_samplers(self, _, dataset: 'BaseDataset') -> None:
        """Initialize sampling for training and validation."""
        self.sampler_train = RayPoolSampler(dataset=dataset.train(), img_sampler_cls=SequentialRandomImageSampler)
        if self.RUN_VALIDATION:
            self.sampler_val = RayPoolSampler(dataset=dataset.eval(), img_sampler_cls=SequentialRandomImageSampler)

    @pre_training_callback(priority=100)
    @torch.no_grad()
    def carve_occupancy_grid(self, _, dataset: 'BaseDataset') -> None:
        """Initialize occupancy grid from training views."""
        self.renderer.carve_occupancy_grid(dataset.train(), subtractive=False, use_alpha=False)

    @training_callback(priority=1000, iteration_stride='OCCUPANCY_GRID_UPDATE_INTERVAL')
    @torch.no_grad()
    def update_occupancy_grid(self, iteration: int, _) -> None:
        """Update occupancy grid."""
        self.renderer.update_occupancy_grid(warmup=iteration < self.WARMUP_STEPS)

    @training_callback(priority=999, start_iteration='OCCUPANCY_GRID_UPDATE_INTERVAL', iteration_stride='OCCUPANCY_GRID_UPDATE_INTERVAL')
    @torch.no_grad()
    def update_batch_size(self, *_) -> None:
        self.measured_batch_size /= self.OCCUPANCY_GRID_UPDATE_INTERVAL
        self.rays_per_batch = min(next_multiple(self.rays_per_batch * self.TARGET_BATCH_SIZE / self.measured_batch_size, 256), self.TARGET_BATCH_SIZE)
        self.measured_batch_size = 0

    @training_callback(priority=50)
    def training_iteration(self, _, dataset: 'BaseDataset') -> None:
        """Process a ray batch for training."""
        self.model.train()
        self.loss.train()
        dataset.train()
        ray_batch = self.sampler_train.get(dataset=dataset, ray_batch_size=self.rays_per_batch)['ray_batch']
        with torch.amp.autocast('cuda'):
            bg_color = torch.rand(3, device=ray_batch.device)
            output = self.renderer.render_rays(ray_batch, dataset.default_camera, custom_bg_color=bg_color, train_mode=True)
            loss = self.loss(output, ray_batch, bg_color)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.measured_batch_size += output['rm_samples'].item()

    @training_callback(active='RUN_VALIDATION', priority=100)
    @torch.no_grad()
    def validation_iteration(self, _, dataset: 'BaseDataset') -> None:
        """Process a ray batch for validation."""
        self.model.eval()
        self.loss.eval()
        dataset.eval()
        ray_batch = self.sampler_val.get(dataset=dataset, ray_batch_size=self.rays_per_batch)['ray_batch']
        with torch.amp.autocast('cuda'):
            output = self.renderer.render_rays(ray_batch, dataset.default_camera, train_mode=True)
            self.loss(output, ray_batch, dataset.default_camera.background_color)

    @training_callback(active='WANDB.ACTIVATE', priority=500, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Logs all losses and visualizes training and validation samples using Weights & Biases."""
        super().log_wandb(iteration, dataset)
        # visualize scene and occupancy grid as point clouds
        if self.WANDB.VISUALIZE_OCCUPANCY_GRIDS:
            log_occupancy_grids(self.renderer, iteration, dataset, 'occupancy grid')
        # commit current step
        Framework.wandb.log(data={}, commit=True)
