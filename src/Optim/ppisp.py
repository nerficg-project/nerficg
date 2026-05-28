"""Optim/ppisp.py: Provides a wrapper class for the PPISP module from https://arxiv.org/abs/2601.18336."""

from pathlib import Path
from itertools import accumulate

import torch

import Framework
from Logging import Logger
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Thirdparty.PPISP import PPISP, PPISPConfig, export_ppisp_report


class PPISPWrapper(torch.nn.Module):
    """A wrapper class for the PPISP module."""

    def __init__(self, config: Framework.ConfigParameterList) -> None:
        super().__init__()
        self.config = PPISPConfig(
            use_controller=config.CONTROLLER_TRAINING_STEPS > 0,
            controller_distillation=config.CONTROLLER_DISTILLATION,
        )
        self.config.controller_training_steps = config.CONTROLLER_TRAINING_STEPS  # not a native ppisp config param
        self.total_training_steps = config.CONTROLLER_TRAINING_STEPS
        self.model = None
        self.optimizers = None
        self.schedulers = None
        self.known_camera_indices = {}
        self.known_global_frame_indices = {}
        self.frames_per_camera = []

    def initialize(self, dataset: BaseDataset, n_iterations: int) -> None:
        """Initializes the PPISP module based on the given dataset."""
        # set up indexing helpers (ppisp per-frame params must be sorted by camera for correct PDF reports)
        for view in dataset:
            camera_index = self.known_camera_indices.setdefault(view.camera_index, len(self.known_camera_indices))
            if camera_index == len(self.frames_per_camera):
                self.frames_per_camera.append(0)
            self.frames_per_camera[camera_index] += 1
        per_camera_offsets = [0] + list(accumulate(self.frames_per_camera))
        for view in dataset:
            camera_index = self.known_camera_indices[view.camera_index]
            frame_idx = per_camera_offsets[camera_index]
            self.known_global_frame_indices[view.global_frame_idx] = frame_idx
            per_camera_offsets[camera_index] += 1
        n_cameras = len(self.known_camera_indices)
        n_frames = len(self.known_global_frame_indices)
        Logger.log_info(f'initializing PPISP module (cameras: {n_cameras}, total frames: {n_frames})')
        # update config based on n_iterations
        if self.config.controller_distillation:
            n_iterations += self.config.controller_training_steps
        self.total_training_steps = n_iterations
        self.config.controller_activation_ratio = max((self.total_training_steps - self.config.controller_training_steps) / self.total_training_steps, 0.0)
        # setup ppisp module, optimizers, and schedulers
        self.model = PPISP(n_cameras, n_frames, self.config)
        self.optimizers = self.model.create_optimizers()
        self.schedulers = self.model.create_schedulers(self.optimizers, n_iterations)

    def get_extra_state(self):
        """Returns extra state to be stored in the checkpoint for correct behavior during inference."""
        return {
            'total_training_steps': self.total_training_steps,
            'known_camera_indices': self.known_camera_indices,
            'known_global_frame_indices': self.known_global_frame_indices,
            'frames_per_camera': self.frames_per_camera,
        }

    def set_extra_state(self, state):
        """Sets extra state from a checkpoint for correct behavior during inference."""
        self.total_training_steps = state['total_training_steps']
        self.known_camera_indices = state['known_camera_indices']
        self.known_global_frame_indices = state['known_global_frame_indices']
        self.frames_per_camera = state['frames_per_camera']

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Custom state loading that handles dataset-dependent parameter shapes."""
        # call parent method to make sure extra state is restored correctly
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
        # restore training config for correct controller handling
        self.config.controller_activation_ratio = max((self.total_training_steps - self.config.controller_training_steps) / self.total_training_steps, 0.0)
        # load ppisp parameters from checkpoint
        model_prefix = f'{prefix}model.'
        state_dict = {key[len(model_prefix):]: value for key, value in state_dict.items() if key.startswith(model_prefix)}
        self.model = PPISP.from_state_dict(state_dict, self.config)

    def create_report(self, output_directory: Path) -> None:
        """Creates a PDF report of the PPISP parameters."""
        export_ppisp_report(
            ppisp=self.model,
            frames_per_camera=self.frames_per_camera,
            output_dir=output_directory / 'ppisp_report',
        )

    def step(self) -> None:
        """Perform one optimization step of the PPISP module."""
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad()
        for scheduler in self.schedulers:
            scheduler.step()

    def forward(self, rgb: torch.Tensor, view: View) -> torch.Tensor:
        """Process the input image based on the given view."""
        # validate input and convert to hwc if needed
        width, height = view.camera.width, view.camera.height
        input_shape = rgb.shape
        if rgb.dim() != 3:
            raise Framework.ModelError(f'expected rgb image with 3 dimensions, got {rgb.dim()}')
        has_invalid_shape = input_shape[-1] != 3 and input_shape[0] != 3
        if to_chw := (input_shape[0] == 3):
            rgb = rgb.permute(1, 2, 0)
        has_invalid_shape |= rgb.shape[0] != height or rgb.shape[1] != width
        if has_invalid_shape:
            raise Framework.ModelError(
                f'expected rgb image with shape ({height}, {width}, 3) or (3, {height}, {width}), got {input_shape}'
            )
        # note that framework defaults to view.camera_index=0 for novel views (e.g., in GUI)
        camera_index = self.known_camera_indices.get(view.camera_index)
        if camera_index is None:
            Logger.log_warning('ppisp has limited support for cameras not seen during training, defaulting to camera 0')
            camera_index = 0
        # apply ppisp
        rgb = self.model(
            rgb=rgb,
            camera_idx=camera_index,
            frame_idx=self.known_global_frame_indices.get(view.global_frame_idx),
        )
        # convert back to chw if needed
        rgb = rgb.permute(2, 0, 1) if to_chw else rgb

        return rgb
