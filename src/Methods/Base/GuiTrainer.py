# -- coding: utf-8 --

"""Base/GuiTrainer.py: Basic Trainer with live GUI support."""
import typing
from copy import deepcopy
from pathlib import Path

import torch
from torch import multiprocessing as mp

import Framework
from Cameras.utils import CameraProperties
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Methods.Base.Trainer import BaseTrainer
from Methods.Base.utils import postTrainingCallback, preTrainingCallback, trainingCallback
from Logging import Logger

try:
    from ICGui.GuiConfig import LaunchConfig
    from ICGui.ModelRunners import ModelState, FPSRollingAverage
    from ICGui.Applications.AsyncLauncher import launchGuiProcess
    from ICGui.Viewers import saveScreenshot, transformGtImage

    @Framework.Configurable.configure(
        GUI=Framework.ConfigParameterList(
            ACTIVATE=False,
            RENDER_INTERVAL=5,
            GUI_STATUS_ENABLED=True,
            GUI_STATUS_INTERVAL=20,
            SKIP_GUI_SETUP=False,
            FPS_ROLLING_AVERAGE_SIZE=100,
        )
    )
    class GuiTrainer(BaseTrainer):
        """Basic Trainer with live GUI support."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fps = FPSRollingAverage(window_size=self.GUI.FPS_ROLLING_AVERAGE_SIZE)

            self._gui_config: LaunchConfig | None = None
            self._gui_renderer_overrides: dict[str, typing.Any] = {}
            self._restore_renderer_overrides: dict[str, typing.Any] = {}
            self._shared_state: ModelState | None = None
            self._gui_camera: BaseCamera | None = None
            self._gui_process: mp.Process | None = None
            self._gt_image_cache = {'idx': -1, 'rgb': None}

        @preTrainingCallback(priority=7500, active='GUI.ACTIVATE')
        @torch.no_grad()
        def initGUI(self, _, dataset: 'BaseDataset'):
            """Initializes the GUI process and the shared state between the GUI and the training process"""
            reference_sample = dataset.train()[0]
            # pylint: disable=protected-access
            self._gui_config = LaunchConfig.fromCommandLine(
                overrides={'training_config_path': Path(Framework.config._path)},
                disable_cmd_args=['training-config-path', 'checkpoint-path'],
                skip_gui_setup=self.GUI.SKIP_GUI_SETUP,
                training=True,
            )

            gui_resolution = self._gui_config.initial_resolution
            resolution_factor = self._gui_config.resolution_factor

            self._shared_state, self._gui_process = \
                launchGuiProcess(gui_config=self._gui_config, dataset=dataset,
                                 width=gui_resolution[0], height=gui_resolution[1],
                                 resolution_factor=resolution_factor, training=True)
            gui_camera_properties = CameraProperties(
                c2w=reference_sample.c2w.clone(),
                focal_x=reference_sample.focal_x * resolution_factor,
                focal_y=reference_sample.focal_y * resolution_factor,
                principal_offset_x=reference_sample.principal_offset_x * resolution_factor,
                principal_offset_y=reference_sample.principal_offset_y * resolution_factor,
                height=int(gui_resolution[1] * resolution_factor),
                width=int(gui_resolution[0] * resolution_factor))
            self._gui_camera = deepcopy(dataset.camera)
            self._gui_camera.properties = gui_camera_properties.toDefaultDevice()

        @preTrainingCallback(priority=7000, active='GUI.ACTIVATE')
        def advertiseRendererConfig(self, *_):
            """Advertises the renderer configuration options to the GUI process"""
            self._gui_renderer_overrides = {}
            config_options = []
            for key, value in self.renderer.__dict__.items():
                # Assume configuration options do not start with _ and are fully uppercase
                if key[0] == '_' or not key.isupper():
                    continue
                config_options.append((f'Renderer.{key}', value))

            self._shared_state.advertiseConfig(config_options)

        @postTrainingCallback(priority=0, active='GUI.ACTIVATE')
        def postTrainGuiLoop(self, _: int, dataset: BaseDataset):
            """Terminates the GUI process"""
            self._shared_state.is_training = False

            # Continue rendering until GUI is closed
            while self._gui_process.is_alive():
                self.renderImageGUI(0, dataset)

        def _updateConfig(self, config_changes: dict[str, typing.Any]):
            for key, value in config_changes.items():
                match key.split('.'):
                    case ['Renderer', attr]:
                        if hasattr(self.renderer, attr):
                            self._gui_renderer_overrides[attr] = value
                        else:
                            Logger.logWarning(f'Unknown renderer key {key} with value {value}')
                    case _:
                        Logger.logWarning(f'Unknown configuration key {key} with value {value}')

        def _useGuiRendererOverrides(self):
            """Swaps in the renderer overrides provided by the GUI process"""
            for key, value in self._gui_renderer_overrides.items():
                self._restore_renderer_overrides[key] = getattr(self.renderer, key)
                setattr(self.renderer, key, value)

        def _restoreRendererOverrides(self):
            """Restores the renderer overrides to their original values"""
            for key, value in self._restore_renderer_overrides.items():
                setattr(self.renderer, key, value)
            self._restore_renderer_overrides = {}

        @trainingCallback(priority=100, iteration_stride='GUI.RENDER_INTERVAL', active='GUI.ACTIVATE')
        @torch.no_grad()
        def renderImageGUI(self, iteration: int, dataset: BaseDataset) -> None:
            """Renders a single frame using the current model and send the result to the GUI process"""
            try:
                config_changes = self._shared_state.config_requests
                if config_changes:
                    self._updateConfig(config_changes)
                self._useGuiRendererOverrides()

                self.model.eval()
                # Save one screenshot per frame if requested
                screenshot_camera = self._shared_state.screenshot_camera
                if screenshot_camera is not None:
                    camera = screenshot_camera[0]
                    color_mode = screenshot_camera[1]
                    color_map = screenshot_camera[2]
                    output = self.renderer.renderImage(camera)
                    saveScreenshot(output, camera.properties.timestamp, color_mode, color_map, iteration=iteration)
                    del output

                if self._shared_state.should_terminate_training:
                    raise KeyboardInterrupt('Training terminated from GUI')

                if not self._gui_process.is_alive():
                    return  # Continue training without GUI

                # FPS calculation
                self._fps.update()

                # Receive new camera properties from the GUI process if available
                new_camera = self._shared_state.camera
                if new_camera is not None:
                    self._gui_camera = new_camera
                    self._gui_camera.properties = self._gui_camera.properties.toDefaultDevice()
                    self._gt_image_cache['idx'] = -1  # Invalidate cache

                gt_idx = self._shared_state.gt_index
                gt_split = self._shared_state.gt_split
                if gt_idx < 0:
                    # Save training camera and restore it after rendering with the GUI camera
                    training_camera = dataset.camera
                    dataset.camera = self._gui_camera
                    output = self.renderer.renderImage(dataset.camera)
                    dataset.camera = training_camera

                    output['type'] = 'render'
                else:
                    if self._gt_image_cache['idx'] != gt_idx:
                        previous_mode = dataset.mode
                        self._gt_image_cache['result'] = transformGtImage(dataset.setMode(gt_split), gt_idx,
                                                                          self._gui_camera, self._shared_state.window_size)
                        self._gt_image_cache['idx'] = gt_idx
                        dataset.setMode(previous_mode)
                    output = {**self._gt_image_cache['result'], 'type': 'gt'}

                # Send frame
                self._shared_state.frame = {
                    **output,
                    'camera': self._gui_camera,
                    **self._fps.stats
                }
            finally:
                self._restoreRendererOverrides()

        @trainingCallback(priority=5, iteration_stride='GUI.GUI_STATUS_INTERVAL', active='GUI.ACTIVATE')
        def updateTrainingStatus(self, iteration: int, _: BaseDataset):
            """Sends the current training progress to the GUI process"""
            # If GUI status should not be sent, skip this callback
            if not self.GUI.GUI_STATUS_ENABLED:
                return
            self._shared_state.training_iteration = iteration

        @trainingCallback(priority=0, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL',
                          active='GUI.ACTIVATE')
        def updateCheckpointPathIntermediary(self, iteration: int, _: BaseDataset) -> None:
            """Stores the intermediary checkpoint location in the GUI config."""
            self._gui_config.checkpoint_path = self.checkpoint_directory / f'{iteration:07d}.pt'
            self._gui_config.save()

        @postTrainingCallback(active='GUI.ACTIVATE', priority=0)
        def updateCheckpointPathFinal(self, _, __: BaseDataset) -> None:
            """Stores the fin checkpoint location in the GUI config."""
            # Requirement for the final backup to have been created in the first place
            if not Framework.config.TRAINING.BACKUP.FINAL_CHECKPOINT:
                return

            self._gui_config.checkpoint_path = self.checkpoint_directory / 'final.pt'
            self._gui_config.save()

except ImportError as e:
    Logger.logError('GUI support not available. Please initialize and update submodules to enable GUI support.')
    Logger.logError(e)
    GuiTrainer = BaseTrainer
