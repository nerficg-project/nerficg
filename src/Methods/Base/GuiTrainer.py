"""Base/GuiTrainer.py: Basic Trainer with live GUI support."""

import typing
from copy import deepcopy
from pathlib import Path

import torch
from torch import multiprocessing as mp

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Methods.Base.Trainer import BaseTrainer
from Methods.Base.utils import post_training_callback, pre_training_callback, training_callback
from Logging import Logger

try:
    from ICGui.Applications import LaunchParser
    from ICGui.State import LaunchConfig, SharedState
    from ICGui.util.FPSRollingAverage import FPSRollingAverage
    from ICGui.util.Runner import launch_gui_process
    from ICGui.util.Screenshots import save_screenshot
    from ICGui.util.Transforms import transform_gt_image, transform_gt_changed


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
            self._shared_state: SharedState | None = None
            self._gui_view: View | None = None
            self._gui_process: mp.Process | None = None
            self._gt_image_cache = {'idx': -1, 'split': 'None', 'rgb': None}

        @pre_training_callback(priority=9000, active='GUI.ACTIVATE')
        @torch.no_grad()
        def _gui_init(self, _, dataset: 'BaseDataset'):
            """Initializes the GUI process and the shared state between the GUI and the training process."""
            # pylint: disable=protected-access
            self._gui_config = LaunchParser.from_command_line(
                overrides={'training_config_path': Framework.config.path},
                argparse_ignore=['training-config-path', 'checkpoint-path'],
                skip_gui_setup=self.GUI.SKIP_GUI_SETUP,
                training=True,
            )

            gui_resolution = self._gui_config.initial_resolution
            resolution_factor = self._gui_config.resolution_factor

            self._shared_state, self._gui_process = \
                launch_gui_process(gui_config=self._gui_config, dataset=dataset,
                                   width=gui_resolution[0], height=gui_resolution[1],
                                   resolution_factor=resolution_factor, training=True)
            self._gui_view = deepcopy(dataset.default_view.to_simple())
            self._gui_view.width = int(gui_resolution[0] * resolution_factor)
            self._gui_view.height = int(gui_resolution[1] * resolution_factor)
            if isinstance(self._gui_view, PerspectiveCamera):
                self._gui_view.focal_x *= resolution_factor
                self._gui_view.focal_y *= resolution_factor
                self._gui_view.center_x *= resolution_factor
                self._gui_view.center_y *= resolution_factor

        @pre_training_callback(priority=8900, active='GUI.ACTIVATE')
        def _gui_advertise_renderer_config(self, *_):
            """Advertises the renderer configuration options to the GUI process."""
            self._gui_renderer_overrides = {}
            config_options = []
            for key, value in self.renderer.__dict__.items():
                # Assume configuration options do not start with _ and are fully uppercase
                if key[0] == '_' or not key.isupper():
                    continue
                config_options.append((f'Renderer.{key}', value))

            self._shared_state.configurable_advertisements = config_options

        @post_training_callback(priority=0, active='GUI.ACTIVATE')
        def _gui_post_training_loop(self, _, dataset: BaseDataset):
            """Terminates the GUI process."""
            self._shared_state.is_training = False

            # Continue rendering until GUI is closed
            while self._gui_process.is_alive():
                self._gui_render_frame(0, dataset)

        def _gui_update_config(self, config_changes: dict[str, typing.Any]):
            for key, value in config_changes.items():
                match key.split('.'):
                    case ['Renderer', attr]:
                        if hasattr(self.renderer, attr):
                            self._gui_renderer_overrides[attr] = value
                        else:
                            Logger.log_warning(f'Unknown renderer key {key} with value {value}')
                    case ['CallbackStride', name]:
                        self._update_callback(name, iteration_stride=value)
                    case _:
                        Logger.log_warning(f'Unknown configuration key {key} with value {value}')

        def _gui_set_renderer_overrides(self):
            """Swaps in the renderer overrides provided by the GUI process."""
            for key, value in self._gui_renderer_overrides.items():
                self._restore_renderer_overrides[key] = getattr(self.renderer, key)
                setattr(self.renderer, key, value)

        def _gui_reset_renderer_overrides(self):
            """Restores the renderer overrides to their original values."""
            for key, value in self._restore_renderer_overrides.items():
                setattr(self.renderer, key, value)
            self._restore_renderer_overrides = {}

        @training_callback(priority=100, iteration_stride='GUI.RENDER_INTERVAL', active='GUI.ACTIVATE')
        @Framework.catch(_gui_reset_renderer_overrides, is_method=True)
        @torch.no_grad()
        def _gui_render_frame(self, iteration: int, dataset: BaseDataset) -> None:
            """Renders a single frame using the current model and send the result to the GUI process."""
            if not self._gui_process.is_alive():
                return  # Continue training without GUI

            config_changes = self._shared_state.configurable_changes
            if config_changes:
                self._gui_update_config(config_changes)
            self._gui_set_renderer_overrides()

            self.model.eval()
            # Save one screenshot per frame if requested
            screenshot_view = self._shared_state.screenshot_view
            if screenshot_view is not None:
                view, color_mode, color_map = screenshot_view
                output = self.renderer.render_image(view)
                save_screenshot(output, view.timestamp, color_mode, color_map, iteration=iteration)
                del output

            if self._shared_state.terminate_training:
                raise KeyboardInterrupt('Training terminated from GUI')

            # Receive new camera properties from the GUI process if available
            new_view = self._shared_state.view
            if new_view is not None:
                if isinstance(self._gui_view.camera, PerspectiveCamera) and isinstance(new_view.camera, PerspectiveCamera):
                    # Check if the new camera has relevant changes, that affect transformation
                    if transform_gt_changed(self._gui_view.camera, new_view.camera):
                        self._gt_image_cache['idx'] = -1  # Invalidate cache
                elif isinstance(self._gui_view.camera, PerspectiveCamera) ^ isinstance(new_view.camera, PerspectiveCamera):
                    self._gt_image_cache['idx'] = -1  # Camera mode changed, invalidate cache
                self._gui_view = new_view

            # TODO: Send as one unit to avoid race conditions
            gt_idx = self._shared_state.gt_index
            gt_split = self._shared_state.gt_split
            if gt_idx < 0 or not isinstance(self._gui_view.camera, PerspectiveCamera):
                self._fps.enable()
                self._fps.start_timer()
                output = {
                    **self.renderer.render_image(self._gui_view),
                    'view': self._gui_view,
                    'type': 'render'
                }
                self._fps.update()
            else:
                self._fps.disable()  # Disable FPS calculation when showing GT
                if self._gt_image_cache['idx'] != gt_idx or self._gt_image_cache['split'] != gt_split:
                    previous_mode = dataset.mode
                    self._gt_image_cache['result'] = transform_gt_image(dataset.set_mode(gt_split), gt_idx,
                                                                        self._gui_view.camera)
                    self._gt_image_cache['idx'] = gt_idx
                    self._gt_image_cache['split'] = gt_split
                    dataset.set_mode(previous_mode)
                output = {**self._gt_image_cache['result'], 'type': 'gt'}

            # Send frame
            self._shared_state.frame = {
                **output,
                **self._fps.stats
            }

        @training_callback(priority=5, iteration_stride='GUI.GUI_STATUS_INTERVAL', active='GUI.ACTIVATE')
        def _gui_update_training_status(self, iteration: int, _):
            """Sends the current training progress to the GUI process."""
            # If GUI status should not be sent, skip this callback
            if not self.GUI.GUI_STATUS_ENABLED:
                return
            self._shared_state.training_iteration = iteration

        @post_training_callback(priority=9999, active='GUI.ACTIVATE')
        def _gui_finalize_training_status(self, iteration: int, _):
            """Ensure we send a final training status update before processing post-training callbacks."""
            # If GUI status should not be sent, skip this callback
            if not self.GUI.GUI_STATUS_ENABLED:
                return
            self._shared_state.training_iteration = iteration

        @training_callback(priority=0, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL')
        @Framework.catch(is_method=True)
        def _gui_store_intermediate_checkpoint_path(self, iteration: int, _) -> None:
            """Stores the location of a created intermediate checkpoint to the GUI config,
            such that it is autofilled on the next GUI launch."""
            if not self.GUI.ACTIVATE:
                return
            self._gui_config.training_config_path = self.output_directory / 'training_config.yaml'
            self._gui_config.checkpoint_path = self.checkpoint_directory / f'{iteration:07d}.pt'
            self._gui_config.save_to_disk()

        @post_training_callback(active='BACKUP.FINAL_CHECKPOINT', priority=0)
        @Framework.catch(is_method=True)
        def _gui_store_final_checkpoint_path(self, *_) -> None:
            """Stores the location of the final checkpoint to the GUI config,
            such that it is autofilled on the next GUI launch."""
            if not self.GUI.ACTIVATE:
                return
            self._gui_config.training_config_path = self.output_directory / 'training_config.yaml'
            self._gui_config.checkpoint_path = self.checkpoint_directory / 'final.pt'
            self._gui_config.save_to_disk()

except KeyboardInterrupt as e:
# except ImportError as e:
    Logger.log_error('GUI support not available. Please initialize and update submodules to enable GUI support.')
    Logger.log_error(e)
    GuiTrainer = BaseTrainer
