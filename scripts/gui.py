#! /usr/bin/env python3

"""gui.py: Opens graphical user interface."""

import sys

import torch
import torch.multiprocessing as mp

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Datasets.Base import BaseDataset
    from Implementations import Datasets as DI
    from Logging import Logger
    try:
        from ICGui.util.Runner import launch_gui_process
        from ICGui.Backend import CheckpointRunner
        from ICGui.Applications import LaunchParser
    except ImportError as e:
        Logger.set_mode(Logger.MODE_VERBOSE)
        Logger.log_error(e.with_traceback(None))
        Logger.log_error(f'Failed to open GUI: {e}\n'
                        f'Make sure the icgui submodule is initialized and '
                        f'updated before running this script. ')
        sys.exit(1)

@torch.no_grad()
def main():
    """Main entrypoint for the GUI application."""
    Logger.set_mode(Logger.MODE_DEBUG)
    config = LaunchParser.from_command_line()
    Framework.setup(require_custom_config=True, config_path=config.training_config_path)

    dataset: BaseDataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )

    shared_state, gui_process = launch_gui_process(config, dataset)
    model_runner = CheckpointRunner(dataset, shared_state,
                                    checkpoint_path=config.checkpoint_path,
                                    initial_resolution=config.initial_resolution,
                                    initial_resolution_factor=config.resolution_factor,
                                    rolling_average_size=config.fps_rolling_average_size)
    model_runner.run(gui_process)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
