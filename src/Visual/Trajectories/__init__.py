"""
Visual.Trajectories is a package for adding custom camera trajectories to a dataset, enabling convenient visualization in the GUI and inference scripts.
"""

import importlib
from pathlib import Path

from Logging import Logger
from .utils import CameraTrajectory

base_path = Path(__file__).resolve().parent

for python_file in base_path.glob("*.py"):
    trajectory = python_file.stem
    if trajectory in {"__init__", "utils"}:
        continue

    try:
        importlib.import_module(f'Visual.Trajectories.{trajectory}')
    except ImportError as e:
        Logger.log_warning(f'Failed to import {trajectory}: {e}')
