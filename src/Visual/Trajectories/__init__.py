"""
Visual.Trajectories is a package for adding custom camera trajectories to a dataset, enabling convenient visualization in the gui and inference scripts.
"""

import importlib
from pathlib import Path
from .utils import CameraTrajectory as CameraTrajectory

base_path: Path = Path(__file__).resolve().parents[0]
for module in [str(i.name)[:-3] for i in base_path.iterdir() if i.is_file() and i.name not in ['__init__.py', 'utils.py']]:
    importlib.import_module(f'Visual.Trajectories.{module}')
del base_path, module
