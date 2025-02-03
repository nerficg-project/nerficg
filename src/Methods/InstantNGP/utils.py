# -- coding: utf-8 --

"""
InstantNGP/utils.py: Utility functions for InstantNGP.
"""

import numpy as np
import torch

import Framework
from Datasets.Base import BaseDataset


def next_multiple(value: int | float, multiple: int) -> int:
    return int(((value + multiple - 1) // multiple) * multiple)


@torch.no_grad()
def logOccupancyGrids(renderer, iteration: int, dataset: 'BaseDataset', log_string: str = 'occupancy grid') -> None:
    """visualize occupancy grid as pointcloud in wandb."""
    dataset.train()
    cameras: list[np.array] = []
    for i in range(len(dataset)):
        dataset.camera.setProperties(dataset[i])
        data = dataset.camera.getPositionAndViewdir().cpu().numpy()
        # data[:, 2] *= -1
        cameras.append({"start": data[0].tolist(), "end": (data[0] + (0.1 * data[1])).tolist()})
    cameras: np.ndarray = np.array(cameras)
    # gather boxes and active cells
    center = np.array([renderer.model.CENTER])
    unit_cube_points = np.array([
        [-1, -1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, +1],
        [+1, -1, +1],
        [+1, +1, +1]
    ])
    cells = renderer.getAllCells()
    boxes = []
    active_cells = [(unit_cube_points * renderer.model.SCALE * 1.1) + center]
    thresh = min(renderer.model.density_grid[renderer.model.density_grid > 0].mean().item(), renderer.density_threshold)
    for c in range(renderer.model.cascades):
        indices, coords = cells[c]
        s = min(2 ** (c - 1), renderer.model.SCALE)
        boxes.append(
            {
                "corners": ((unit_cube_points * s) + center).tolist(),
                "label": f"Cascade {c + 1}",
                "color": [255, 0, 0],
            }
        )
        half_grid_size = s / renderer.model.RESOLUTION
        points = (coords / (renderer.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size)
        active_cells.append((points[renderer.model.density_grid[c, indices] > thresh]).cpu().numpy() + center)
    active_cells = np.concatenate(active_cells, axis=0)
    while active_cells.shape[0] > 200000:
        active_cells = active_cells[::2]
    scene = Framework.wandb.Object3D({
        "type": "lidar/beta",
        "points": active_cells,
        "boxes": np.array(boxes),
        "vectors": cameras
        })
    Framework.wandb.log(
        data={log_string: scene},
        step=iteration
    )
