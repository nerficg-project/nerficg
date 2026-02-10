"""InstantNGP/utils.py: Utility functions for InstantNGP."""

import numpy as np
import torch

import Framework
from Datasets.Base import BaseDataset


def next_multiple(value: int | float, multiple: int) -> int:
    return int(((value + multiple - 1) // multiple) * multiple)


@torch.no_grad()
def log_occupancy_grids(renderer, iteration: int, dataset: BaseDataset, log_string: str = 'occupancy grid') -> None:
    """Visualize occupancy grid as point cloud in wandb."""
    dataset.train()
    cameras = []
    for view in dataset:
        c2w = view.c2w_numpy
        position = c2w[:3, 3]
        forward = c2w[:3, 2]
        cameras.append({"start": position.tolist(), "end": (position + 0.1 * forward).tolist()})
    cameras = np.array(cameras)
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
    cells = renderer.get_occupancy_grid_cells()
    boxes = []
    active_cells = [unit_cube_points * renderer.model.SCALE * 1.1 + center]
    thresh = min(renderer.model.occupancy_grid[renderer.model.occupancy_grid > 0].mean().item(), renderer.density_threshold)
    for c in range(renderer.model.cascades):
        indices, coords = cells[c]
        s = min(2 ** (c - 1), renderer.model.SCALE)
        boxes.append({
            "corners": (unit_cube_points * s + center).tolist(),
            "label": f"Cascade {c + 1}",
            "color": [255, 0, 0],
        })
        half_grid_size = s / renderer.model.RESOLUTION
        points = (coords / (renderer.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size)
        active_cells.append(points[renderer.model.occupancy_grid[c, indices] > thresh].cpu().numpy() + center)
    active_cells = np.concatenate(active_cells, axis=0)
    while active_cells.shape[0] > 200_000:
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
