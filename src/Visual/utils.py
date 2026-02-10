"""Visual/utils.py: Utilities for visualization tasks."""

import torch

from Visual.ColorMap import ColorMap


def apply_color_map(
    color_map: str,
    image: torch.Tensor,
    min_max: tuple[float, float] | None = None,
    mask: torch.Tensor | None = None,
    interpolate: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    """Applies a color map to the given input tensor."""
    if min_max is None:
        masked_image = image[mask > 0.99] if mask is not None else image
        min_val = masked_image.min() if masked_image.numel() > 0 else 0.0
        max_val = masked_image.max() if masked_image.numel() > 0 else 1.0
    else:
        min_val, max_val = min_max
    image = torch.clamp((image - min_val) / (max_val - min_val), min=0.0, max=1.0)
    if invert:
        image = 1.0 - image

    if color_map == 'Grayscale':
        image = image.expand((3, image.shape[1], image.shape[2]))
    else:
        image = ColorMap.apply(image, color_map, interpolate)

    if mask is not None:
        image *= mask
    return image
