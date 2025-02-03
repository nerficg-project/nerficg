# -- coding: utf-8 --

"""
NeRF/utils.py: Contains utility functions used for the implementation of the NeRF method.
"""
import torch
from torch import Tensor

import Framework
from Logging import Logger


class FrequencyEncoding(torch.nn.Module):
    """Defines a network layer that performs frequency encoding with linear coefficients."""

    def __init__(self, encoding_length: int, append_input: bool):
        super().__init__()
        # calculate frequencies
        self.register_buffer('frequency_factors', (
                2 ** torch.linspace(start=0.0, end=encoding_length - 1.0, steps=encoding_length)
            )[None, None, :]  # * math.pi
        )
        self.append_input: bool = append_input

    def getOutputSize(self, num_inputs: int) -> int:
        """Returns the number of output nodes"""
        num_outputs: int = num_inputs * 2 * self.frequency_factors.numel()
        if self.append_input:
            num_outputs += num_inputs
        return num_outputs

    def forward(self, inputs: Tensor) -> Tensor:
        """Returns the input tensor after applying the periodic embedding to it."""
        outputs: list[Tensor] = []
        # append inputs original inputs if requested
        if self.append_input:
            outputs.append(inputs)
        frequencies: Tensor = (inputs[:, :, None] * self.frequency_factors).flatten(start_dim=1)
        # apply periodic functions over frequencies
        for periodic_function in (torch.cos, torch.sin):
            outputs.append(periodic_function(frequencies))
        return torch.cat(outputs, dim=-1)


# dictionary variable containing all available activation functions as well as their parameters and initial biases
ACTIVATION_FUNCTION_OPTIONS: dict[str, tuple] = {
    'relu': (torch.nn.ReLU, (True,), None),
    'softplus': (torch.nn.Softplus, (10.0,), -1.5)
}


def getActivationFunction(type: str) -> tuple:
    """Returns the requested activation function, parameters and initial bias."""
    # log error message and stop execution if requested key is invalid
    if type not in ACTIVATION_FUNCTION_OPTIONS:
        Logger.logError(
            f'requested invalid model activation function: {type} \n'
            f'available options are: {list(ACTIVATION_FUNCTION_OPTIONS.keys())}'
        )
        raise Framework.ModelError(f'Invalid activation function "{type}"')
    # return requested model instance
    return ACTIVATION_FUNCTION_OPTIONS[type]


def generateSamples(rays: Tensor, num_samples: int, near_plane: float, far_plane: float,
                    randomize_samples: bool) -> Tensor:
    """Returns random samples (positions in space) for the given set of rays."""
    device: torch.device = rays.device
    lin_steps: Tensor = torch.linspace(0., 1., steps=num_samples, device=device)
    lin_steps: Tensor = (near_plane * (1.0 - lin_steps)) + (far_plane * lin_steps)
    depth_samples: Tensor = lin_steps.expand([rays.shape[0], num_samples])
    if randomize_samples:
        # use linear samples as interval borders for random samples
        mid_points: Tensor = 0.5 * (depth_samples[..., 1:] + depth_samples[..., :-1])
        upper_border: Tensor = torch.cat([mid_points, depth_samples[..., -1:]], -1)
        lower_border: Tensor = torch.cat([depth_samples[..., :1], mid_points], -1)
        random_offsets: Tensor = torch.rand(depth_samples.shape, device=device)
        depth_samples: Tensor = lower_border + ((upper_border - lower_border) * random_offsets)
    return depth_samples


def integrateRaySamples(depth_samples: Tensor, ray_directions: Tensor, densities: Tensor, rgb: Tensor | None,
                        background_color: Tensor, final_distance: float=1e10) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
    """Estimates final color, depth, and alpha values from samples along ray."""
    distances: Tensor = depth_samples[..., 1:] - depth_samples[..., :-1]
    distances: Tensor = torch.cat([distances, Tensor([final_distance]).expand(distances[..., :1].shape)], dim=-1) * torch.norm(ray_directions[..., None, :], dim=-1)
    alpha: Tensor = 1.0 - torch.exp(-densities * distances)
    alpha_weights: Tensor = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
    )[:, :-1]
    alpha_final: Tensor = torch.sum(alpha_weights, dim=-1, keepdim=True)
    # render only if color is available
    final_rgb: Tensor | None
    final_depth: Tensor | None
    final_rgb = final_depth = None
    if rgb is not None:
        final_depth = torch.sum((alpha_weights / (alpha_final + 1e-8)) * depth_samples, -1)
        final_rgb = torch.sum(alpha_weights[..., None] * rgb, -2)
        if background_color is not None:
            final_rgb += ((1.0 - alpha_final) * background_color)
    return final_rgb, final_depth, alpha_final, alpha_weights
