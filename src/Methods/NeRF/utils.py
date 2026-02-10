"""
NeRF/utils.py: Contains utility functions used for the implementation of the NeRF method.
"""

import torch

import Framework
from Logging import Logger
from Datasets.utils import RayBatch


class FrequencyEncoding(torch.nn.Module):
    """A module that performs frequency encoding with linear coefficients."""

    def __init__(self, n_inputs: int, append_input: bool):
        super().__init__()
        self.register_buffer('frequency_factors', (
            torch.linspace(start=0.0, end=n_inputs - 1.0, steps=n_inputs).exp2()[None, None, :]  # * math.pi
        ))
        self.append_input = append_input

    def __repr__(self) -> str:
        return f'FrequencyEncoding(n_frequencies={self.frequency_factors.numel()}, append_input={self.append_input}'

    def get_n_outputs(self, n_inputs: int) -> int:
        """Returns the number of outputs for the given number of inputs."""
        n_outputs = n_inputs * 2 * self.frequency_factors.numel()
        if self.append_input:
            n_outputs += n_inputs
        return n_outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the encoding to the input of shape (batch_size, n_inputs)."""
        frequencies = inputs[..., None] * self.frequency_factors
        encoded = torch.cat((torch.cos(frequencies), torch.sin(frequencies)), dim=-1).flatten(start_dim=1)
        return torch.cat([inputs, encoded], dim=-1) if self.append_input else encoded


# dictionary variable containing all available activations as well as their parameters and initial biases
ACTIVATIONS: dict[str, tuple] = {
    'relu': (torch.nn.ReLU, (True,), None),
    'softplus': (torch.nn.Softplus, (10.0,), -1.5)
}


def get_activation_function(name: str) -> tuple:
    """Returns the requested activation function, parameters and initial bias."""
    if name not in ACTIVATIONS:
        Logger.log_error(
            f'requested invalid model activation function: {name} \n'
            f'available options are: {list(ACTIVATIONS.keys())}'
        )
        raise Framework.ModelError(f'Invalid activation function "{name}"')
    return ACTIVATIONS[name]


def generate_samples(
    rays: RayBatch,
    n_samples: int,
    near_plane: float,
    far_plane: float,
    randomize_samples: bool,
) -> torch.Tensor:
    """Returns random samples (positions in space) for the given set of rays."""
    depth_samples = torch.linspace(
        near_plane, far_plane, n_samples, dtype=rays.dtype, device=rays.device
    ).expand(len(rays), n_samples)
    if randomize_samples:
        # use linear samples as interval borders for random samples
        mid_points = 0.5 * (depth_samples[..., 1:] + depth_samples[..., :-1])
        upper_border = torch.cat((mid_points, depth_samples[..., -1:]), dim=-1)
        lower_border = torch.cat((depth_samples[..., :1], mid_points), dim=-1)
        random_offsets = torch.rand(depth_samples.shape, dtype=rays.dtype, device=rays.device)
        depth_samples = lower_border + ((upper_border - lower_border) * random_offsets)
    return depth_samples


def generate_samples_from_pdf(
    bins: torch.Tensor,
    values: torch.Tensor,
    n_samples: int,
    randomize_samples: bool,
) -> torch.Tensor:
    """Returns samples from probability density function along ray."""
    bins = 0.5 * (bins[..., :-1] + bins[..., 1:])
    values = values[..., 1:-1] + 1e-5
    pdf = values / torch.sum(values, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)
    if randomize_samples:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=bins.device)
    else:
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=bins.device).expand(*cdf.shape[:-1], n_samples)
    # invert cdf
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp_min(0)
    above = inds.clamp_max(cdf.shape[-1] - 1)
    inds_g = torch.stack((below, above), dim=-1)  # (batch, N_samples, 2)

    target_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(target_shape), dim=2, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(target_shape), dim=2, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, 1.0, denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples.detach()


def integrate_samples(
    depth_samples: torch.Tensor,
    ray_directions: torch.Tensor,
    densities: torch.Tensor,
    colors: torch.Tensor,
    background_color: torch.Tensor | None,
    final_delta: float = 1.0e10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes color, depth, and alpha values from samples."""
    deltas = depth_samples[:, 1:] - depth_samples[..., :-1]
    final_delta = torch.tensor(final_delta, dtype=deltas.dtype, device=deltas.device).expand(*deltas.shape[:-1], 1)
    deltas = torch.cat((deltas, final_delta), dim=-1) * ray_directions.norm(dim=-1, keepdim=True)
    alphas = 1.0 - torch.exp(-densities * deltas)
    initial_transmittance = torch.ones(1, dtype=alphas.dtype, device=alphas.device).expand(*alphas.shape[:-1], 1)
    transmittance = torch.cumprod(torch.cat((initial_transmittance, 1.0 - alphas), dim=-1), dim=-1)
    blending_weights = alphas * transmittance[..., :-1]
    transmittance_final = transmittance[..., -1:]
    alpha_final = 1.0 - transmittance_final
    # use the commented-out depth computation when propagating gradients through the depth output
    final_depth = torch.where(transmittance_final < 1.0, torch.sum(blending_weights * depth_samples, dim=-1, keepdim=True) / alpha_final, 0.0)
    # final_depth = torch.sum(blending_weights * depth_samples, dim=-1, keepdim=True) / (alpha_final + 1.0e-12)
    final_colors = torch.sum(blending_weights[..., None] * colors, dim=-2)
    if background_color is not None:
        final_colors = final_colors + transmittance_final * background_color
    return final_colors, final_depth, alpha_final, blending_weights
