# -- coding: utf-8 --

"""
HierarchicalNeRF/utils.py: Contains utility functions used for the implementation of the HierarchicalNeRF method.
"""

import torch
from torch import Tensor


def generateSamplesFromPDF(bins: Tensor, values: Tensor, num_samples: int, randomize_samples: bool) -> Tensor:
    """Returns samples from probability density function along ray."""
    device: torch.device = bins.device
    bins = 0.5 * (bins[..., :-1] + bins[..., 1:])
    values = values[..., 1:-1] + 1e-5
    pdf = values / torch.sum(values, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if randomize_samples:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=device)
    else:
        u = torch.linspace(0., 1., steps=num_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom: Tensor = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t: Tensor = (u - cdf_g[..., 0]) / denom
    samples: Tensor = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples.detach()
