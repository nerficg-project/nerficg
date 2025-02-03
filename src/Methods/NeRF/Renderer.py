# -- coding: utf-8 --

"""
NeRF/Renderer.py: Implementation of the renderer for the vanilla (i.e. original) NeRF.
Borrows heavily from the PyTorch NeRF reimplementation of Yenchen Lin
Source: https://github.com/yenchenlin/nerf-pytorch/
"""

import torch
from torch import Tensor

import Framework
from Cameras.utils import RayPropertySlice
from Methods.Base.Renderer import BaseRenderingComponent, BaseRenderer
from Methods.NeRF.Model import NeRF, NeRFBlock
from Methods.NeRF.utils import generateSamples, integrateRaySamples
from Methods.Base.Renderer import BaseModel
from Cameras.Base import BaseCamera


class NeRFRayRenderingComponent(BaseRenderingComponent):
    """Defines a NeRF ray rendering component used to access the NeRF model."""

    def __init__(self, scene_function: 'NeRFBlock') -> None:
        super().__init__()
        self.scene_function = scene_function

    def forward(self, rays: Tensor, camera: 'BaseCamera',
                ray_batch_size: int, num_samples: int, return_samples: bool, randomize_samples: bool,
                random_noise_densities: float) -> dict[str, Tensor | None]:
        """Generates samples from the given rays and queries the NeRF model to produce the desired outputs."""
        outputs = {'rgb': [], 'depth': [], 'depth_samples': [], 'alpha_weights': [], 'alpha': []}
        # split rays into chunks that fit into VRAM
        ray_batches: list[Tensor] = torch.split(rays, ray_batch_size, dim=0)
        background_color: Tensor = camera.background_color.to(rays.device)
        for ray_batch in ray_batches:
            depth_samples = generateSamples(
                ray_batch, num_samples, camera.near_plane, camera.far_plane, randomize_samples
            )
            positions: Tensor = ray_batch[:, None, RayPropertySlice.origin] + (
                ray_batch[:, None, RayPropertySlice.direction] * depth_samples[:, :, None])
            directions: Tensor = ray_batch[:, None, RayPropertySlice.view_direction].expand(positions.shape)
            densities, rgb = self.scene_function(
                positions.reshape(-1, 3), directions.reshape(-1, 3),
                return_rgb=True, random_noise_densities=random_noise_densities
            )
            final_rgb, final_depth, final_alpha, final_alpha_weights = integrateRaySamples(
                depth_samples, ray_batch[:, RayPropertySlice.direction],
                densities.reshape(-1, num_samples), rgb.reshape(-1, num_samples, 3), background_color
            )
            # append outputs
            outputs['rgb'].append(final_rgb)
            outputs['depth'].append(final_depth)
            outputs['alpha'].append(final_alpha)
            if return_samples:
                outputs['depth_samples'].append(depth_samples)
                outputs['alpha_weights'].append(final_alpha_weights)
        # concat ray batches
        for key in outputs:
            outputs[key] = (
                torch.cat(outputs[key], dim=0) if len(ray_batches) > 1 else outputs[key][0]
            ) if outputs[key] else None
        return outputs


@Framework.Configurable.configure(
    RAY_BATCH_SIZE=8192,
    NUM_SAMPLES=256
)
class NeRFRenderer(BaseRenderer):
    """Defines the renderer for the vanilla (i.e. original) NeRF method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [NeRF])
        self.ray_rendering_component = NeRFRayRenderingComponent.get(self.model.net)

    def renderRays(self, rays: Tensor, camera: 'BaseCamera',
                   return_samples: bool = False, randomize_samples: bool = False,
                   random_noise_densities: float = 0.0) -> dict[str, Tensor | None]:
        """Renders the given set of rays using the renderer's rendering component."""
        return self.ray_rendering_component(
            rays, camera,
            self.RAY_BATCH_SIZE, self.NUM_SAMPLES,
            return_samples, randomize_samples, random_noise_densities
        )

    def renderImage(self, camera: 'BaseCamera', to_chw: bool = False, benchmark: bool = False) -> dict[str, Tensor | None]:
        """Renders a complete image using the given camera."""
        rays: Tensor = camera.generateRays()
        rendered_rays = self.renderRays(
            rays, camera,
            return_samples=False, randomize_samples=False, random_noise_densities=0.0
        )
        # reshape rays to images
        for key in rendered_rays:
            if rendered_rays[key] is not None:
                rendered_rays[key] = rendered_rays[key].reshape(camera.properties.height, camera.properties.width, -1)
                if to_chw:
                    rendered_rays[key] = rendered_rays[key].permute((2, 0, 1))
        return rendered_rays
