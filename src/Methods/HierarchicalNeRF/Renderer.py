# -- coding: utf-8 --

"""
HierarchicalNeRF/Renderer.py: Implementation of the renderer for the hierarchical NeRF method.
Borrows heavily from the PyTorch NeRF reimplementation of Yenchen Lin
Source: https://github.com/yenchenlin/nerf-pytorch/
"""

import torch
from Datasets.Base import BaseDataset

import Framework
from Cameras.utils import RayPropertySlice
from Methods.Base.Renderer import BaseRenderingComponent, BaseRenderer
from Visual.utils import pseudoColorDepth
from Methods.NeRF.Renderer import NeRFRenderer
from Methods.HierarchicalNeRF.Model import HierarchicalNeRF
from Methods.NeRF.utils import generateSamples, integrateRaySamples
from Methods.HierarchicalNeRF.utils import generateSamplesFromPDF
from Methods.Base.Model import BaseModel
from Methods.NeRF.Model import NeRFBlock
from Cameras.Base import BaseCamera


class HierarchicalNeRFRayRenderingComponent(BaseRenderingComponent):
    """Defines a hierarchical NeRF ray rendering component used to access the coarse and fine NeRF models."""

    def __init__(self, scene_function_coarse: 'NeRFBlock', scene_function_fine: 'NeRFBlock') -> None:
        super().__init__()
        self.scene_function_coarse = scene_function_coarse
        self.scene_function_fine = scene_function_fine

    def forward(self, rays: torch.Tensor, camera: 'BaseCamera',
                ray_batch_size: int, num_samples_coarse: int, num_samples_fine: int, render_coarse: bool,
                return_samples: bool, randomize_samples: bool,
                random_noise_densities: float) -> dict[str, torch.Tensor | None]:
        """Generates samples from the given rays and queries both NeRF models to produce the desired outputs."""
        outputs = {
            'rgb_coarse': [], 'depth_coarse': [], 'depth_samples_coarse': [], 'alpha_weights_coarse': [],
            'alpha_coarse': [], 'rgb': [], 'depth': [], 'depth_samples_all': [],
            'alpha_weights_fine': [], 'alpha': []
        }
        # split rays into chunks that fit into VRAM
        ray_batches: list[torch.Tensor] = torch.split(rays, ray_batch_size, dim=0)
        background_color: torch.Tensor = camera.background_color.to(rays.device)
        for ray_batch in ray_batches:
            # first, execute coarse model to sample from density pdf
            depth_samples_coarse = generateSamples(
                ray_batch, num_samples_coarse, camera.near_plane, camera.far_plane, randomize_samples
            )
            positions_coarse: torch.Tensor = ray_batch[:, None, RayPropertySlice.origin] + (
                    ray_batch[:, None, RayPropertySlice.direction] * depth_samples_coarse[:, :, None])
            directions: torch.Tensor = ray_batch[:, None, RayPropertySlice.view_direction].expand(positions_coarse.shape)
            densities_coarse, rgb_coarse = self.scene_function_coarse(
                positions_coarse.reshape(-1, 3), directions.reshape(-1, 3),
                return_rgb=render_coarse, random_noise_densities=random_noise_densities
            )
            final_rgb_coarse, final_depth_coarse, final_alpha_coarse, final_alpha_weights_coarse = integrateRaySamples(
                depth_samples_coarse, ray_batch[:, RayPropertySlice.direction],
                densities_coarse.reshape(-1, num_samples_coarse),
                rgb_coarse.reshape(-1, num_samples_coarse, 3) if render_coarse else None, background_color
            )
            # render fine model by sampling additional points based on coarse prediction
            depth_samples_fine = generateSamplesFromPDF(
                bins=depth_samples_coarse,
                values=final_alpha_weights_coarse,
                num_samples=num_samples_fine,
                randomize_samples=randomize_samples
            )
            depth_samples_all, _ = torch.sort(torch.cat([depth_samples_coarse, depth_samples_fine], -1), -1)
            positions_all: torch.Tensor = ray_batch[:, None, RayPropertySlice.origin] + (
                    ray_batch[:, None, RayPropertySlice.direction] * depth_samples_all[:, :, None])
            directions = ray_batch[:, None, RayPropertySlice.view_direction].expand(positions_all.shape)
            densities_fine, rgb = self.scene_function_fine(
                positions_all.reshape(-1, 3), directions.reshape(-1, 3),
                return_rgb=True, random_noise_densities=random_noise_densities
            )
            final_rgb, final_depth, final_alpha, final_alpha_weights_fine = integrateRaySamples(
                depth_samples_all, ray_batch[:, RayPropertySlice.direction],
                densities_fine.reshape(ray_batch.shape[0], -1), rgb.reshape(ray_batch.shape[0], -1, 3),
                background_color)
            # append outputs
            if render_coarse:
                outputs['rgb_coarse'].append(final_rgb_coarse)
                outputs['depth_coarse'].append(final_depth_coarse)
                outputs['alpha_coarse'].append(final_alpha_coarse)
                if return_samples:
                    outputs['depth_samples_coarse'].append(depth_samples_coarse)
                    outputs['alpha_weights_coarse'].append(final_alpha_weights_coarse)
            outputs['rgb'].append(final_rgb)
            outputs['depth'].append(final_depth)
            outputs['alpha'].append(final_alpha)
            if return_samples:
                outputs['depth_samples_all'].append(depth_samples_all)
                outputs['alpha_weights_fine'].append(final_alpha_weights_fine)
        # concat ray batches
        for key in outputs:
            outputs[key] = (
                torch.cat(outputs[key], dim=0) if len(ray_batches) > 1 else outputs[key][0]
            ) if outputs[key] else None
        return outputs


@Framework.Configurable.configure(
    RAY_BATCH_SIZE=8192,
    NUM_SAMPLES_COARSE=64,
    NUM_SAMPLES_FINE=192,
    RENDER_COARSE=False
)
class HierarchicalNeRFRenderer(NeRFRenderer):
    """Defines the renderer for the hierarchical NeRF method."""

    def __init__(self, model: 'BaseModel') -> None:
        BaseRenderer.__init__(self, model, [HierarchicalNeRF])
        self.ray_rendering_component = HierarchicalNeRFRayRenderingComponent.get(self.model.coarse, self.model.fine)

    def renderRays(self, rays: torch.Tensor, camera: 'BaseCamera',
                   return_samples: bool = False, randomize_samples: bool = False,
                   random_noise_densities: float = 0.0) -> dict[str, torch.Tensor | None]:
        """Renders the given set of rays using the renderer's rendering component."""
        return self.ray_rendering_component(
            rays, camera,
            self.RAY_BATCH_SIZE, self.NUM_SAMPLES_COARSE, self.NUM_SAMPLES_FINE, self.RENDER_COARSE,
            return_samples, randomize_samples, random_noise_densities
        )

    def pseudoColorOutputs(self, outputs: dict[str, torch.Tensor | None], camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = super().pseudoColorOutputs(outputs, camera, dataset, index)
        outputs_color['rgb_coarse'] = outputs['rgb_coarse'].clamp_(0.0, 1.0)
        outputs_color['alpha_coarse'] = outputs['alpha_coarse'].expand(outputs['rgb_coarse'].shape)
        outputs_color['depth_coarse'] = pseudoColorDepth(
                color_map='TURBO',
                depth=outputs['depth_coarse'],
                near_far=None,
                alpha=outputs['alpha_coarse']
            )
        return outputs_color
