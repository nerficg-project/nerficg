"""
NeRF/Renderer.py: Implementation of the renderer for the original NeRF.
Borrows heavily from the PyTorch NeRF reimplementation of Yenchen Lin
Source: https://github.com/yenchenlin/nerf-pytorch/
"""

import torch

import Framework
from Logging import Logger
from Datasets.Base import BaseDataset
from Datasets.utils import View, RayBatch
from Methods.Base.Renderer import BaseRenderingComponent, BaseRenderer
from Methods.Base.Model import BaseModel
from Methods.NeRF.Model import NeRF, NeRFBlock
from Methods.NeRF.utils import generate_samples, integrate_samples, generate_samples_from_pdf
from Cameras.Base import BaseCamera
from Visual.utils import apply_color_map


class NeRFRayRenderingComponent(BaseRenderingComponent):
    """Defines a NeRF ray rendering component used to access the NeRF model."""

    def __init__(self, coarse_nerf: NeRFBlock, nerf: NeRFBlock) -> None:
        super().__init__()
        self.coarse_nerf = coarse_nerf
        self.nerf = nerf

    def forward(
        self,
        rays: RayBatch,  # FIXME: used to be a torch.Tensor, which enables torch to distribute it for multi-GPU training
        camera: BaseCamera,
        ray_batch_size: int,
        n_samples_coarse_nerf: int,
        n_samples_nerf: int,
        randomize_samples: bool,
        random_noise_density: float,
    ) -> dict[str, torch.Tensor]:
        """Generates samples from the given rays and queries the NeRF model to produce the desired outputs."""
        use_coarse_nerf = n_samples_coarse_nerf > 0
        outputs = {'rgb': [], 'alpha': [], 'depth': []}
        if use_coarse_nerf:
            outputs |= {'rgb_coarse': [], 'alpha_coarse': [], 'depth_coarse': []}
        # split rays into chunks that fit into VRAM
        ray_batches = rays.split(ray_batch_size)
        background_color = camera.background_color.to(rays.device)
        for ray_batch in ray_batches:
            origins = ray_batch.origin
            directions = ray_batch.direction
            view_directions = ray_batch.view_direction
            if use_coarse_nerf:
                # query and render coarse NeRF
                depth_samples_coarse = generate_samples(ray_batch, n_samples_coarse_nerf, camera.near_plane, camera.far_plane, randomize_samples)
                positions_coarse = origins[:, None, :] + directions[:, None, :] * depth_samples_coarse[:, :, None]
                densities_coarse, colors_coarse = self.coarse_nerf(
                    positions_coarse.reshape(-1, 3), view_directions[:, None, :].expand_as(positions_coarse).reshape(-1, 3),
                    random_noise_density=random_noise_density
                )
                rgb_coarse, depth_coarse, alpha_coarse, blending_weights_coarse = integrate_samples(
                    depth_samples_coarse, directions, densities_coarse.reshape(-1, n_samples_coarse_nerf),
                    colors_coarse.reshape(-1, n_samples_coarse_nerf, 3), background_color
                )
                # compute additional query points based on coarse prediction
                depth_samples_fine = generate_samples_from_pdf(
                    bins=depth_samples_coarse,
                    values=blending_weights_coarse,
                    n_samples=n_samples_nerf,
                    randomize_samples=randomize_samples
                )
                depth_samples, _ = torch.sort(torch.cat((depth_samples_coarse, depth_samples_fine), dim=-1), dim=-1)
            else:
                depth_samples = generate_samples(ray_batch, n_samples_nerf, camera.near_plane, camera.far_plane, randomize_samples)
            # query and render main NeRF model
            n_samples_total = n_samples_coarse_nerf + n_samples_nerf
            positions = origins[:, None, :] + directions[:, None, :] * depth_samples[:, :, None]
            densities, colors = self.nerf(
                positions.reshape(-1, 3), view_directions[:, None, :].expand_as(positions).reshape(-1, 3),
                random_noise_density=random_noise_density
            )
            rgb, depth, alpha, blending_weights = integrate_samples(
                depth_samples, directions, densities.reshape(-1, n_samples_total),
                colors.reshape(-1, n_samples_total, 3), background_color
            )
            # append outputs
            outputs['rgb'].append(rgb)
            outputs['depth'].append(depth)
            outputs['alpha'].append(alpha)
            if use_coarse_nerf:
                outputs['rgb_coarse'].append(rgb_coarse)  # noqa
                outputs['depth_coarse'].append(depth_coarse)  # noqa
                outputs['alpha_coarse'].append(alpha_coarse)  # noqa
        # concat ray batches
        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=0) if len(ray_batches) > 1 else outputs[key][0]
        return outputs


@Framework.Configurable.configure(
    RAY_BATCH_SIZE=8192,
    N_SAMPLES=256,
    COARSE_RATIO=0.25,
)
class NeRFRenderer(BaseRenderer):
    """Defines the renderer for the original NeRF method."""

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model, [NeRF])
        if self.model.coarse_nerf is None:
            self.n_samples_coarse_nerf = 0
            self.n_samples_nerf = self.N_SAMPLES
            Logger.log_info(f'using {self.n_samples_nerf} samples per ray')
        else:
            self.n_samples_coarse_nerf = round(self.N_SAMPLES * self.COARSE_RATIO)
            self.n_samples_nerf = self.N_SAMPLES - self.n_samples_coarse_nerf
            Logger.log_info(f'using {self.n_samples_coarse_nerf} coarse and {self.n_samples_nerf} fine samples per ray')
        self.ray_rendering_component = NeRFRayRenderingComponent.get(self.model.coarse_nerf, self.model.nerf)

    def render_rays(
        self,
        rays: RayBatch,
        camera: BaseCamera,
        randomize_samples: bool = False,
        random_noise_density: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Renders the given set of rays using the renderer's rendering component."""
        return self.ray_rendering_component(
            rays, camera,
            self.RAY_BATCH_SIZE, self.n_samples_coarse_nerf, self.n_samples_nerf,
            randomize_samples, random_noise_density
        )

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders a complete image for the given view."""
        rendered_rays = self.render_rays(view.get_rays(), view.camera)
        # reshape rays to images
        for key in rendered_rays:
            rendered_rays[key] = rendered_rays[key].reshape(view.camera.height, view.camera.width, -1)
            if to_chw:
                rendered_rays[key] = rendered_rays[key].permute(2, 0, 1)
        return rendered_rays

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor | None], view: View, dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = {
            'rgb': outputs['rgb'].clamp_(0.0, 1.0),
            'alpha': outputs['alpha'].expand_as(outputs['rgb']),
            'depth': apply_color_map(
                color_map='SPECTRAL',
                image=outputs['depth'],
                min_max=(view.camera.near_plane, view.camera.far_plane),
                mask=outputs['alpha'],
            ),
        }
        if self.n_samples_coarse_nerf > 0:
            outputs_color |= {
                'rgb_coarse': outputs['rgb_coarse'].clamp_(0.0, 1.0),
                'alpha_coarse': outputs['alpha_coarse'].expand_as(outputs['rgb_coarse']),
                'depth_coarse': apply_color_map(
                    color_map='SPECTRAL',
                    image=outputs['depth_coarse'],
                    min_max=(view.camera.near_plane, view.camera.far_plane),
                    mask=outputs['alpha_coarse'],
                ),
            }
        return outputs_color

