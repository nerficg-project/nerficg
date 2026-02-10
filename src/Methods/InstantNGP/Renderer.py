"""InstantNGP/Renderer.py: InstantNGP rendering routines partially adapted from https://github.com/kwea123/ngp_pl."""

import torch
from einops import rearrange

import Framework
from Logging import Logger
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View, RayBatch
from Visual.utils import apply_color_map
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer, BaseRenderingComponent
from Methods.InstantNGP.Model import InstantNGPModel
import Methods.InstantNGP.VolumeRenderingV2 as VolumeRenderingCuda


class InstantNGPRayRenderingComponent(BaseRenderingComponent):

    def __init__(self, model: InstantNGPModel) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def get(cls, *args) -> 'BaseRenderingComponent':
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning('InstantNGP rendering should be run in single GPU mode.')
        return super(InstantNGPRayRenderingComponent, cls).get(*args)

    def forward(
        self,
        rays: RayBatch,
        camera: BaseCamera,
        max_samples: int,
        bg_color: torch.Tensor,
        exponential_steps: bool,
        train_mode: bool,
    ) -> dict[str, torch.Tensor]:
        rays_o = rays.origin - self.model.center
        rays_d = rays.view_direction.contiguous()
        hits_t = VolumeRenderingCuda.RayAABBIntersector.apply(rays_o, rays_d, torch.zeros((1, 3), device=rays.device), self.model.half_size, 1)[1]
        hits_t[..., 0].clamp_min_(camera.near_plane)
        hits_t[..., 1].clamp_max_(camera.far_plane)
        exp_step_factor = 1 / 256 if exponential_steps else 0.0
        render_fn = self.render_rays_training if train_mode else self.render_rays_inference
        return render_fn(rays_o, rays_d, hits_t, max_samples, bg_color, exp_step_factor)

    def query_model(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate density and color for a set of spatial samples."""
        h = self.model.encoding_xyz((x - self.model.xyz_min) / self.model.xyz_size)
        sigmas = VolumeRenderingCuda.TruncExp.apply(h[:, 0])
        rgbs = self.model.color_mlp_with_encoding(torch.cat([(d * 0.5 + 0.5).to(h.dtype), h], dim=-1))
        return sigmas, rgbs

    def query_density(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate density for a set of spatial samples."""
        h = self.model.encoding_xyz((x - self.model.xyz_min) / self.model.xyz_size)
        sigmas = VolumeRenderingCuda.TruncExp.apply(h[:, 0])
        return sigmas

    @torch.amp.autocast('cuda')
    def render_rays_training(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        hits_t: torch.Tensor,
        max_samples: int,
        bg_color: torch.Tensor,
        exp_step_factor: float,
    ) -> dict[str, torch.Tensor]:
        """Renders rays for training."""
        rays_a, xyzs, dirs, deltas, ts, rm_samples = VolumeRenderingCuda.RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], self.model.occupancy_bitfield,
            self.model.cascades, self.model.SCALE,
            exp_step_factor, self.model.RESOLUTION, max_samples
        )
        sigmas, rgbs = self.query_model(xyzs, dirs)
        vr_samples, alpha, depth, rgb, ws = VolumeRenderingCuda.VolumeRenderer.apply(
            sigmas, rgbs.contiguous(), deltas, ts, rays_a, 1e-4
        )
        rgb = rgb + bg_color * (1 - alpha[:, None])
        depth = depth / (alpha + 1e-6)  # untested, more accurate alternative: torch.where(transmittance < 1.0, depth / alpha, 0.0)
        return {'rgb': rgb, 'alpha': alpha, 'depth': depth, 'rm_samples': rm_samples}

    @torch.no_grad()
    def render_rays_inference(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        hits_t: torch.Tensor,
        max_samples: int,
        bg_color: torch.Tensor,
        exp_step_factor: float,
    ) -> dict[str, torch.Tensor]:
        """Renders large amount of rays using efficient ray marching."""
        n_rays = len(rays_o)
        device = rays_o.device
        rgb = torch.zeros(n_rays, 3, device=device)
        alpha = torch.zeros(n_rays, device=device)
        depth = torch.zeros(n_rays, device=device)
        alive_indices = torch.arange(n_rays, device=device)
        min_samples = 1 if exp_step_factor == 0 else 4
        samples = 0
        while samples < max_samples:
            n_alive = len(alive_indices)
            if n_alive == 0:
                break
            n_samples = max(min(n_rays // n_alive, 64), min_samples)
            samples += n_samples
            xyzs, dirs, deltas, ts, n_eff_samples = VolumeRenderingCuda.raymarching_test(
                rays_o, rays_d, hits_t[:, 0], alive_indices,
                self.model.occupancy_bitfield, self.model.cascades,
                self.model.SCALE, exp_step_factor,
                self.model.RESOLUTION, max_samples, n_samples
            )
            xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
            dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
            valid_mask = (dirs != 0).any(dim=1)
            if valid_mask.sum() == 0:
                break
            sigmas = torch.zeros(len(xyzs), device=device)
            rgbs = torch.zeros(len(xyzs), 3, device=device)
            _sigmas, _rgbs = self.query_model(xyzs[valid_mask], dirs[valid_mask])
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.float(), _rgbs.float()
            sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=n_samples)
            rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=n_samples)
            VolumeRenderingCuda.composite_test_fw(
                sigmas, rgbs, deltas, ts,
                hits_t[:, 0], alive_indices, 1e-4,
                n_eff_samples, alpha, depth, rgb
            )
            alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays
        alpha.clamp_(0, 1)
        transmittance = 1 - alpha
        rgb += transmittance[:, None] * bg_color
        rgb.clamp_(0, 1)
        depth = torch.where(transmittance < 1.0, depth / alpha, 0.0)
        return {'rgb': rgb, 'alpha': alpha, 'depth': depth}


@Framework.Configurable.configure(
    MAX_SAMPLES=1024,
    EXPONENTIAL_STEPS=False,
    DENSITY_THRESHOLD=0.01
)
class InstantNGPRenderer(BaseRenderer):
    """Rendering routines for InstantNGP."""

    def __init__(self, model: BaseModel) -> None:
        BaseRenderer.__init__(self, model, [InstantNGPModel])
        self.ray_rendering_component = InstantNGPRayRenderingComponent.get(self.model)
        self.density_threshold = self.DENSITY_THRESHOLD * self.MAX_SAMPLES / 3 ** 0.5

    def render_rays(
        self,
        rays: RayBatch,
        camera: BaseCamera,
        train_mode: bool = False,
        custom_bg_color: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Renders the given set of rays using the renderer's rendering component."""
        outputs = self.ray_rendering_component(
            rays=rays,
            camera=camera,
            max_samples=self.MAX_SAMPLES,
            bg_color=custom_bg_color if custom_bg_color is not None else camera.background_color,
            exponential_steps=self.EXPONENTIAL_STEPS,
            train_mode=train_mode
        )
        return outputs

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for the given view."""
        rendered_rays = self.render_rays(view.get_rays(), view.camera)
        # reshape rays to images
        for key in rendered_rays:
            rendered_rays[key] = rendered_rays[key].reshape(view.camera.height, view.camera.width, -1)
            if to_chw:
                rendered_rays[key] = rendered_rays[key].permute(2, 0, 1)
        return rendered_rays

    @torch.no_grad()
    def get_occupancy_grid_cells(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns all cells in the occupancy grid."""
        indices = VolumeRenderingCuda.morton3D(self.model.grid_coords).long()
        cells = [(indices, self.model.grid_coords)] * self.model.cascades
        return cells

    @torch.no_grad()
    def sample_occupancy_grid(self, n_samples: int, density_threshold: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Sample n_samples uniform and occupied (density > density_threshold) cells per cascade."""
        cells = []
        for c in range(self.model.cascades):
            # uniform cells
            coords1 = torch.randint(self.model.RESOLUTION, (n_samples, 3), dtype=torch.int32, device=self.model.occupancy_grid.device)
            indices1 = VolumeRenderingCuda.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.model.occupancy_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (n_samples,), device=self.model.occupancy_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = VolumeRenderingCuda.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
        return cells

    @torch.no_grad()
    def carve_occupancy_grid(self, dataset: BaseDataset, subtractive: bool = False, use_alpha: bool = False) -> None:
        """
        Carves the occupancy grid using the training views.
        subtractive=False -> keep points visible in at least one view, True-> point must be visible in all views
        use_alpha=True -> use images alpha channel to carve the grid, False -> only camera frustum
        """
        Logger.log_info(f'carving occupancy grid from training views (using alpha masks: {use_alpha})')
        dataset.train()
        cells = self.get_occupancy_grid_cells()
        cell_positions_world = []
        for c in range(self.model.cascades):
            _, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = (coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size) + self.model.center
            cell_positions_world.append(xyzs_w)
        remaining_cells = torch.full_like(self.model.occupancy_grid, fill_value=subtractive, dtype=torch.bool, device=self.model.occupancy_grid.device)
        dilation_kernel_2d = torch.ones(1, 1, 3, 3)
        for view in Logger.log_progress(dataset, desc='frame', leave=False):
            if use_alpha and (alpha_gt := view.alpha) is not None:
                alpha_gt = torch.nn.functional.conv2d(alpha_gt[None], dilation_kernel_2d, padding=1)[0] > 0.0
            for c in range(self.model.cascades):
                xy_screen, _, in_frustum = view.project_points(cell_positions_world[c])
                if use_alpha and alpha_gt is not None:
                    xy_screen = torch.floor(xy_screen[in_frustum]).long()
                    alpha_values = alpha_gt[:, xy_screen[:, 1], xy_screen[:, 0]] > 0.0
                    in_frustum[in_frustum.clone()] = alpha_values
                remaining_cells[c] = remaining_cells[c] & in_frustum if subtractive else remaining_cells[c] | in_frustum
        dilation_kernel_3d = torch.ones(1, 1, 3, 3, 3)
        for c in range(self.model.cascades):
            dilated = torch.nn.functional.conv3d(
                remaining_cells[c].reshape(1, 1, self.model.RESOLUTION, self.model.RESOLUTION, self.model.RESOLUTION).float(),
                dilation_kernel_3d, padding=1
            )
            values = torch.where(dilated.flatten() > 0.0, 0.0, -1.0)
            self.model.occupancy_grid[c, cells[c][0]] = values

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def update_occupancy_grid(self, warmup: bool = False, decay: float = 0.95) -> None:
        """Updates the occupancy grid."""
        occupancy_grid_tmp = torch.zeros_like(self.model.occupancy_grid)
        if warmup:
            cells = self.get_occupancy_grid_cells()
        else:
            cells = self.sample_occupancy_grid(self.model.RESOLUTION ** 3 // 4, self.density_threshold)

        for c in range(self.model.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = (coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size)
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            occupancy_grid_tmp[c, indices] = self.ray_rendering_component.query_density(xyzs_w)
        self.model.occupancy_grid = torch.where(
            self.model.occupancy_grid < 0,
            self.model.occupancy_grid,
            torch.maximum(self.model.occupancy_grid * decay, occupancy_grid_tmp)
        )
        mean_density = self.model.occupancy_grid[self.model.occupancy_grid > 0].mean().item()
        VolumeRenderingCuda.packbits(
            self.model.occupancy_grid,
            min(mean_density, self.density_threshold),
            self.model.occupancy_bitfield
        )

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], view: View, dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = {
            'rgb': outputs['rgb'],
            'alpha': outputs['alpha'].expand_as(outputs['rgb']),
            'depth': apply_color_map(
                color_map='SPECTRAL',
                image=outputs['depth'],
                min_max=(view.camera.near_plane, view.camera.far_plane),
                mask=outputs['alpha'],
            ),
        }
        return outputs_color
