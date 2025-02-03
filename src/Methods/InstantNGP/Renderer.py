# -- coding: utf-8 --

"""InstantNGP/Renderer.py: InstantNGP Renderering Routines"""

import torch
from einops import rearrange
from torch import Tensor

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import RayPropertySlice
from Logging import Logger
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer, BaseRenderingComponent
from Methods.InstantNGP.Model import InstantNGPModel
# Import CUDA extension containing fast rendering routines adapted from kwea123 (https://github.com/kwea123/ngp_pl)
import Methods.InstantNGP.CudaExtensions.VolumeRenderingV2 as VolumeRenderingCuda


class InstantNGPRayRenderingComponent(BaseRenderingComponent):

    def __init__(self, model: InstantNGPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, rays: Tensor, camera: BaseCamera, max_samples: int, bg_color: bool, exponential_steps: bool, train_mode: bool):
        # prepare rays for rendering
        rays_o = rays[:, RayPropertySlice.origin].contiguous() - self.model.center
        rays_d = rays[:, RayPropertySlice.view_direction].contiguous()
        _, hits_t, _ = VolumeRenderingCuda.RayAABBIntersector.apply(rays_o, rays_d, torch.zeros((1, 3), device=rays.device), self.model.half_size, 1)
        hits_t[..., 0].clamp_min_(camera.near_plane)
        hits_t[..., 1].clamp_max_(camera.far_plane)
        # render rays
        render_func = self.renderRaysTrain if train_mode else self.renderRaysTest
        kwargs = {}
        if exponential_steps:
            kwargs['exp_step_factor'] = 1 / 256
        results = render_func(rays_o, rays_d, hits_t, max_samples, bg_color, **kwargs)
        return results

    def queryModel(self, x, d, **_):
        h = self.model.encoding_xyz((x - self.model.xyz_min) / self.model.xyz_size)
        sigmas = VolumeRenderingCuda.TruncExp.apply(h[..., 0])
        rgbs = self.model.color_mlp_with_encoding(torch.cat([(d * 0.5 + 0.5).to(h.dtype), h], dim=-1))
        return sigmas, rgbs

    def queryDensity(self, x):
        h = self.model.encoding_xyz((x - self.model.xyz_min) / self.model.xyz_size)
        sigmas = VolumeRenderingCuda.TruncExp.apply(h[:, 0])
        return sigmas

    @torch.no_grad()
    def renderRaysTest(self, rays_o, rays_d, hits_t, max_samples, bg_color, **kwargs):
        """
        Renders large amount of rays using efficient ray marching
        Code adapted from pytorch InstantNGP reimplementation of kwea123 (https://github.com/kwea123/ngp_pl)
        """
        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        results = {}
        # output tensors to be filled
        N_rays = len(rays_o)
        device = rays_o.device
        opacity = torch.zeros(N_rays, device=device)
        depth = torch.zeros(N_rays, device=device)
        rgb = torch.zeros(N_rays, 3, device=device)
        samples = total_samples = 0
        alive_indices = torch.arange(N_rays, device=device)
        # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
        # otherwise, 4 is more efficient empirically
        min_samples = 1 if exp_step_factor == 0 else 4
        while samples < kwargs.get('max_samples', max_samples):
            N_alive = len(alive_indices)
            if N_alive == 0:
                break
            # the number of samples to add on each ray
            N_samples = max(min(N_rays // N_alive, 64), min_samples)
            samples += N_samples
            xyzs, dirs, deltas, ts, N_eff_samples = \
                VolumeRenderingCuda.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                                     self.model.density_bitfield, self.model.cascades,
                                                     self.model.SCALE, exp_step_factor,
                                                     self.model.RESOLUTION, max_samples, N_samples)
            total_samples += N_eff_samples.sum()
            xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
            dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
            valid_mask = ~torch.all(dirs == 0, dim=1)
            if valid_mask.sum() == 0:
                break
            sigmas = torch.zeros(len(xyzs), device=device)
            rgbs = torch.zeros(len(xyzs), 3, device=device)
            _sigmas, _rgbs = self.queryModel(xyzs[valid_mask], dirs[valid_mask], **kwargs)
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.float(), _rgbs.float()
            sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
            rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            VolumeRenderingCuda.composite_test_fw(
                sigmas, rgbs, deltas, ts,
                hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
                N_eff_samples, opacity, depth, rgb)
            alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays
        results['alpha'] = opacity
        results['depth'] = depth
        results['rgb'] = rgb + bg_color * (1 - opacity[:, None])
        results['total_samples'] = total_samples  # total samples for all rays
        return results

    @torch.amp.autocast('cuda')
    def renderRaysTrain(self, rays_o, rays_d, hits_t, max_samples, bg_color, **kwargs):
        """
        Renders training rays.
        Code adapted from pytorch InstantNGP reimplementation of kwea123 (https://github.com/kwea123/ngp_pl)
        """
        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        results = {}
        rays_a, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples'] = VolumeRenderingCuda.RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], self.model.density_bitfield,
            self.model.cascades, self.model.SCALE,
            exp_step_factor, self.model.RESOLUTION, max_samples
        )
        for k, v in kwargs.items():  # supply additional inputs, repeated per ray
            if isinstance(v, Tensor):
                kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
        sigmas, rgbs = self.queryModel(xyzs, dirs, **kwargs)
        results['vr_samples'], results['alpha'], results['depth'], results['rgb'], results['ws'] = VolumeRenderingCuda.VolumeRenderer.apply(
            sigmas, rgbs.contiguous(), results['deltas'], results['ts'], rays_a, kwargs.get('T_threshold', 1e-4)
        )
        results['rays_a'] = rays_a
        results['rgb'] = results['rgb'] + bg_color * (1 - results['alpha'][:, None])
        return results


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

    def renderRays(self, rays: Tensor, camera: 'BaseCamera', train_mode=False, custom_bg_color=None, **_) -> dict[str, Tensor | None]:
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

    def renderImage(self, camera: 'BaseCamera', to_chw: bool = False, benchmark: bool = False) -> dict[str, Tensor | None]:
        """Renders a complete image at timestep using the given camera."""
        rays: Tensor = camera.generateRays()
        rendered_rays = self.renderRays(
            rays=rays,
            camera=camera,
        )
        # reshape rays to images
        for key in rendered_rays:
            if rendered_rays[key] is not None and key not in ['total_samples']:
                rendered_rays[key] = rendered_rays[key].reshape(camera.properties.height, camera.properties.width, -1)
                if to_chw:
                    rendered_rays[key] = rendered_rays[key].permute((2, 0, 1))
        return rendered_rays

    @torch.no_grad()
    def getAllCells(self):
        indices = VolumeRenderingCuda.morton3D(self.model.grid_coords).long()
        cells = [(indices, self.model.grid_coords)] * self.model.cascades
        return cells

    @torch.no_grad()
    def sampleCells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > density_threshold
        """
        cells = []
        for c in range(self.model.cascades):
            # uniform cells
            coords1 = torch.randint(self.model.RESOLUTION, (M, 3), dtype=torch.int32, device=self.model.density_grid.device)
            indices1 = VolumeRenderingCuda.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.model.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,), device=self.model.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = VolumeRenderingCuda.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.amp.autocast('cuda', enabled=True)
    @torch.no_grad()
    def carveDensityGrid(self, dataset, subtractive=False, use_alpha=False):
        # subtractive=False -> keep points visible in at least one camera, True -> point must be visible in all cameras
        Logger.logInfo(f'carving density grid from camera poses (using alpha masks: {use_alpha})')
        dataset.train()
        cells = self.getAllCells()
        cell_positions_world = []
        for c in range(self.model.cascades):
            _, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = (coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size) + self.model.center
            cell_positions_world.append(xyzs_w)
        remaining_cells = torch.full_like(self.model.density_grid, fill_value=subtractive, dtype=torch.bool, device=self.model.density_grid.device)
        for camera_properties in Logger.logProgressBar(dataset, desc='frame', leave=False):
            dataset.camera.setProperties(camera_properties)
            alpha_img = None
            if use_alpha and camera_properties.alpha is not None:
                alpha_img = torch.nn.functional.conv2d(camera_properties.alpha[None], torch.ones(1, 1, 3, 3), padding=1)[0] > 0.0
            for c in range(self.model.cascades):
                uv, valid, _ = dataset.camera.projectPoints(cell_positions_world[c])
                if alpha_img is not None:
                    uv = torch.round(uv).long()[valid]
                    alpha_values = alpha_img[:, uv[:, 1], uv[:, 0]] > 0.0
                    valid[valid.clone()] = alpha_values
                remaining_cells[c] = torch.logical_and(remaining_cells[c], valid) if subtractive else torch.logical_or(remaining_cells[c], valid)
        for c in range(self.model.cascades):
            values = torch.where(
                torch.nn.functional.conv3d(remaining_cells[c].reshape(1, 1, self.model.RESOLUTION, self.model.RESOLUTION, self.model.RESOLUTION).float(), torch.ones(1, 1, 3, 3, 3), padding=1).flatten() > 0.0,
                0.0,
                -1.0
            )
            self.model.density_grid[c, cells[c][0]] = values

    @torch.no_grad()
    @torch.amp.autocast('cuda', enabled=True)
    def updateDensityGrid(self, warmup=False, decay=0.95):
        density_grid_tmp = torch.zeros_like(self.model.density_grid)
        if warmup:  # during the first steps
            cells = self.getAllCells()
        else:
            cells = self.sampleCells(self.model.RESOLUTION**3//4, self.density_threshold)

        # infer sigmas
        for c in range(self.model.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = (coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.ray_rendering_component.queryDensity(xyzs_w)
        self.model.density_grid = torch.where(
            self.model.density_grid < 0,
            self.model.density_grid,
            torch.maximum(self.model.density_grid * decay, density_grid_tmp)
        )
        mean_density = self.model.density_grid[self.model.density_grid > 0].mean().item()
        VolumeRenderingCuda.packbits(
            self.model.density_grid,
            min(mean_density, self.density_threshold),
            self.model.density_bitfield)
