# -- coding: utf-8 --

"""GaussianSplatting/Model.py: Implementation of the model for 3D Gaussian Splatting."""

import torch
from Thirdparty.SimpleKNN import distCUDA2

import Framework
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.Model import BaseModel
from Cameras.utils import quaternion_to_rotation_matrix
from Methods.GaussianSplatting.utils import inverse_sigmoid, LRDecayPolicy, rgb_to_sh0, extract_upper_triangular_matrix, build_covariances
from CudaUtils.MortonEncoding import morton_encode
from Optim.AdamUtils import replace_param_group_data, prune_param_groups, extend_param_groups


class Gaussians(torch.nn.Module):
    """Stores a set of points with 3D Gaussian extent."""

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.max_sh_degree = sh_degree
        self.register_parameter('_positions', None)
        self.register_parameter('_features_dc', None)
        self.register_parameter('_features_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self.register_parameter('_baked_covariances', None)
        self.densification_gradient_accum = torch.empty(0)
        self.n_observations = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.training_cameras_extent = 1.0
        # activation functions
        self.scaling_activation = torch.nn.Identity() if pretrained else torch.exp
        self.inverse_scaling_activation = torch.nn.Identity() if pretrained else torch.log
        self.opacity_activation = torch.nn.Identity() if pretrained else torch.sigmoid
        self.inverse_opacity_activation = torch.nn.Identity() if pretrained else inverse_sigmoid
        self.rotation_activation = torch.nn.Identity() if pretrained else torch.nn.functional.normalize
        self.covariance_activation = build_covariances

    @property
    def get_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales."""
        return self.scaling_activation(self._scales)

    @property
    def get_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotation matrices."""
        return self.rotation_activation(self._rotations)

    @property
    def get_positions(self) -> torch.Tensor:
        """Returns the Gaussians' means."""
        return self._positions

    @property
    def get_features(self) -> torch.Tensor:
        """Returns the Gaussians' concatenated SH features."""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities."""
        return self.opacity_activation(self._opacities)

    @property
    def get_baked_covariances(self) -> torch.Tensor:
        """Returns the Gaussians' baked covariance matrices."""
        return self._baked_covariances

    def get_covariances(self, scale_modifier: float) -> torch.Tensor:
        """Returns the Gaussians' covariance matrices without duplicate entries."""
        return extract_upper_triangular_matrix(self.covariance_activation(self.get_scales * scale_modifier, self.get_rotations))

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, training_cameras_extent: float) -> None:
        """Initializes the model from a point cloud."""
        self.training_cameras_extent = training_cameras_extent
        positions = point_cloud.positions.cuda()
        rgbs = torch.full_like(positions, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        n_initial_points = positions.shape[0]
        features = torch.zeros((n_initial_points, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device='cuda')
        features[:, :3, 0] = rgb_to_sh0(rgbs)

        Logger.logInfo(f'Number of points at initialization: {n_initial_points:,}')

        dist2 = distCUDA2(positions).clamp_min(0.0000001)
        scales = self.inverse_scaling_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros((n_initial_points, 4), device='cuda')
        rotations[:, 0] = 1

        opacities = self.inverse_opacity_activation(torch.full((n_initial_points, 1), fill_value=0.1, dtype=torch.float32, device='cuda'))

        self._positions = torch.nn.Parameter(positions.contiguous())
        self._features_dc = torch.nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        self.densification_gradient_accum = torch.zeros((n_initial_points, 1), dtype=torch.float32, device='cuda')
        self.n_observations = torch.zeros((n_initial_points, 1), dtype=torch.int32, device='cuda')

    def training_setup(self, training_wrapper) -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.PERCENT_DENSE
        param_groups = [
            {'params': [self._positions], 'lr': training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent, 'name': 'positions'},
            {'params': [self._features_dc], 'lr': training_wrapper.LEARNING_RATE_FEATURE, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': training_wrapper.LEARNING_RATE_FEATURE / 20.0, 'name': 'f_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.LEARNING_RATE_OPACITY, 'name': 'opacities'},
            {'params': [self._scales], 'lr': training_wrapper.LEARNING_RATE_SCALING, 'name': 'scales'},
            {'params': [self._rotations], 'lr': training_wrapper.LEARNING_RATE_ROTATION, 'name': 'rotations'}
        ]

        try:
            from Thirdparty.Apex import FusedAdam
            # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15, adam_w_mode=False)
        except Framework.ExtensionError:
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, fused=True)

        self.position_lr_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent,
            lr_final=training_wrapper.LEARNING_RATE_POSITION_FINAL * self.training_cameras_extent,
            lr_delay_mult=training_wrapper.LEARNING_RATE_POSITION_DELAY_MULT,
            max_steps=training_wrapper.LEARNING_RATE_POSITION_MAX_STEPS)

    def update_learning_rate(self, iteration: int) -> None:
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'positions':
                lr = self.position_lr_scheduler(iteration)
                param_group['lr'] = lr

    def reset_opacities(self) -> None:
        """Resets the opacities to a fixed value."""
        opacities_new = self.inverse_opacity_activation(self.get_opacities.clamp_max(0.01))
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune_points(self, prune_mask: torch.Tensor) -> None:
        """Prunes points that are not visible or too large."""
        valid_mask = ~prune_mask
        optimizable_tensors = prune_param_groups(self.optimizer, valid_mask)

        self._positions = optimizable_tensors['positions']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

        self.densification_gradient_accum = self.densification_gradient_accum[valid_mask]
        self.n_observations = self.n_observations[valid_mask]

    def densification_postfix(
        self,
        new_positions: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scales: torch.Tensor,
        new_rotations: torch.Tensor
    ) -> None:
        """Incorporate the changes from the densification step into the parameter groups."""
        d = {
            'positions': new_positions,
            'f_dc': new_features_dc,
            'f_rest': new_features_rest,
            'opacities': new_opacities,
            'scales': new_scales,
            'rotations': new_rotations
        }

        optimizable_tensors = extend_param_groups(self.optimizer, d)
        self._positions = optimizable_tensors['positions']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

        self.densification_gradient_accum = torch.zeros((self.get_positions.shape[0], 1), dtype=torch.float32, device='cuda')
        self.n_observations = torch.zeros((self.get_positions.shape[0], 1), dtype=torch.int32, device='cuda')

    def split(self, grads: torch.Tensor, grad_threshold: float) -> None:
        """Densify by splitting Gaussians that satisfy the gradient condition."""
        n_init_points = self.get_positions.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values > self.percent_dense * self.training_cameras_extent

        stds = self.get_scales[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask]).repeat(2, 1, 1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask].repeat(2, 1)
        new_scales = self.inverse_scaling_activation(self.get_scales[selected_pts_mask].repeat(2, 1) / 1.6)
        new_rotations = self._rotations[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacities[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_positions, new_features_dc, new_features_rest, new_opacities, new_scales, new_rotations)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum().item(), device='cuda', dtype=torch.bool)))
        self.prune_points(prune_filter)

    def duplicate(self, grads: torch.Tensor, grad_threshold: float) -> None:
        """Densify by duplicating Gaussians that satisfy the gradient condition."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values <= self.percent_dense * self.training_cameras_extent

        new_positions = self._positions[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacities[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]

        self.densification_postfix(new_positions, new_features_dc, new_features_rest, new_opacities, new_scales, new_rotations)

    def densify_and_prune(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool) -> None:
        """Densifies the point cloud and prunes points that are not visible or too large."""
        grads = self.densification_gradient_accum / self.n_observations.clamp_min(1)

        self.duplicate(grads, grad_threshold)
        self.split(grads, grad_threshold)

        prune_mask = self.get_opacities.flatten() < min_opacity
        if prune_large_gaussians:
            prune_mask |= self.get_scales.max(dim=1).values > 0.1 * self.training_cameras_extent
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor: torch.Tensor, visibility_mask: torch.Tensor) -> None:
        """Accumulates the gradient for densification."""
        self.densification_gradient_accum[visibility_mask] += torch.norm(viewspace_point_tensor.grad[visibility_mask, :2], dim=-1, keepdim=True)
        self.n_observations[visibility_mask] += 1

    def bake_activations(self) -> None:
        """Bakes relevant activation functions into the final parameters."""
        # bake activation functions into final parameters
        self._rotations.data = self.get_rotations
        self.rotation_activation = torch.nn.Identity()
        self._opacities.data = self.get_opacities
        self.opacity_activation = torch.nn.Identity()
        self.inverse_opacity_activation = torch.nn.Identity()
        self._scales.data = self.get_scales
        self.scaling_activation = torch.nn.Identity()
        self.inverse_scaling_activation = torch.nn.Identity()
        # prune points that would never be visible anyway
        self.prune_points(self._opacities.flatten() < 0.00392156862)  # 1/255
        # morton sort
        morton_encoding = morton_encode(self._positions)
        order = torch.argsort(morton_encoding)
        self._positions.data = self._positions[order].contiguous()
        self._rotations.data = self._rotations[order].contiguous()
        self._features_dc.data = self._features_dc[order].contiguous()
        self._features_rest.data = self._features_rest[order].contiguous()
        self._scales.data = self._scales[order].contiguous()
        self._opacities.data = self._opacities[order].contiguous()
        # bake covariances
        self._baked_covariances = torch.nn.Parameter(self.get_covariances(1.0), requires_grad=False)


@Framework.Configurable.configure(
    SH_DEGREE=3,
)
class GaussianSplattingModel(BaseModel):
    """Defines the 3DGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None

    def build(self) -> 'GaussianSplattingModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        return self
