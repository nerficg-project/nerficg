"""GaussianSplatting/Renderer.py: """

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger

from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.GaussianSplatting.Model import GaussianSplattingModel
from Methods.GaussianSplatting.utils import convert_sh_features
from Thirdparty.DiffGaussianRasterization import GaussianRasterizationSettings, GaussianRasterizer


@Framework.Configurable.configure(
    USE_FUSED_COVARIANCE_COMPUTATION=True,
    USE_FUSED_SH_CONVERSION=True,
    SCALE_MODIFIER=1.0,
    DISABLE_SH0=False,
    DISABLE_SH1=False,
    DISABLE_SH2=False,
    DISABLE_SH3=False,
    USE_BAKED_COVARIANCE=False,
)
class GaussianSplattingRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [GaussianSplattingModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('GaussianSplatting renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'GaussianSplatting renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self.cached_sh_features = None

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if not isinstance(view.camera, PerspectiveCamera):
            raise Framework.RendererError('GaussianSplatting renderer only supports perspective cameras')
        if view.camera.distortion is not None:
            Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
        if benchmark:
            return self.render_image_benchmark(view)
        elif self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View) -> dict[str, torch.Tensor]:
        """Renders an image for a given view for optimization."""
        if not isinstance(view.camera, PerspectiveCamera):
            raise Framework.RendererError('GaussianSplatting renderer only supports perspective cameras')
        if view.camera.distortion is not None:
            Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
        positions = self.model.gaussians.get_positions
        viewspace_points = torch.zeros_like(positions, requires_grad=True) + 0
        viewspace_points.retain_grad()
        w2c = view.w2c.T
        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=view.camera.height,
            image_width=view.camera.width,
            tanfovx=view.camera.width / view.camera.focal_x * 0.5,
            tanfovy=view.camera.height / view.camera.focal_y * 0.5,
            bg=view.camera.background_color,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=w2c @ view.camera.get_projection_matrix().T,
            sh_degree=self.model.gaussians.active_sh_degree,
            campos=view.position,
            prefiltered=False,
            debug=False
        ))
        image, radii = rasterizer(
            means3D=positions,
            means2D=viewspace_points,
            shs=self.model.gaussians.get_features,
            opacities=self.model.gaussians.get_opacities,
            scales=self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations)
        return {
            'rgb': image,
            'viewspace_points': viewspace_points,
            'visibility_mask': radii > 0
        }

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view during inference."""
        positions = self.model.gaussians.get_positions
        w2c = view.w2c.T
        camera_position = view.position
        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=view.camera.height,
            image_width=view.camera.width,
            tanfovx=view.camera.width / view.camera.focal_x * 0.5,
            tanfovy=view.camera.height / view.camera.focal_y * 0.5,
            bg=view.camera.background_color,
            scale_modifier=self.SCALE_MODIFIER,
            viewmatrix=w2c,
            projmatrix=w2c @ view.camera.get_projection_matrix().T,
            sh_degree=self.model.gaussians.active_sh_degree,
            campos=camera_position,
            prefiltered=False,
            debug=False
        ))

        # modify sh features for visualization
        sh_features = self.model.gaussians.get_features
        if self.DISABLE_SH0:
            sh_features[:, 0].zero_()
        if self.DISABLE_SH1:
            sh_features[:, 1:4].zero_()
        if self.DISABLE_SH2:
            sh_features[:, 4:9].zero_()
        if self.DISABLE_SH3:
            sh_features[:, 9:16].zero_()

        if self.USE_FUSED_SH_CONVERSION:
            rgbs = None
        else:
            rgbs = convert_sh_features(
                sh_features=sh_features.transpose(1, 2).view(-1, 3, (self.model.SH_DEGREE + 1) ** 2),
                view_directions=torch.nn.functional.normalize(positions - camera_position),
                degree=self.model.gaussians.active_sh_degree
            )
            sh_features = None

        compute_covariance = True
        if not self.model.training and self.USE_BAKED_COVARIANCE:
            compute_covariance = False
            covariances = self.model.gaussians.get_baked_covariances
            if covariances.shape[0] != positions.shape[0]:
                Logger.log_warning('Baked covariance requested but not available')
                covariances = None
                compute_covariance = True
            scales = None
            rotations = None
        if compute_covariance:
            scales = self.model.gaussians.get_scales if self.USE_FUSED_COVARIANCE_COMPUTATION else None
            rotations = self.model.gaussians.get_rotations if self.USE_FUSED_COVARIANCE_COMPUTATION else None
            covariances = None if self.USE_FUSED_COVARIANCE_COMPUTATION else self.model.gaussians.get_covariances(self.SCALE_MODIFIER)

        image, radii = rasterizer(
            means3D=positions,
            means2D=torch.empty_like(positions),
            shs=sh_features,
            colors_precomp=rgbs,
            opacities=self.model.gaussians.get_opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=covariances)
        image.clamp_(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    @torch.inference_mode()
    def render_image_benchmark(self, view: View) -> dict[str, torch.Tensor]:
        """Renders an image "as fast as possible"."""
        if self.cached_sh_features is None:  # this is a hack to avoid blowing up model size on disk
            self.cached_sh_features = self.model.gaussians.get_features
        w2c = view.w2c.T
        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=view.camera.height,
            image_width=view.camera.width,
            tanfovx=view.camera.width / view.camera.focal_x * 0.5,
            tanfovy=view.camera.height / view.camera.focal_y * 0.5,
            bg=view.camera.background_color,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=w2c @ view.camera.get_projection_matrix().T,
            sh_degree=self.model.gaussians.active_sh_degree,
            campos=view.position,
            prefiltered=False,
            debug=False
        ))
        positions = self.model.gaussians.get_positions
        image = rasterizer(
            means3D=positions,
            means2D=torch.empty_like(positions),
            shs=self.cached_sh_features,
            opacities=self.model.gaussians.get_opacities,
            cov3D_precomp=self.model.gaussians.get_baked_covariances)[0]
        return {'rgb': image.clamp_(0.0, 1.0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb'].clamp_(0.0, 1.0)}
