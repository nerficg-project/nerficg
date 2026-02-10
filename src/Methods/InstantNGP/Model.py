"""InstantNGP/Model.py: InstantNGP scene model implementation."""

import math

import torch

import Framework
from Logging import Logger
from Methods.Base.Model import BaseModel
from Methods.InstantNGP.utils import next_multiple
import Thirdparty.TinyCudaNN as tcnn


@Framework.Configurable.configure(
    SCALE=0.5,
    RESOLUTION=128,
    CENTER=[0.0, 0.0, 0.0],
    HASHGRID_N_LEVELS=16,
    HASHGRID_N_FEATURES_PER_LEVEL=2,
    HASHGRID_LOG2_SIZE=19,
    HASHGRID_BASE_RESOLUTION=16,
    HASHGRID_TARGET_RESOLUTION=2048,
    N_DENSITY_OUTPUT_FEATURES=16,
    N_DENSITY_NEURONS=64,
    N_DENSITY_LAYERS=1,
    DIR_SH_ENCODING_DEGREE=4,
    N_COLOR_NEURONS=64,
    N_COLOR_LAYERS=2,
    ENABLE_JIT_FUSION=True,
)
class InstantNGPModel(BaseModel):
    """Defines the InstantNGP data model."""

    def __del__(self) -> None:
        torch.cuda.empty_cache()
        tcnn.free_temporary_memory()

    def weight_decay_mlp(self) -> torch.Tensor:
        """Calculates the weight decay for the MLPs."""
        loss = self.encoding_xyz.params[:self.n_params_encoding_mlp].pow(2).sum()
        loss += self.color_mlp_with_encoding.params.pow(2).sum()
        # loss /= 2  # fused into loss lambda
        loss /= self.n_mlp_params
        return loss

    def build(self) -> 'InstantNGPModel':
        """Builds the model."""
        self.center = torch.tensor([self.CENTER])
        self.xyz_min = -torch.ones(1, 3) * self.SCALE
        self.xyz_max = torch.ones(1, 3) * self.SCALE
        self.xyz_size = self.xyz_max - self.xyz_min
        self.half_size = self.xyz_size / 2
        self.cascades = max(1 + int(math.ceil(math.log2(2 * self.SCALE))), 1)
        grid_coords_single = torch.arange(self.RESOLUTION, dtype=torch.int32)
        self.grid_coords = torch.stack(torch.meshgrid([grid_coords_single, grid_coords_single, grid_coords_single], indexing='xy'), dim=-1).reshape(-1, 3)
        self.register_buffer('occupancy_grid', torch.zeros(self.cascades, self.RESOLUTION ** 3))
        self.register_buffer('occupancy_bitfield', torch.zeros(self.cascades * self.RESOLUTION ** 3 // 8, dtype=torch.uint8))
        self.encoding_xyz = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.N_DENSITY_OUTPUT_FEATURES,
            encoding_config={
                'otype': 'Grid',
                'type': 'Hash',
                'n_levels': self.HASHGRID_N_LEVELS,
                'n_features_per_level': self.HASHGRID_N_FEATURES_PER_LEVEL,
                'log2_hashmap_size': self.HASHGRID_LOG2_SIZE,
                'base_resolution': self.HASHGRID_BASE_RESOLUTION,
                'per_level_scale': math.exp(math.log(self.HASHGRID_TARGET_RESOLUTION * (2 * self.SCALE) / self.HASHGRID_BASE_RESOLUTION) / (self.HASHGRID_N_LEVELS - 1)),
                'interpolation': 'Linear'
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.N_DENSITY_NEURONS,
                'n_hidden_layers': self.N_DENSITY_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        # calculate number of parameters for the MLP part of the encoding for weight decay
        n_params_mlp = 0
        n_inputs = next_multiple(self.HASHGRID_N_FEATURES_PER_LEVEL * self.HASHGRID_N_LEVELS, 16)
        for i in range(self.N_DENSITY_LAYERS):
            n_params_layer = self.N_DENSITY_NEURONS * n_inputs
            n_params_layer = next_multiple(n_params_layer, 16)
            n_params_mlp += n_params_layer
            n_inputs = self.N_DENSITY_NEURONS
        n_params_mlp += next_multiple(self.encoding_xyz.n_output_dims, 16) * n_inputs
        self.n_params_encoding_mlp = n_params_mlp
        self.color_mlp_with_encoding = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + self.encoding_xyz.n_output_dims,
            n_output_dims=3,
            encoding_config={
                'otype': 'Composite',
                'nested': [
                    {
                        'n_dims_to_encode': 3,
                        'otype': 'SphericalHarmonics',
                        'degree': self.DIR_SH_ENCODING_DEGREE,
                    },
                    {
                        'otype': 'Identity'
                    }
                ]
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': self.N_COLOR_NEURONS,
                'n_hidden_layers': self.N_COLOR_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        self.n_mlp_params = len(self.color_mlp_with_encoding.params) + self.n_params_encoding_mlp
        if self.ENABLE_JIT_FUSION:
            if tcnn.supports_jit_fusion():
                Logger.log_info('enabling tiny-cuda-nn JIT fusion')
                self.encoding_xyz.jit_fusion = True
                self.color_mlp_with_encoding.jit_fusion = True
            else:
                Logger.log_warning('requested tiny-cuda-nn JIT fusion but feature is unavailable')
        return self
