# -- coding: utf-8 --

"""InstantNGP/Model.py: InstantNGP Scene Model Implementation."""

import numpy as np
import torch
from kornia.utils.grid import create_meshgrid3d

import Framework
from Methods.Base.Model import BaseModel
from Methods.InstantNGP.utils import next_multiple
from Thirdparty import TinyCudaNN as tcnn


@Framework.Configurable.configure(
    SCALE=0.5,
    RESOLUTION=128,
    CENTER=[0.0, 0.0, 0.0],
    HASHMAP_NUM_LEVELS=16,
    HASHMAP_NUM_FEATURES_PER_LEVEL=2,
    HASHMAP_LOG2_SIZE=19,
    HASHMAP_BASE_RESOLUTION=16,
    HASHMAP_TARGET_RESOLUTION=2048,
    NUM_DENSITY_OUTPUT_FEATURES=16,
    NUM_DENSITY_NEURONS=64,
    NUM_DENSITY_LAYERS=1,
    DIR_SH_ENCODING_DEGREE=4,
    NUM_COLOR_NEURONS=64,
    NUM_COLOR_LAYERS=2,
)
class InstantNGPModel(BaseModel):
    """Defines InstantNGP data model"""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

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
        # createGrid
        self.center = torch.Tensor([self.CENTER])
        self.xyz_min = -torch.ones(1, 3) * self.SCALE
        self.xyz_max = torch.ones(1, 3) * self.SCALE
        self.xyz_size = self.xyz_max - self.xyz_min
        self.half_size = (self.xyz_max - self.xyz_min) / 2
        self.cascades = max(1 + int(np.ceil(np.log2(2 * self.SCALE))), 1)
        self.register_buffer('density_grid', torch.zeros(self.cascades, self.RESOLUTION ** 3))
        self.register_buffer('grid_coords', create_meshgrid3d(self.RESOLUTION, self.RESOLUTION, self.RESOLUTION, False, dtype=torch.int32, device=self.density_grid.device).reshape(-1, 3))
        self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.RESOLUTION ** 3 // 8, dtype=torch.uint8))
        # create encodings and networks
        self.encoding_xyz = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.NUM_DENSITY_OUTPUT_FEATURES,
            encoding_config={
                'otype': 'Grid',
                'type': 'Hash',
                'n_levels': self.HASHMAP_NUM_LEVELS,
                'n_features_per_level': self.HASHMAP_NUM_FEATURES_PER_LEVEL,
                'log2_hashmap_size': self.HASHMAP_LOG2_SIZE,
                'base_resolution': self.HASHMAP_BASE_RESOLUTION,
                'per_level_scale': np.exp(np.log(self.HASHMAP_TARGET_RESOLUTION * (2 * self.SCALE) / self.HASHMAP_BASE_RESOLUTION) / (self.HASHMAP_NUM_LEVELS - 1)),
                'interpolation': 'Linear'
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.NUM_DENSITY_NEURONS,
                'n_hidden_layers': self.NUM_DENSITY_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        # calculate number of parameters for the MLP part of the encoding for weight decay
        n_params_mlp = 0
        n_inputs = next_multiple(self.HASHMAP_NUM_FEATURES_PER_LEVEL * self.HASHMAP_NUM_LEVELS, 16)
        for i in range(self.NUM_DENSITY_LAYERS):
            n_params_layer = self.NUM_DENSITY_NEURONS * n_inputs
            n_params_layer = next_multiple(n_params_layer, 16)
            n_params_mlp += n_params_layer
            n_inputs = self.NUM_DENSITY_NEURONS
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
                        "otype": "Identity"
                    }
                ]

            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': self.NUM_COLOR_NEURONS,
                'n_hidden_layers': self.NUM_COLOR_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        self.n_mlp_params = len(self.color_mlp_with_encoding.params) + self.n_params_encoding_mlp
        return self
