# -- coding: utf-8 --

"""NeRF/Model.py: Implementation of the neural model for the vanilla (i.e. original) NeRF method."""

import torch

import Framework
from Methods.Base.Model import BaseModel
from Methods.NeRF.utils import getActivationFunction, FrequencyEncoding


class NeRFBlock(torch.torch.nn.Module):
    """Defines a NeRF block (input: position, direction -> output: density, color)."""

    def __init__(self, num_layers: int, num_color_layers: int, num_features: int,
                 encoding_length_position: int, encoding_length_direction: int, encoding_append_input: bool,
                 input_skips: list[int], activation_function: str) -> None:
        super(NeRFBlock, self).__init__()
        # set parameters
        self.input_skips = input_skips  # layer indices after which input is appended
        # get activation function (type, parameters, initial bias for last density layer)
        af_class, af_parameters, af_bias = getActivationFunction(activation_function)
        # embedding layers
        self.embedding_position = FrequencyEncoding(encoding_length_position, encoding_append_input)
        self.embedding_direction = FrequencyEncoding(encoding_length_direction, encoding_append_input)
        input_size_position = self.embedding_position.getOutputSize(3)
        input_size_direction = self.embedding_direction.getOutputSize(3)
        # initial linear layers
        self.initial_layers = [
            torch.nn.Sequential(torch.nn.Linear(input_size_position, num_features, bias=True), af_class(*af_parameters))
        ]
        for layer_index in range(1, num_layers):
            self.initial_layers.append(torch.nn.Sequential(
                torch.nn.Linear(num_features if layer_index not in input_skips else num_features + input_size_position,
                          num_features, bias=True), af_class(*af_parameters)
            ))
        self.initial_layers = torch.nn.ModuleList(self.initial_layers)
        # intermediate feature and density layers
        self.feature_layer = torch.nn.Linear(num_features, num_features, bias=True)
        self.density_layer = torch.nn.Linear(num_features, 1, bias=True)
        self.density_activation = af_class(*af_parameters)
        # final color layer
        self.color_layers = torch.nn.Sequential(*(
            [torch.nn.Linear(num_features + input_size_direction, num_features // 2, bias=True), af_class(*af_parameters)]
            + [torch.nn.Sequential(torch.nn.Linear(num_features // 2, num_features // 2, bias=True), af_class(*af_parameters))
                for _ in range(num_color_layers - 1)]
            + [torch.nn.Linear(num_features // 2, 3, bias=True), torch.nn.Sigmoid()]
        ))
        # initialize bias for density layer activation function (for better convergence, copied from pytorch3d examples)
        if af_bias is not None:
            self.density_layer.bias.data[0] = af_bias

    def forward(self, positions: torch.Tensor, directions: torch.Tensor,
                return_rgb: bool = False, random_noise_densities: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        # transform inputs to higher dimensional space
        positions_embedded: torch.Tensor = self.embedding_position(positions)
        # run initial layers
        x = positions_embedded
        for index, layer in enumerate(self.initial_layers):
            x = layer(x)
            if index + 1 in self.input_skips:
                x = torch.cat((x, positions_embedded), dim=-1)
        # extract density, add random noise before activation function
        density: torch.Tensor = self.density_layer(x)
        density = self.density_activation(density + (torch.randn(density.shape) * random_noise_densities))
        # extract features, append view_directions and extract color
        color: torch.Tensor | None = None
        if return_rgb:
            directions_embedded: torch.Tensor = self.embedding_direction(directions)
            features: torch.Tensor = self.feature_layer(x)
            features = torch.cat((features, directions_embedded), dim=-1)
            color = self.color_layers(features)
        return density, color


@Framework.Configurable.configure(
    NUM_LAYERS=8, 
    NUM_COLOR_LAYERS=1, 
    NUM_FEATURES=256,
    ENCODING_LENGTH_POSITIONS=10, 
    ENCODING_LENGTH_DIRECTIONS=4, 
    ENCODING_APPEND_INPUT=True,
    INPUT_SKIPS=[5], 
    ACTIVATION_FUNCTION='relu'
)
class NeRF(BaseModel):
    """Defines a plain NeRF with a single MLP."""

    def __init__(self, name: str = None) -> None:
        super(NeRF, self).__init__(name)

    def build(self) -> 'NeRF':
        """Builds the model."""
        self.net = NeRFBlock(
            self.NUM_LAYERS, self.NUM_COLOR_LAYERS, self.NUM_FEATURES,
            self.ENCODING_LENGTH_POSITIONS, self.ENCODING_LENGTH_DIRECTIONS, self.ENCODING_APPEND_INPUT,
            self.INPUT_SKIPS, self.ACTIVATION_FUNCTION
        )
        return self
