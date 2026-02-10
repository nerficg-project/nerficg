"""NeRF/Model.py: Implementation of the model for the original NeRF method."""

import torch

import Framework
from Methods.Base.Model import BaseModel
from Methods.NeRF.utils import get_activation_function, FrequencyEncoding


class NeRFBlock(torch.nn.Module):
    """Defines a NeRF block (input: position, direction -> output: density, color)."""

    def __init__(
        self,
        n_layers: int,
        n_color_layers: int,
        n_features: int,
        n_frequencies_position: int,
        n_frequencies_direction: int,
        encoding_append_input: bool,
        input_skips: list[int],
        activation_function: str,
    ) -> None:
        super().__init__()
        # set parameters
        self.input_skips = input_skips  # layer indices after which input is appended
        # get activation function (type, parameters, initial bias for last density layer)
        af_class, af_parameters, af_bias = get_activation_function(activation_function)
        # frequency encoding layers
        self.encoding_position = FrequencyEncoding(n_frequencies_position, encoding_append_input)
        self.encoding_direction = FrequencyEncoding(n_frequencies_direction, encoding_append_input)
        n_inputs_position = self.encoding_position.get_n_outputs(3)
        n_inputs_direction = self.encoding_direction.get_n_outputs(3)
        # initial linear layers
        self.initial_layers = [torch.nn.Sequential(
            torch.nn.Linear(n_inputs_position, n_features, bias=True), af_class(*af_parameters)
        )]
        for layer_index in range(1, n_layers):
            in_features = n_features if layer_index not in input_skips else n_features + n_inputs_position
            self.initial_layers.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, n_features, bias=True), af_class(*af_parameters)
            ))
        self.initial_layers = torch.nn.ModuleList(self.initial_layers)
        # intermediate feature and density layers
        self.feature_layer = torch.nn.Linear(n_features, n_features, bias=True)
        self.density_layer = torch.nn.Linear(n_features, 1, bias=True)
        self.density_activation = af_class(*af_parameters)
        # final color layer
        self.color_layers = torch.nn.Sequential(*(
            [torch.nn.Linear(n_features + n_inputs_direction, n_features // 2, bias=True), af_class(*af_parameters)]
            + [torch.nn.Sequential(torch.nn.Linear(n_features // 2, n_features // 2, bias=True), af_class(*af_parameters))
                for _ in range(n_color_layers - 1)]
            + [torch.nn.Linear(n_features // 2, 3, bias=True), torch.nn.Sigmoid()]
        ))
        # initialize bias for density layer activation function (for better convergence, copied from PyTorch3D examples)
        if af_bias is not None:
            self.density_layer.bias.data[0] = af_bias

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        random_noise_density: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # transform inputs to higher dimensional space
        positions_encoded = self.encoding_position(positions)
        # run initial layers
        x = positions_encoded
        for index, layer in enumerate(self.initial_layers):
            x = layer(x)
            if index + 1 in self.input_skips:
                x = torch.cat((x, positions_encoded), dim=-1)
        # extract density, add random noise before activation function
        density = self.density_layer(x)
        if random_noise_density > 0.0:
            density = density + random_noise_density * torch.randn_like(density)
        density = self.density_activation(density)
        # extract features, append view_directions, and extract color
        directions_encoded = self.encoding_direction(directions)
        features = self.feature_layer(x)
        features = torch.cat((features, directions_encoded), dim=-1)
        color = self.color_layers(features)
        return density, color


@Framework.Configurable.configure(
    HIERARCHICAL=True,
    N_LAYERS=8,
    N_COLOR_LAYERS=1,
    N_FEATURES=256,
    N_FREQUENCIES_POSITION=10,
    N_FREQUENCIES_DIRECTION=4,
    ENCODING_APPEND_INPUT=True,
    INPUT_SKIPS=[5], 
    NETWORK_ACTIVATION='relu',
)
class NeRF(BaseModel):
    """Defines a plain NeRF with a single MLP."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.coarse_nerf: NeRFBlock | None = None
        self.nerf: NeRFBlock | None = None

    def build(self) -> 'NeRF':
        """Builds the model."""
        if self.HIERARCHICAL:
            self.coarse_nerf = NeRFBlock(
                self.N_LAYERS,
                self.N_COLOR_LAYERS,
                self.N_FEATURES,
                self.N_FREQUENCIES_POSITION,
                self.N_FREQUENCIES_DIRECTION,
                self.ENCODING_APPEND_INPUT,
                self.INPUT_SKIPS,
                self.NETWORK_ACTIVATION,
            )
        self.nerf = NeRFBlock(
            self.N_LAYERS,
            self.N_COLOR_LAYERS,
            self.N_FEATURES,
            self.N_FREQUENCIES_POSITION,
            self.N_FREQUENCIES_DIRECTION,
            self.ENCODING_APPEND_INPUT,
            self.INPUT_SKIPS,
            self.NETWORK_ACTIVATION,
        )
        return self
