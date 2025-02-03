# -- coding: utf-8 --

"""HierarchicalNeRF/Model.py: Implementation of the neural model for the hierarchical NeRF method."""

from Methods.NeRF.Model import NeRF, NeRFBlock


class HierarchicalNeRF(NeRF):
    """Defines a hierarchical NeRF model containing a coarse and a fine version for efficient sampling."""

    def __init__(self, name: str = None) -> None:
        super(HierarchicalNeRF, self).__init__(name)

    def build(self) -> 'HierarchicalNeRF':
        """Builds the model."""
        self.coarse = NeRFBlock(
            self.NUM_LAYERS, self.NUM_COLOR_LAYERS, self.NUM_FEATURES,
            self.ENCODING_LENGTH_POSITIONS, self.ENCODING_LENGTH_DIRECTIONS, self.ENCODING_APPEND_INPUT,
            self.INPUT_SKIPS, self.ACTIVATION_FUNCTION
        )
        self.fine = NeRFBlock(
            self.NUM_LAYERS, self.NUM_COLOR_LAYERS, self.NUM_FEATURES,
            self.ENCODING_LENGTH_POSITIONS, self.ENCODING_LENGTH_DIRECTIONS, self.ENCODING_APPEND_INPUT,
            self.INPUT_SKIPS, self.ACTIVATION_FUNCTION
        )
        return self
