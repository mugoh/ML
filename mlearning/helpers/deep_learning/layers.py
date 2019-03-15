"""
    This module contains the network model
    layers.
"""


class Layer:

    def set_input_shape(self, shape):
        """
            Sets the input shape expected of the layer
            for the forward pass
        """
        self.input_shape = shape


class ConvolutionTwoD:
    """
        A class that models a 2D data layer

        ______----_______-----_________

    """
