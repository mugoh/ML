"""
    This module contains the network model
    layers.
"""
import numpy as np
import copy


class Layer:
    """
        Parent class layer model. Only contains methods common
        to impemented layer models and does not suffice
        to create a layer.
    """

    def set_input_shape(self, shape):
        """
            Sets the input shape expected of the layer
            for the forward pass
        """
        self.input_shape = shape

    def get_name(self):
        """
            Returns the name of the layer. Thiss is represented by
            the class instance holding the layer.
        """
        return self.__class__.__name__


class ConvolutionTwoD(Layer):
    """
        A class that models a 2D data layer
        Inherits Layer
        Parameters:
        ______----_______-----_________
        no_of_filters: int
            Number of filters to convolve over the input matrix. Represents
            the number of channels of the output shape.
        filter_shape: tuple
            Holds the filter height and width
                e.g (filter_height, filter_width)
        input_shape: tuple
            The expected shape of the input layer. Requires to be specified
            for the  first layer of the network.
                e.g (batch_size, channels, width, height)
        padding: boolean
            True - Ouput height and width matches input height and width
            False - Ouput without padding
        stride: int
            Step size of the filters during convolution over the input

    """

    def __init__(self, no_of_filters, filter_shape,
                 input_shape, padding=False, stride=1, trainable=True):
        self.no_of_filters = no_of_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.trainable = trainable

    def init_weights(self, optimizer):
        """
            Initializes the input weights
        """
        fltr_height, fltr_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / pow(np.prod(
            self.filter_shape),
            .5)
        self.weight_ = np.random.uniform(-limit,
                                         limit,
                                         size=(self.no_of_filters,
                                               channels,
                                               fltr_height,
                                               fltr_width)
                                         )
        self.weight_hat_ = np.zeros((self.no_of_filters, 1))
        self.weight_opt = copy.copy(optimizer)
        self.weight_hat_opt = copy.copy(optimizer)
