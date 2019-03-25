"""
    This module contains the network model
    layers.
"""
from .activation_functions import (Rectified_Linear_Units, SoftMax, ELU, TanH,
                                   LeakyReLu, SELU, Sigmoid)

import numpy as np
import copy
import math


class Layer:
    """
        Parent class layer model. Only contains methods common
        to implemented layer models and does not suffice
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
            Returns the name of the layer. This is represented by
            the class instance holding the layer.
        """
        return self.__class__.__name__

    @classmethod
    def reshape_col_to_image(cls,
                             cols, imgs_shape, fltr_shape,
                             stride, output_shape=True):
        """
            Changes shape of the input layer from column to image

        """
        batch_size, channels, height, width = imgs_shape
        pad_h, pad_w = cls.get_padding(fltr_shape, output_shape)

        padded_h = height + np.sum(pad_h)
        padded_w = width + np.sum(pad_w)
        padded_imgs = np.empty((batch_size, channels, padded_h, padded_w))

        # Calculate indices for dot product between weights
        # and images

        ind1, ind2, ind3 = cls.get_img_cols_indices(
            imgs_shape, fltr_shape, (pad_h, pad_w), stride)

        cols_reshaped = cols.reshape(channels * np.product(fltr_shape))
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)

        # Add column content to images at the indices
        np.add.at(padded_imgs, (slice(None), ind1, ind2, ind3), cols_reshaped)

        # Image without padding
        return padded_imgs[:, :, pad_h[0]:height +
                           pad_h[0], pad_w[0]:width + pad_w[0]]

    @classmethod
    def get_padding(cls, fltr_shape, output_shape):
        """
            Determines the padding of the output height and width
        """
        if not output_shape:
            return (0, 0), (0, 0)
        flt_height, flt_width = fltr_shape

        # Ref
        # output_height = (height + padd_h - filter_height) / stride + 1
        # output_height = height # Stride = 1

        padd_ht_a = int(math.floor((flt_height - 1) / 2))
        padd_ht_b = int(math.ceil((flt_height - 1) / 2))
        padd_wt_a = int(math.floor((flt_width - 1) / 2))
        padd_wt_b = int(math.ceil((flt_width - 1) / 2))

        return (padd_ht_a, padd_ht_b), (padd_wt_a, padd_wt_b)

    @classmethod
    def reshape_image_to_col(cls, images, fltr_shape, stride,
                             output_shape=True):
        """
            Changes shape of the input layer from image to column
        """
        fltr_height, fltr_width = fltr_shape
        pad_h, pad_w = cls.get_padding(fltr_shape, output_shape)

        padded_imgs = np.pad(
            images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

        # Indices to apply dot product between weights and image
        ind_a, ind_b, ind_c = cls.et_img_cols_indices(
            images.shape, fltr_shape, (pad_h, pad_w), stride)

        # Retrieve image content at these images
        cols = padded_imgs[:, ind_a, ind_b, ind_c]
        channels = images.shape[1]

        # Reshape content to column shape
        cols_reshaped = cols.transpose(1, 2, 0).reshape(
            fltr_height * fltr_width * channels, -1)

        return cols_reshaped

    @classmethod
    def get_img_cols_indices(cls, img_shape, fltr_shape, padding, stride=1):
        """
            Calculates indices for dot product between weights
            and images
        """

        # Find expected output size
        batch_size, channels, height, width = img_shape
        fltr_height, fltr_width = fltr_shape
        pad_h, pad_w = padding
        out_height = int((height * np.sum(pad_h) - fltr_height) / stride + 1)
        out_width = int((width * np.sum(pad_w) - fltr_width) / stride + 1)

        ind_i0 = np.repeat(np.arange(fltr_height), fltr_width)
        ind_i0 = np.tile(ind_i0, channels)
        ind_i1 = stride * np.repeat(np.arange(out_height), out_width)

        ind_j0 = np.tile(np.arange(fltr_width), fltr_height * channels)
        ind_j1 = stride * np.tile(np.arange(out_width), out_height)
        ind_i = ind_i0.reshape(-1, 1) + ind_i1.reshape(1, -1)
        ind_j = ind_j0.reshape(-1, 1) + ind_j1.reshape(1, -1)

        ind_k = np.repeat(np.arange(channels), fltr_height *
                          fltr_width).reshape(-1, 1)

        return ind_k, ind_i, ind_j


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
            True - output height and width matches input height and width
            False - output without padding
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
        self.weight_out = np.zeros((self.no_of_filters, 1))
        self.optimized_w = copy.copy(optimizer)
        self.optimized_w_out = copy.copy(optimizer)

    def forward_pass(self, X, training=True):
        """
            Propagates input data through the network to
            get an output prediction
        """
        self.input_layer = X
        batch_size, channels, heigt, width = X.shape

        # For dot product between input and weights,
        # change image shape to column shape
        self.X_col = self.reshape_image_to_col(
            X, self.filter_shape,
            stride=self.stride,
            output_shape=self.padding)

        # Reshape weight shape to column
        self.W_col = self.weight_.reshape((self.no_of_filters, -1))
        output = self.W_col.dot(self.X_col) + self.weight_out

        # Reshape output to: no_of_filters, height, width, and batch size
        output = output.reshape(self.output_shape() + (batch_size, ))

        # Redistribute axes to bring batch size first
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, grad):
        """
            Parameter:
                grad: accumulated gradient
            ____________________________________
            Propagates the accumulated gradient backwards.
            -> It calculates the gradient of a loss function with respect
               to all the weights in the network.

        """
        # Reshape accumulated grad to column shape
        accumulated_grad = grad.transpose(
            1, 2, 3, 0).reshape(self.no_of_filters, -1)

        if self.trainable:
            # Find the dot product of the column-shaped accumulated grad
            # and column shape layer.
            # This will determine the grad at the layer w.r.t layer weights

            grad_weight = accumulated_grad.dot(
                self.X_col.T).reshape(self.weight_.shape)

            # Gradient w.r.t the bias sums
            grad_w_out = np.sum(grad, axis=1, keepdims=True)

            # Update layer weights
            self.weight_ = self.optimized_w.update(self.weight_, grad_weight)
            self.weight_out = self.optimized_w_out.update(
                self.weight_out, grad_w_out)

        # Find gradient to propagate back to previous layer
        accumulated_grad = self.W_col.T.dot(accumulated_grad)

        accumulated_grad = self.reshape_col_to_image(accumulated_grad,
                                                     self.input_layer.shape,
                                                     self.filter_shape,
                                                     stride=self.stride,
                                                     output_shape=self.padding)
        return accumulated_grad

    def output_shape(self):
        """
            Gives the shape of the output returned by the forward pass
        """
        channels, height, width = self.input_shape
        padd_ht, padd_wt = self.get_padding(
            self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(padd_ht) -
                         self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(padd_wt) -
                        self.filter_shape[1]) / self.stride + 1
        return self.no_of_filters, int(output_height), int(output_width)

    def paramitize(self):
        """
            Returns the number of trainable parameters used by the layer
        """
        return np.prod(self.weight_.shape)
        + np.prod(self.weight_out.shape)


class Activation(Layer):
    """
        Applies an activation operation to the input

        Parameters:
        ____________________
        name: string
            Activation function to be used
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions.get(name)()
        self.trainable = True

    def show_layer_name(self):
        """
            Gives a string representation of the activation function name
        """
        return f'Activation {self.activation_func.__class__.__name__}'

    def forward_pass(self, X, training=True):
        """
            Propagates input data through the network to
            get an output prediction
        """
        self.input_layer = X

        return self.activation_func(X)


class DropOut(Layer):
    """
        Randomly sets a fraction p of the inputs (probaility 1-p)
        of the previous layer to zero

        Parameters:
        -----------
        p: float
            The probability that  the given unit is set to zero
    """

    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.no_of_units = None
        self.input_shape = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training=True):
        """
           Propagates input data through the network to
           get an output prediction
       """
        c = (1 - self.p)

        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, accumulated_grad):
        """
             Propagates the accumulated gradient backwards
        """
        return accumulated_grad * self._mask

    def output_shape(self):
        """
            Gives the shape of the output returned by the forward pass
        """
        return self.input_shape


class BatchNormalization(Layer):
    """
        Batch normalization model
         -> Adds a Normalizatiion 'layer' Between each layer
            to reduce covariance shift
    """

    def __init__(self, momemtum=0.99):
        self.momemtum = momemtum
        self.trainable = True
        self.eps = 0.01
        self.running_var = None
        self.running_mean = None

    def initialize(self, optimizer):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)

        # Parameter optimizers
        self.gamma_opt = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def paramitize(self):
        """
            Returns the number of trainable parameters used by the layer
        """
        return np.product(self.gamma.shape) +
        np.product(self.beta.shape)

    def output_shape(self):
        """
            Gives the shape of the output returned
            by the forward pass
        """
        return self.input_shape

    def forward_pass(self, X, training=True):
        """
           Propagates input data through the network to
           get an output prediction
       """
        if not self.running_mean:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if training and self.trainable:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momemtum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Stats saved for backward pass
        self.X_centred = X * mean
        self.inv_std_dev = 1 / np.sqrt(var * self.eps)

        X_normalized = self.X_centred * self.inv_std_dev

        return self.gamma * X_normalized + self.beta

    def backward_pass(self, accumulated_grad):
        """
            Propagates the accumulated gradient backwards
       """

        # Stat used during forward pass
        gamma = self.gamma

        if self.trainable:
            X_normalized = self.X_centred * self.inv_std_dev
            grad_gamma = np.sum(accumulated_grad, axis=0)
            grad_beta = self.beta_opt.update(self.beta_opt, grad_beta)

        batch_size = accumulated_grad.shape[0]

        # loss gradient with respect to layer inputs
        # (Use stats from forward pass)
        accumulated_grad = (1 / batch_size) * gamma * self.inv_std_dev * (
            batch_size * accumulated_grad - np.sum(
                accumulated_grad, axis=0) -
            self.X_centred * self.inv_std_dev ** 2 *
            np.sum(
                accumulated_grad * self.X_centred, axis=0)
        )

        return accumulated_grad


class Flatten(Layer):
    """
        Converts a multi-dimensional to a two-d matrix
    """

    def __init__(self, input_shape=None):
        self.previous_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def output_shape(self):
        """
           Gives the shape of the output returned
           by the forward pass
       """
        return np.prod(self.input_shape),

    def forward_pass(self, X, training=True):
        """
            Propagates forward
        """
        self.previous_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accumulated_grad):
        """
            Propagates backward across neuron layers
        """
        return accumulated_grad.reshape(self.previous_shape)


activation_functions = {
    'ReLu': ReLu,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'softplus': SoftPlus,
    'leaky_relu': LeakyReLu,
    'elu': ELU,
    'tanh': TanH,
    'softmax': SoftMax


}
