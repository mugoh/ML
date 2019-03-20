"""
    This module contains a Convolutional Neural Network example
"""
from sklearn import datasets

from ..helpers.deep_learning.network import Neural_Network
from ..deep_learning.grad_optimizers import Adam
from ..helpers.utils.data_utils import data_helper
from ..helpers.deep_learning.layers import ConvolutionTwoD


def convolute():
    """
        Runs convolution to a data input
    """
    #
    #
    #
    # Loss Function Implem'?
    # Replace ascii tables with astropy.io.ascii
    optimizer = Adam()

    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    # Convert to one hot encoding

    Y = data_helper.categorize(Y.asType('int'))

    X_train, X_test, y_train, y_test = data_helper.split_train_test

    # Reshape to no. of samples, channels, height, width
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    classifier = Neural_Network(optimizer, CrossEntropy, (X_test, y_test))
    classifier.add_layer(ConvolutionTwoD(
        no_of_filters=16, filter_shape=(3, 3), input_shape=(1, 8, 8),
        padding=True))
