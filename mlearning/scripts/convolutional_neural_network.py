"""
    This module contains a Convolutional Neural Network example
"""
from sklearn import datasets

from ..helpers.deep_learning.network import Neural_Network
from ..deep_learning.grad_optimizers import Adam
from ..helpers.utils.data_utils import data_helper
from ..helpers.deep_learning.layers import ConvolutionTwoD, Activation, DropOut
from ..helpers.deep_learning.loss import CrossEntropyLoss


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
    y = digits.target

    # Convert to one hot encoding

    y = data_helper.categorize(y.asType('int'))

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=0.4, seed=1)

    # Reshape to no. of samples, channels, height, width
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    classifier = Neural_Network(optimizer, CrossEntropyLoss, (X_test, y_test))
    classifier.add_layer(ConvolutionTwoD(
        no_of_filters=16, filter_shape=(3, 3), input_shape=(1, 8, 8),
        padding=True))
    classifier.add_layer(Activation('ReLu'))
    classifier.add_layer(DropOut(0.25))
