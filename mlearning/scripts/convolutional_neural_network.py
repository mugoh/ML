"""
    This module contains a Convolutional Neural Network example
"""
from sklearn import datasets

from ..deep_learning.grad_optimizers import Adam
from ..helpers.data_utils import data_helper


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

    Y = to_categorcal(Y.asType('int'))

    X_train, X_test, Y_train, Y_test = data_helper.split_train_test

    # Reshape to no. of samples, channels, height, width
    X_test = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))
