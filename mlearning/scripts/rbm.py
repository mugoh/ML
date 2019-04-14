"""
    Restricted Boltzman's Machine [Energy based model]
"""

from sklearn import fetch_mldata

import numpy as np


def start_restricted_bolz_machine():
    """
        Drives the restricted boltzmann machine network
    """

    mnist = fetch_mldata('MNIST original')
    X = mnist.data / 255
    y = mnist.target

    # Select samples of digit 2
    X = X[y == 2]

    # Limit dataset to 500 samples
    idx = np.random.choice(range(X.shape[0]), size=500, replace=False)
    X = X[idx]

    rbm = RBM()
