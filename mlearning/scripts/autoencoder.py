import numpy as np

from sklearn.datasets import fetch_mldata

from ..unsupervised.auto_encoder import AutoEncoder


def autoencoder():
    """
        Runs the autoencoder on dataset
    """
    mnist = fetch_mldata('MNIST original')
    X = mnist.data

    # Rescale [-1, 1]
    X = (X.astype(np.float32) - 127.5) / 127.5

    auto_encoder = AutoEncoder()
    auto_encoder.train(X, n_epochs=200000, batch_size=64, save_interval=100)
