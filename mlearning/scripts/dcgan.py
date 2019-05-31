"""
    Deep Convolutional Generative Adversarial Network
"""

from sklearn.datasets import fetch_mldata

from ..unsupervised.dcgan import DCGAN


def dcgan():
    """
        Creates a model of the DCGAN
    """
    mnist = fetch_mldata('MNIST original')

    X, y = mnist.data, mnist.target

    clf = DCGAN()
    clf.train(X, y, epochs=200000, batch_size=64, save_interval=50)
