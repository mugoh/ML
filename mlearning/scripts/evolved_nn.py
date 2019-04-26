"""
    Evolved Neural Network
"""

from sklearn import datasets


def start_evolved_nn():
    """
        Runs the evolved neural network model
    """
    X, y = datasets.make_classification(n_samples=3000,
                                        n_classes=4,
                                        n_clusters_per_class=2,
                                        n_informative=2)
