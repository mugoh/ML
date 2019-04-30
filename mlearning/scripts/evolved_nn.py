"""
    Evolved Neural Network
"""

from sklearn import datasets

from ..helpers.utils.data_utils import data_helper
from ..supervised.evolved_nn import EvolvedNN


def start_evolved_nn():
    """
        Runs the evolved neural network model
    """
    X, y = datasets.make_classification(n_samples=3000,
                                        n_classes=4,
                                        n_clusters_per_class=2
                                        )
    data = datasets.load_digits()
    X = data_helper.normalize(data.data)
    y = data_helper.categorize(data.target.astype('int'))

    neuroev_model = EvolvedNN(X, y,
                              n_individuals=100)
    neuroev_model.evolve(n_generations=1000)
