"""
    Evolved Neural Network
"""

from sklearn import datasets
import numpy as np

from ..helpers.utils.data_utils import data_helper
from ..supervised.evolved_nn import EvolvedNN
from ..helpers.utils.display import plot_dimensioner


def start_evolved_nn():
    """
        Runs the evolved neural network model
    """

    data = datasets.load_digits()

    X = data_helper.normalize(data.data)
    y = data_helper.categorize(data.target.astype('int'))

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=.4, seed=1)

    model = EvolvedNN(X_train, y_train,
                      n_individuals=100)
    individual = model.evolve(n_generations=1000)

    _, acc = individual.test_on_batch(X_test, y_test)
    y_pred = np.argmax(individual.make_prediction(X_test), axis=1)

    plot_dimensioner.plot_in_two_d(X_test, y_pred,
                                   title='Evolved Neural Network',
                                   accuracy=acc,
                                   legend_labels=range(y.shape[1])
                                   )
