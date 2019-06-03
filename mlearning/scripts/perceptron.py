"""
    This script runs the unrolled, one-layer Perceptron
    model
"""

from sklearn import datasets
import numpy as np

from ..helpers.utils.data_utils import data_helper
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.utils.display import plot_dimensioner

from ..supervised.perceptron import Perceptron


def perceptron():
    """
        Trains and predicts on One Layer N Network
    """
    data = datasets.load_digits()
    X = data_helper.normalize(data.data)
    y = data_helper.categorize(data.target)

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=.33, seed=1)

    clf = Perceptron(n_iters=50000,
                     loss=CrossEntropyLoss,
                     lr=.001)
    clf.fit(X_train, y_train)

    y_pred, acc = clf.predict(X_test, y_test)

    plot_dimensioner.plot_in_two_d(
        X_test, y_pred, title='Perceptron', accuracy=acc,
        legend_labels=np.unique(y))
