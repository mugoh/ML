from ..supervised.multilayer_perceptron import MultilayerPerceptron
from ..helpers.utils.data_utils import data_helper
from ..helpers.utils.display import plot_dimensioner as plotter

from sklearn import datasets
import numpy as np


def m_perceptron():
    """
        Creates a multi-layer perceptron classification
    """
    data = datasets.load_digits()
    X = data_helper.normalize(data.data)
    y = data_helper.categorize(data.target)

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=0.4, seed=3)

    clf = MultilayerPerceptron(n_nodes=64,
                               iters=10000, lr=.01)
    clf.fit(X_train, y_train)
    y_pred, acc = clf.predict(X_test, accuracy_required=True, y_test=y_test)

    plotter.plot_in_two_d(X_test, y_pred,
                          title='Multilayer perceptron',
                          accuracy=acc,
                          legend_labels=np.unique(y))
