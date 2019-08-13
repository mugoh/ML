from ..supervised.adaboost import AdaBoost
from ..helpers.utils.data_utils import data_helper
from ..helpers.utils.display import plot_dimensioner

import numpy as np
from sklearn import datasets


def adaboost():
    """
        Classifies dataset samples using Adaboost
    """
    data = datasets.load_digits()
    y = data.target
    digit_one, digit_two = 1, 8

    idx = np.append(
        np.where(y == digit_one)[0],
        np.where(y == digit_two)[0])

    y = data.target[idx]
    # Change labels [-1, 1]
    y[y == digit_one] = -1
    y[y == digit_two] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=.50)

    clf = AdaBoost(n_classifiers=5)
    clf.fit(X_train, y_train)
    y_pred, acc = clf.predict(X_test, y_test=y_test)
    print('Accuracy: ', acc)

    plot_dimensioner.plot_in_two_d(
        X_test, y_pred, title='AdaBoost Classifier',
        accuracy=acc)
