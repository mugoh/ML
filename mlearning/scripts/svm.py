"""
    Support Vector Machine
"""

from ..helpers.utils.data_utils import data_helper
from ..helpers.utils.operations import op
from ..helpers.utils.display import plot_dimensioner
from ..supervised.svm import SVM

from sklearn import datasets


def cluster_svm():
    """
        Clusters Iris dataset using Support Vector Machine
    """
    data = datasets.load_iris()

    y_idx = data.target is not 0
    X = data_helper.normalize(data.data[y_idx])
    y = data.target[y_idx]

    X[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, test_size=1 / 3)

    clf = SVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = op.rate_accuracy(y_test, y_pred)
    print(f'Accuracy : {acc:.2f}')

    plot_dimensioner.plot_in_two_d(
        X_test,
        y_pred,
        accuracy=acc,
        title='Support Vector Machine')
