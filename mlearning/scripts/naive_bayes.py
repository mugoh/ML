from ..supervised.naive_bayes import NaiveBayes
from ..helpers.utils.data_utils import data_helper
from ..helpers.utils.operations import op
from ..helpers.utils.display import plot_dimensioner

from sklearn import datasets


class classify_nv_bayes():
    """
        Runs Naive Bayes classification
    """

    data = datasets.load_digits()
    X = data_helper.normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = data_helper.split_train_test(X, y)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = op.rate_accuracy(y_test, y_pred)

    plot_dimensioner.plot_in_two_d(
        X_test, y_pred,
        title='Naive Bayes',
        accuracy=acc,
        legend_labels=data.target_names)
