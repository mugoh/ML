"""
    K-Means clustering
"""

from ..unsupervised.k_means import KMeans
from ..helpers.utils.display import plot_dimensioner
from sklearn import datasets


def cluster():
    """
        Starts the KMeans clustering with an initialized dataset
    """

    X, y = datasets.make_blobs(n_samples=200, n_features=4)

    clf = KMeans(X, k=4, iterations=200)
    y_pred = clf.predict()

    plot_dimensioner.plot_in_two_d(X, y_pred,
                                   title='KMeans Clustering')
    plot_dimensioner.plot_in_two_d(X, y,
                                   title='Actual Clustering')
