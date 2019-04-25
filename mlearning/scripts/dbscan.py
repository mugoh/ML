"""
    This module holds a function to run the DBSCAN model
"""
from ..unsupervised.dbscan import DBScan
from ..helpers.utils.display import plot_dimensioner

from sklearn import datasets


def run_dbscan():
    """
        Runs the Density Based Clustering model
    """

    X, y = datasets.make_moons(n_samples=500, shuffle=False, noise=0.08)

    classifier = DBScan(X, epsilon=0.17)
    y_pred = classifier.predict()
    plot_dimensioner.plot_in_two_d(X,
                                   y_pred,
                                   title='Density Based Clustering')
    plot_dimensioner.plot_in_two_d(X, y, title=' Actual Clustering')
