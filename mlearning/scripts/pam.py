"""
    Partitioning Around Medoids
"""

from sklean import datasets

from ..unsupervised.partioning_a_medoids import PartitionAMedoids
from ..helpers.utils.display import plot_dimensioner


def cluster_pam():
    """
        Clusters samples by partitioning around medoids

    """
    X, y = datasets.make_blobs()

    clf = PartitionAMedoids(X, k=3)
    y_pred = clf.predict()

    plot_dimensioner.plot_in_two_d(X, y_pred, title='PAM Cluster')
    plot_dimensioner.plot_in_two_d(X, y, title='Actual Cluster')
