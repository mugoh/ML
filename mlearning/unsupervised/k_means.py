"""
    K Means Clustering
"""

import numpy as np


class KMeans:
    """
        An iterative clustering that forms k clusters by assignment of samples
        to closest centroids.
        The centroids are then moved to the centre of the newly
        formed clusters

        Parameters:
        -----------
        X: array-like
            Dataset containing the samples
        k: int
            Number of clusters to form
        iterations: int
            Maximum number of iterations to run if convergence
            is unmet
    """

    def __init__(self, X, k=2, iterations=600):
        self.X = X
        self.k = k
        self.iterations = iterations

    def init_centroids(self):
        """
            Initializes k centroids as random samples drawn from
            the dataset
        """
        n_samples, n_features = self.X.shape

        centroids = [np.random.choice(range(n_samples)) for
                     cluster in self.k]
        return centroids
