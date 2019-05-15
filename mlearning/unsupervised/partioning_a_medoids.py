"""
    Partitioning around Medoids
"""

import numpy as np


class PartitionAMedoids:
    """
        A clustering method that forms k clusters.

        Samples are assigned to closest medoids which(medoids)
        get swapped with non-medoid samples if the total distance
        between the cluster members and their medoids is smaller
        than it was.

        Parameters
        -----------
        X: arraylike
            Dataset
        k: int
            Number of clusters to form
    """

    def __init__(self, X, k=2):
        self.k = k
        self.X = X

    def predict(self):
        """
            Partitions around the medoids.
            Returns the cluster labels
        """
        medoids = self.init_medoids()

    def init_medoids(self):
        """
            Initializes the medoids as random samples
        """

        self.n_samples, self.n_features = self.X.shape

        medoids = np.zeros((self.k, self.n_features))

        for i in range(self.n_samples):
            medoids[i] = self.X[np.random.choice(range(self.n_samples))]

        return medoids
