"""
    Partitioning around Medoids
"""

from ..helpers.utils.operations import op

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
        self.init_medoids()
        self.create_clusters()

    def init_medoids(self):
        """
            Initializes the medoids as random samples
        """

        self.n_samples, self.n_features = self.X.shape

        self.medoids = np.zeros((self.k, self.n_features))

        for i in range(self.n_samples):
            self.medoids[i] = self.X[np.random.choice(range(self.n_samples))]

    def create_clusters(self):
        """
            Allocates samples to closest medoids
        """
        self.clusters = [[] for _ in range(self.k)]

        for sample_idx, sample in enumerate(self.X):
            self.clusters[
                self.find_closest_medoid(sample)].append(sample_idx)

    def find_closest_medoid(self, sample):
        """
            Finds the index of medoid closest to the sample
        """
        closest_dist = float('inf')

        for medoid in self.medoids:
            distance = op.get_eucledian_distance(medoid, sample)
            if distance < closest_dist:
                closest_dist = distance
                closest_medoid = medoid
        return self.medoids.index(closest_medoid)
