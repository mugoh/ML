"""
    K Means Clustering
"""

import numpy as np

from ..helpers.utils.operations import op


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

        centroids = [self.X[np.random.choice(range(n_samples))] for
                     cluster in range(self.k)]
        self.centroids = np.array(centroids)

    def cluster(self):
        """
            Creates clusters by assigning samples to closest
            centroids
        """
        clusters = [[] for _ in range(self.k)]

        for sample_ind, sample in enumerate(self.X):
            cluster_idx = self.find_closest_centroid(sample)
            clusters[cluster_idx].append(sample_ind)

        return clusters

    def find_closest_centroid(self, sample):
        """
            Finds the centroid to which a sample is closest to
        """

        closest = float('inf')
        closest_idx = 0

        for idx, centroid in enumerate(self.centroids):
            dist = op.get_eucledian_distance(sample, centroid)

            if dist < closest:
                closest = dist
                closest_idx = idx
        return closest_idx

    def predict(self):
        """
            Performs K Means clustering and returns cluster indices
        """
        self.init_centroids()

        for _ in range(self.iterations):
            clusters = self.cluster()
            prev_centroids = self.centroids.copy()
            print()
            self.get_cluster_mean(clusters)

            if not np.all(self.centroids - prev_centroids):
                break  # Divergence
        return self.label_samples(clusters)

    def get_cluster_mean(self, clusters):
        """
            Creates new centroids as the means for samples in
            existing clusters
        """

        self.centroids = np.array([
            np.mean(self.X[cluster], axis=0) for cluster in clusters
        ])

    def label_samples(self, clusters):
        """
            Labels a sample with the index of the cluster
            to which it belongs

        """

        return np.array(
            [sample_idx for cluster in clusters for sample_idx in cluster]
        )
