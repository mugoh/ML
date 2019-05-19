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
        cost = self.calculate_cost()

        while True:
            best_medoids, lowest_cost = self.medoids, cost

            for medoid in self.medoids:
                for unvisited_smp in self.get_non_medoids():
                    medoids_ = self.medoids.copy()
                    medoids_[self.medoids == medoid] = unvisited_smp

                    self.medoids = medoids_.copy()
                    self.create_clusters()

                    updated_cost = self.calculate_cost()

                    if updated_cost < lowest_cost:
                        lowest_cost = updated_cost
                        best_medoids = medoids_
            if lowest_cost < cost:
                cost = lowest_cost
                self.medoids = best_medoids
            else:
                break
        self.create_clusters()

        return self.getcluster_labels()

    def init_medoids(self):
        """
            Initializes the medoids as random samples
        """

        self.n_samples, self.n_features = self.X.shape

        self.medoids = np.zeros((self.k, self.n_features))

        for i in range(self.k):
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

        for i, medoid in enumerate(self.medoids):
            distance = op.get_eucledian_distance(medoid, sample)
            if distance < closest_dist:
                closest_dist = distance
                closest_medoid_i = i
        return closest_medoid_i

    def calculate_cost(self):
        """
            Gives distance between each sample and its medoid
        """

        cost = 0

        for i, cluster in enumerate(self.clusters):
            medoid = self.medoids[i]

            for sample_idx in cluster:
                cost += op.get_eucledian_distance(self.X[sample_idx], medoid)

        return cost

    def get_non_medoids(self):
        """
            Returns all samples that are currenly not medoids
        """
        non_m = [sample for sample in self.X if sample not in self.medoids]

        return non_m

    def get_cluster_labels(self):
        """
            Labels samples from the indices of their cluster
        """
        y_pred = np.zeros(self.n_samples)

        for clst_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = clst_idx

        return y_pred
