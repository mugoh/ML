"""
    Density Based Clustering
"""

from ..helpers.utils.operations import op

import numpy as np


class DBScan:
    """
        A Density Based Clustering Model.
        It expands clusters from samples whose number
        of points within a given radius exceed the minimum sample value.

        Outliers are marked as outliers points lying in low density areas
    """

    def __init__(self, X, epsilon=1, min_samples=5):
        self.eps = epsilon
        self.min_samples = min_samples
        self.X = X
        self.no_samples = np.shape(X)[0]

        self.clusters = []
        self.visited_samples = []
        self.neighbouring_s = {}

    def predict(self):
        """
            Iterates through samples, creating new clusters
            to add to the present ones.
        """
        unvisited_samples = [sample for sample in range(self.no_samples)
                             if sample not in self.visited_samples]
        for sample in unvisited_samples:
            self.neighbouring_s[sample] = self.find_sample_neighbours(sample)

            if len(self.neighbouring_s[sample]) >= self.min_samples:
                # neighbour is a core point
                # expand cluster from neighbour
                self.visited_samples.append(sample)
                self.clusters.append(
                    self.expand_cluster(sample,
                                        self.neighbouring_s[sample])
                )
        return self.get_cluster_index()

    def find_sample_neighbours(self, sample_index):
        """
            Gives indexes of samples within a radius of epsilon to
            the parameter sample value
        """

        indx = np.arange(len(self.X))
        neighbours = []

        for i, smpl in enumerate(
                self.X[indx != sample_index]):
            dist = op.get_eucledian_distance(self.X[sample_index], smpl)
            neighbours.append(i) if dist < self.eps else None
        return np.array(neighbours)

    def expand_cluster(self, smpl_indx, neighbours):
        """
            Expands the cluster of a dense area until
            the border is arrived at
        """
        cluster = [smpl_indx]

        for ngbr_index in neighbours:
            if ngbr_index in self.visited_samples:
                continue
            else:
                self.visited_samples.append(ngbr_index)

            self.neighbouring_s[ngbr_index] = self.find_sample_neighbours(
                ngbr_index)

            if len(self.neighbouring_s[ngbr_index]) >= self.min_samples:
                cluster += self.expand_cluster(
                    ngbr_index,
                    self.neighbouring_s[ngbr_index]
                )
            else:
                cluster.append(ngbr_index)

        return cluster

    def get_cluster_index(self):
        """
            Gives the index labels in which samples are contained

            Outliers have a default index value equal to the number
            of samples
        """
        indices = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))

        for indx, cluster_ in enumerate(self.clusters):
            for sample_indx in cluster_:
                indices[sample_indx] = indx
        return indices
