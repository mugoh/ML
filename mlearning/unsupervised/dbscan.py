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
        unvisited_samples = [sample for sample in self.no_samples
                             if sample not in self.visited_samples]
        for sample in range(unvisited_samples):
            self.neighbouring_s[sample] = self.find_sample_neighbours(sample)

    def find_sample_neighbours(self, sample_index):
        """
            Gives indexes of samples within a radius of epsilon to
            the parameter sample value
        """

        indx = np.arange(len(self.X))
        neighbours = []

        for i, smpl in enumerate(
                self.X[indx is not sample_index]):
            dist = op.get_eucledian_distance(self.X[sample_index], smpl)
            neighbours.append(i) if dist < self.eps else None
        return np.array(neighbours)
