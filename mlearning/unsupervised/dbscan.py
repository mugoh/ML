"""
    Density Based Clustering
"""


class DBScan:
    """
        A Density Based Clustering Model.
        It expands clusters from samples whose number
        of points within a given radius exceed the minimum sample value.

        Outliers are marked as outliers points lying in low density areas
    """

    def __init__(self, epsilon=1, min_samples=5):
        self.eps = epsilon
        self.min_samples = min_samples

        self.clusters = []
        self.visited_samples = []
        self.close_samples = {}
