"""
    This module contains various data operations common
    to data functions and statistical formalae
"""
import numpy as np


class Operations:
    """
        Holds methods that aid in performing of commonly
        recurring data operations.
    """

    def rate_accuracy(self, y, y_hat):
        """
            Conputes the accuracy by comapring the actual value of y
            to the predicted value
        """
        return np.sum(y == y_hat, axis=0) / len(y)

    def get_covariance_matrix(self, X, y=None):
        """
            Calculates the covariance of matrix
            of the given dataset.
        """

        if not y:
            y = X

        no_of_samples = np.shape(X)[0]
        cov_matrix = (1 / (no_of_samples - 1)) * \
            (X - X.mean(axis=0)).T.dot(y - y.mean(axis=0))
        return np.array(cov_matrix, dtype=float)

    def get_eucledian_distance(self, x_a, x_b):
        """
            CAlculates distance between two vectors
        """
        dist = 0

        for i in range(len(x_a)):  # Distance between each co-ordinnate
            dist += pow((x_a[i] - x_b[i]), 2)

        return pow(dist, .5)


op = Operations()
