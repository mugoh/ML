"""
    This module contains functions that help in sending
    outputs on data configuration and model progress
    to the command line.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

import progressbar

progress_bar_widgets = [
    'Training: ', progressbar.Percentage(),
    ' ',
    progressbar.Bar(marker='~',
                    left='[',
                    right=']'
                    ),
    ' ',
    progressbar.ETA(), ' '
]


class Plot:
    """
        A data plot display class
    """

    def __init__(self):
        self.cmap = plt.get_cmap('viridis')

    def transform(self, X, dimension):
        """
            Transforms a given layer to s specified dimension
        """

        covariance = get_matrix_covariance(X)
        eigen_values, eigen_vectors = np.linalg.eig(covariance)

        # Sort eigen values ande eigen_vectors by largest eigen values
        index = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[index][:dimension]
        eigen_vectors = np.atleast_1d(eigen_vectors[:, index])[:, :dimension]

        # Project onto principal components
        transformed_X = X.dot(eigen_vectors)

        return transformed_X
