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

    def plot_regression(self, lines, title, axis_labels=None,
                        mse=None, scatter=None,
                        legend={'type': 'lines', 'location': 'lower right'}):
        """
            Helps in plotting of a regression fit
        """
        if scatter:
            scatter_plots = []
            scatter_labels = []

            for item in scatter:
                scatter_plots += [plt.scatter(item['x'],
                                              item['y'], color=item['color'],
                                              s=item['size'])]
                scatter_labels += [item['label']]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)

        for line in lines:
            line = plt.plot(line['x'], line['y'],
                            color=line['color'],
                            linewidth=line['width'],
                            label=line['label'])
        plt.suptitle(title)
        if mse:
            plt.suptitle('Mean Sq. Error: {mse}', fontsize=9)
        if axis_labels:
            plt.xlabel(axis_labels['x'])
            plt.ylabel(axis_labels['y'])

        if legend['type'] == 'lines':
            plt.legend(loc='lower_left')
        elif scatter and legend['type'] == 'scatter':
            plt.legend(scatter_plots, scatter_labels, loc=legend['location'])

        plt.show()
