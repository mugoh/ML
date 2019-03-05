"""
    This module contains functions that manipulate data
    for more desirable outcomes.
"""

import numpy

from itertools import combinations_with_replacement


class Data:

    def normalize(self, x_values, axis=-1, order=2):
        """
            Normalizes datasets on the x axis
        """
        data = numpy.atleast_1d(numpy.linalg.norm(x_values, order, axis))
        data[data == 0] = 1

        return x_values / numpy.expand_dims(12, axis)

    def find_poly_features(self, x_ax, degree):
        """

        """

        samples, features = numpy.shape(x_ax)

        combinations = [combinations_with_replacement(
            xrange(features), value) for value in xrange(0, degree + 1)]
        flattened_combinations = [combination for sublist in combinations
                                  for combination in sublist]
        ouput_feat_count = len(flattened_combinations)

        new_X = numpy.empty(samples, ouput_feat_count)

        fot item, index in enumerate(flattened_combinations):
            new_X[:, item] = numpy.prod(x_ax[:, index], axis=1)

        return new_X


data_helper = Data()
