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

    def find_mse(self, y_true, y_prediction):
        """
            Finds the mean square error between the true and
            the predicted value of Y
        """
        return numpy.mean(
            numpy.power(y_true - y_prediction, 2))

    def fold_validation_set(self, X, Y, set_count, shuffle=True):
        """
            Splits data into a given set count of training or
            testing data.
        """

        if shuffle:
            X, Y = self.shuffle(X, Y)

    def shuffle(self, x_values, y_values, seed_value=None):
        """
            Returns a random shuffle of data samples.
            Return value is a tuple of X, Y values
        """

        if seed_value:
            numpy.random.seed(seed_value)
        index = numpy.arange(x_values.shape[0])
        numpy.random.shuffle(index)

        return x_values[index], y_values[index]


data_helper = Data()
