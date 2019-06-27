"""
    This module contains functions that manipulate data
    for more desirable outcomes.
"""

import numpy

from itertools import combinations_with_replacement


class Data:
    """
        Manipulates data for favourable input structures.
    """

    def normalize(self, x_values, axis=-1, order=2):
        """
            Normalizes dataset
        """
        data = numpy.atleast_1d(numpy.linalg.norm(x_values, order, axis))
        data[data == 0] = 1

        return x_values / numpy.expand_dims(data, axis)

    def find_poly_features(self, x_ax, degree):
        """

        """

        samples, features = numpy.shape(x_ax)

        combinations = [combinations_with_replacement(
            range(features), value) for value in range(0, degree + 1)]
        flattened_combinations = [combination for sublist in combinations
                                  for combination in sublist]
        ouput_feat_count = len(flattened_combinations)

        new_X = numpy.empty((samples, ouput_feat_count))

        for item, index in enumerate(flattened_combinations):
            new_X[:, item] = numpy.prod(x_ax[:, index], axis=1)

        return new_X

    def find_mse(self, y_true, y_prediction):
        """
            Finds the mean square error between the true and
            the predicted value of y
        """
        return numpy.mean(
            numpy.power(y_true - y_prediction, 2))

    def fold_validation_set(self, X, Y, set_count, shuffle=True):
        """
            Splits data into a given set count of training or
            testing data.
        """
        no_of_samples = len(Y)
        remainder_data = {}
        data_sets = []
        no_of_untouched = no_of_samples % set_count

        if shuffle:
            X, Y = self.shuffle(X, Y)

        if no_of_untouched:

            remainder_data['X'] = X[-no_of_untouched]
            remainder_data['Y'] = Y[-no_of_untouched]

        x_split = numpy.split(X, set_count)
        y_split = numpy.split(Y, set_count)

        for i in range(set_count):
            X_test, Y_test = x_split[i], y_split[i]
            X_train = numpy.concatenate(x_split[:i] + x_split[i + 1:], axis=0)
            Y_train = numpy.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
            data_sets.append([X_train, X_test, Y_train, Y_test])

        # Add leftover sampes to last set as
        # training samples

        if no_of_untouched:
            numpy.append(data_sets[-1][0], remainder_data['X'], axis=0)
            numpy.append(data_sets[-1][2], remainder_data['Y'], axis=0)

        return numpy.array(data_sets)

    def shuffle(self, x_values, y_values, seed_value=None):
        """
            Returns a a tuple of X, Y values as a random
            shuffle of data samples.
        """

        if seed_value:
            numpy.random.seed(seed_value)
        index = numpy.arange(x_values.shape[0])
        numpy.random.shuffle(index)

        return x_values[index], y_values[index]

    def split_train_test(self, X, Y, test_size=0.5, shuffle=True, seed=None):
        """
            Splits data into train and testing values using
            the "test_size" ratio.
        """

        X, Y = self.shuffle(X, Y, seed)

        split_data = len(Y) - int(len(Y) // (1 / test_size))
        X_train, X_test = X[:split_data], X[split_data:]
        Y_train, Y_test = Y[:split_data], Y[split_data:]

        return X_train, X_test, Y_train, Y_test

    def iterate_over_batch(self, X, y=None, batch_size=64):
        """
            Creates a Generator for a batch of given size
        """
        no_of_samples = X.shape[0]

        for i in numpy.arange(0, no_of_samples, batch_size):
            start, finish = i, min(i + batch_size, no_of_samples)

            if y is not None:
                yield X[start:finish], y[start:finish]
            else:
                yield X[start:finish]

    def categorize(self, x_value, columns=None):
        """
            Performs one hot encoding for nominal values
        """
        if not columns:
            columns = numpy.amax(x_value) + 1
        res = numpy.zeros((x_value.shape[0], columns))
        res[numpy.arange(x_value.shape[0]), x_value] = 1

        return res


data_helper = Data()
