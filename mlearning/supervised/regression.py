"""
    This module contains the regression models
"""
import numpy

from .helpers.data_utils import data_helper


class Regression:
    """
    Models the relationship between a scalar independent and
    a dependent variable
    """

    def __init__(self, no_of_iters, step_rate):
        """
        no_of_iters: Training iterations to run on the weight
        step_rate: The length of the step to be used in updating the weights

        """
        self.iters = no_of_iters
        self.learning_factor = step_rate
        self.training_errs = []

    def set_up_weights(self, features_count):
        """
            Sets up the weights to utilize. This is done
            in a random manner
        """

        threshold = 1 / pow(features_count, 0.5)
        self.weights = numpy.random.uniform(-threshold,
                                            threshold, (features_count))

    def fit_constants(self, x_value, y_value):
        """
            Insert constant 1 values for the bias weights
        """
        ndarrray = numpy.insert(x_value, 0, 1, axis=1)
        self.set_up_weights(ndarrray.shape[1])

        # Descent of gradient for number of iters

        for i in range(self.iters):
            y_prediction = ndarrray.dot(self.weights)

            ##
            # subclass regularization method

            ms_error = numpy.mean(0.5 * (y_value - y_prediction) ** 2 +
                                  self.regularization(self.weights))
            self.training_errs.append(ms_error)

            # Gradient loss of 12 (weight)
            gradient_weight = -(y_value - y_prediction).dot(ndarrray) + \
                self.regularization.grad(self.weights)
            self.weights -= self.learning_factor * gradient_weight

    def regularization(self, factor):
        """
            Implement in subclasses
        """
        return 0

    def make_prediction(self, x_value):
        """
        Inserts constant ones for bias weights
        """
        x = numpy.insert(x_value, 0, 1, axis=1)

        return x.dot(self.weights)


class PolynomialRRegression(Regression):
    """
        Balances the model fit with respect to the training data and
        complexity of the model. Transforms data to allow for polynomial
        regression.
    """

    def __init__(self, degree,
                 reg_factor, iters=3000,
                 learning_factor=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = RegularizedRidge(reg_factor)
        super(PolynomialRRegression, self).__init__(iters, learning_factor)

    def fit_constants(self, X, Y):
        """
            Normalizes data before fitting constant values
            for weights.
        """
        x = data_helper.normalize(
            data_helper.find_poly_features(X, degree=self.degree))
        super().fit_constants(x, Y)

    def make_prediction(self, x):
        """
            Normalizes data and inserts constants [1]
            for bias weights.
        """
        normalized_values = data_helper.normalize(
            data_helper.find_poly_features(x, self.degree))

        return super().make_prediction(normalized_values)


class ClassName(object):
    """docstring for ClassName"""

    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


class RegularizedRidge:
    """
        Performs regularization for Ridge Regression.
    """

    def __init__(self, reg_constant):
        self.factor = reg_constant

    def __call__(self, weight):
        return self.factor * 0.5 * weight.T.dot(weight)

    def grad(self, weight):
        return self.factor * weight
