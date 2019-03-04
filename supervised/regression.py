"""
    This module contains the regression models
"""
import numpy


class Regression:
    """
    Models the relationship between a scalar independent and a dependent variable
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

        for i in xrange(self.iters):
            y_prediction = x_value.dot(self.weights)

            ms_error = numpy.mean(0.5 * (y_value - y_prediction) ** 2 +
                                  self.regularization(self.weights))
            self.training_errs.append(ms_error)
