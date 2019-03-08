"""
    This module contains optimizers that utiilze gradient descents
    for finding weights at global loss minima.
"""

import numpy


class Adam:
    """
        Adaptive Moment Estimation
        Maintains adaptive learning rates for each parameter (first moment)
        and an exponentially decaying average of past gradients (momentum).
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = eps
        # Rates of decay
        self.beta1 = beta1
        self.beta2 = beta2

        # First and second moment estimates
        self.mean
        self.uncentered_variance

    def update(self, weight, grad_wrt_weight):
        """
            Updates parameters from bias-corrected first and second
            moment estimates.
        """

        if not self.mean or not self.uncentered_variance:
            self.mean = numpy.zeros(numpy.shape(grad_wrt_weight))
            self.uncentered_variance = numpy.zeros(
                numpy.shape(grad_wrt_weight))

        # Compute decaying averages of past and past squared gradients

        self.mean = self.beta1 * self.mean + (1 - self.beta1) * grad_wrt_weight
        self.uncentered_variance = self.beta2 * self.uncentered_variance + \
            (1 - self.beta2) * numpy.power(grad_wrt_weight, 2)

        bias_corrected_mean = self.mean / (1 - self.beta1)
        bias_corrected_variance = self.uncentered_variance / (1 - self.beta2)
        updated_weight = self.learning_rate * bias_corrected_mean / (
            numpy.sqrt(bias_corrected_variance) + self.epsilon)

        return weight - updated_weight
