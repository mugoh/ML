"""
    This moddule contains  activation functions that are
    used to give the output of the network from the biased inputs.
"""

import numpy as np


class Rectified_Linear_Units:
    """
            A rectified linear units activation function
    """

    def __call__(self, x):
        """
                Runs a ReLu function for given input and
                returs the ouput
        """
        return np.where(x >= 0, x, 0)

    def find_grad(self, x):
        """Calculates the gradient (ReLu : 0 < x 1)

        """
        return np.where(x >= 0, 1, 0)


class Sigmoid:
    """
        1/ (1 + exp(x))
        Exists between 0 and 1 therefore usefful in predicting
        probability
    """

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class SoftMax:
    """
        SoftMax activation function
        : exp(x)/ sum of exp(x)
    """

    def __call__(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def grad(self, x):
        prob = self.__call__(x)
        return prob * (1 - prob)


class TanH:
    """
        TanH activation function

    """
