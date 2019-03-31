"""
    This moddule contains  activation functions that are
    used to give the output of the network from the biased inputs.
"""

import numpy as np


class Rectified_Linear_Units:
    """
        A rectified linear units activation function
        0 : x < 0
        1 : x >= 0
    """

    def __call__(self, x):
        """
                Runs a ReLu function for given input and
                returs the ouput
        """
        return np.where(x >= 0, x, 0)

    def grad(self, x):
        """
            Predicts the activation for f`(x)
           (ReLu : 0 < x 1)

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
        -1 to 1
        : (2 / [1 - exp(-2x)]) -1
    """

    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def grad(self, x):
        return 1 - np.power(self.__call__(x), 2)


class LeakyReLu:
    """
        Leaky Relu activation function
        - infinity to  infinity
        inf : x < 0
        1 : x >= 0
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def grad(self, x):
        return np.where(x >= 0, x, 1, self.alpha)


class ELU:
    """
        Exponential Linear Unit
        alpha(exp(x) - 1): x < 0
        x : x >= 0
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)


class SELU:
    """
        Scaled Exponential Linear Units
        alpha(exp(x) - 1): x < 0
        x : x >= 0
    """

    def __init__(self, alpha, scale):
        self.alpha = alpha or 1.6732632423543772848170429916717
        self.scale = scale or 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))


class SoftPlus:
    """
        SoftPlus activation function
        ln(1 + exp(x))
        log(exp)[1 + exp(x)]
    """

    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def grad(self, x):
        """
            f`(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))
