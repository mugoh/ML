"""
    This module contains various data operations common
    to data functions and statistical formalae
"""
import numpy as np


class Operations:
    """
        Holds methods that aid in performing of commonly
        recurring data operations.
    """

    def rate_accuracy(self, y, y_hat):
        """
            Conputes the accuracy by comapring the actual value of y
            to the predicted value
        """
        return np.sum(y == y_hat, axis=0) / len(y)


op = Operations()
