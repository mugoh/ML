"""
    This module contains deep learning loss functions
"""

import numpy as np

from ..utils.operations import op


class CrossEntropyLoss:
    """
        Perform operations that aid in computing the
        cross entropy loss between predicted and true data values.
    """

    def compute_loss(self, y_true, y_prediction):
        """
            Computes the cross entropy loss between the true
            and predicted values
        """
        pred = np.clip(y_prediction, 1e-15, 1 - 1e-15)

        return -y_true * np.log(pred) - (1 - y_true) * np.log(1 - pred)

    def get_acc_score(self, y_true, y_pred):
        """
            Finds the accuracy score of the actual values of
            y to the predicted values.
        """

        return op.rate_accuracy(np.argmax(y_true, axis=1),
                                np.argmax(y_pred, axis=1))

    def find_gradient(self, y_true, y_pred):
        """
            Finds the gradient between p, and q values H(p, q)
        """
        y_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_clipped) + (1 - y_true) / (1 - y_clipped)


class MSE:
    """
        Computes loss as a sum of squared distances between the target variable
        and the predicted values
    """

    def compute_loss(self, y, y_hat):
        """
            Gives the Mean Square Error from the loss function
        """
        return 0.5 * np.power((y - y_hat), 2)

    def find_gradient(self, y, y_hat):
        """
            Gives the loss computed from the function's
            derivative
        """
        return -(y - y_hat)

    def get_acc_score(self, y, y_hat):
        return 0
