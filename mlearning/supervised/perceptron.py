"""
    This module holds a single-hidden-layer Neural Network
"""

from ..helpers.deep_learning.activation_functions import Sigmoid
from ..helpers.deep_learning.loss import MSE

from ..helpers.utils.display import progress_bar_widgets

import progressbar

import numpy as np


class Perceptron:
    """
        One layer Neural Network classifier

        Parameters:
        ----------
        n_iters: int
            No. of traiining iterations to update weights
        activaton_func: obj
            Activation function to be used for each neuron
        loss: obj
            Loss function
        lr: float
            Step length used in tuning the weights
    """

    def __init__(
            self,
            n_iters=20000,
            activaton_func=Sigmoid,
            loss=MSE,
            lr=.01):
        self.n_iters = n_iters
        self.activaton_func = activaton_func()
        self.loss = loss()
        self.lr = lr

        self.pgressbar = progressbar.Progressbar(widget=progress_bar_widgets)

    def fit(self, X, y):
        """
            Forward propagates and updates the model weights
        """
        n_samples, n_features = X.shape
        _, n_outputs, _ = y.shape

        # Init weights in range (-1/ sqrt(n), 1/sqrt(n))
        limit = 1 / np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit,
                                   (n_features, n_outputs))
        self.w_0 = np.zeros([1, n_outputs])

        for i in range(self.pgressbar):
            # Outputs
            linear_ouput = X.dot(self.w) + self.w_0
            y_pred = self.activaton_func(linear_ouput)

            err_gradient = self.loss.find_gradient(
                y, y_pred) * self.activaton_func.grad(linear_ouput)

            # Loss gradients
            grad_w = X.T.dot(err_gradient)
            grad_w_0 = np.sum(err_gradient, axis=0, keepdims=True)

            self.w -= self.lr * grad_w
            self.w_0 -= self.lr * grad_w_0

        def predict(self, X, y_test):
            """
                Predict labels using trained model
            """
            y_pred = self.activaton_func(X.dot(self.w) + self.w_0)
            acc = self.loss.get_acc_score(y_test, y_pred)
            return y_pred, acc
