"""
    This module contains an unrolled two-layer Neural Network
"""
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.deep_learning.activation_functions import Sigmoid, SoftMax
from ..helpers.utils.display import progress_bar_widgets

from dataclasses import dataclass, field
from typing import Any
import progressbar
import copy

import numpy as np


@dataclass
class MultilayerPerceptron:
    """
        A fully connected one-hidden layer Neural Network

        Parameters
        ----------
        n_nodes: int
            Number  of neurons in the hidden layer
        iters: int
            No. of training iterations for which to tune the
            model weights
        lr: float
            Step length used in weight update
    """
    n_nodes: int
    iters: int = 3000
    lr: float = 0.01
    loss: Any = field(default=CrossEntropyLoss())
    hidden_activation: Any = field(default=Sigmoid())
    output_activation: Any = field(default=SoftMax())

    def init_weights(self):
        """
            Initializes model weights
        """
        n_samples, n_features = self.X.shape
        _, n_outputs = self.y.shape

        # Hidden Layer
        limit = self._get_limit(n_features)
        self.weights = np.random.uniform(-limit,
                                         limit, (n_features, self.n_nodes))
        self.bias = np.zeros((1, self.n_nodes))

        # Output Layer
        limit = self._get_limit(self.n_nodes)
        self.v = np.random.uniform(-limit, limit, (self.n_nodes, n_outputs))
        self.v_b = np.zeros((1, n_outputs))

    def _get_limit(self, lm):
        """
            Finds limit used in weights
            Gives the inverse square root of the value
        """
        return 1 / pow(lm, .5)

    def fit(self, X, y):
        """
            Trains the model, updating the weights
        """
        self.X = X
        self.y = y
        pgrbar = progressbar.ProgressBar(widgets=progress_bar_widgets)

        self.init_weights()

        for i in pgrbar(range(self.iters)):
            y_pred = self.run_forward_pass()
            self.backward_pass(y_pred)

    def run_forward_pass(self, X=None):
        """
            Traverses the input, hidden layer to give the label
            prediction
        """
        X = copy.deepcopy(self.X) if not np.any(X) else X

        self.input_ = X.dot(self.weights) + self.bias
        self.hidden_output = self.hidden_activation(self.input_)

        self.output_layer_in = self.hidden_output.dot(self.v) + self.v_b
        y_pred = self.output_activation(self.output_layer_in)

        return y_pred

    def backward_pass(self, y_pred):
        """
            Tunes weights to minimize loss
        """

        # Gradient with respect to input of output layer
        grad_output_layer_in = self.loss.find_gradient(
            self.y, y_pred) * self.output_activation.grad(self.output_layer_in)
        grad_v = self.hidden_output.T.dot(grad_output_layer_in)
        grad_v_b = np.sum(grad_output_layer_in, axis=0, keepdims=True)

        # Grad with respect to input of hidden layer
        grad_hidden_in = grad_output_layer_in.dot(
            self.v.T) * self.hidden_activation.grad(self.input_)

        grad_w = self.X.T.dot(grad_hidden_in)
        grad_bias = np.sum(grad_hidden_in, axis=0, keepdims=True)

        self.update_weights(
            {'grad_w': grad_w,
             'grad_bias': grad_bias,
             'grad_v': grad_v,
             'grad_v_b': grad_v_b}
        )

    def update_weights(self, grads):
        """
            Tunes the weights by gradient descent for minimized loss
        """
        self.v -= self.lr * grads.get('grad_v')
        self.v_b -= self.lr * grads.get('grad_v_b')
        self.weights -= self.lr * grads.get('grad_w')
        self.bias -= self.lr * grads.get('grad_bias')

    def predict(self, X, accuracy_required=False, y_test=None):
        """
            Predicts dataset labels using the trained model
        """

        y_pred = self.run_forward_pass(X)
        acc = ''

        if accuracy_required:
            acc = self.loss.get_acc_score(y_test, y_pred)
        return np.argmax(y_pred, axis=1), acc
