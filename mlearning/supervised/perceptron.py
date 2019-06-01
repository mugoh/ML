"""
    Single Layer Neural Network
"""

from ..helpers.deep_learning.activation_functions import Sigmoid
from ..helpers.deep_learning.loss import MSE


class Perceptron:
    """
        One layer Neural Network classifier

        Parameters:
        ----------
        n_iters: int
            No. of traiining iterations to update weights
        activaton_fuc: obj
            Activation function to be used for each neuron
        loss: obj
            Loss function
        lr: float
            Step length used in tuning the weights
    """

    def __init__(self, n_iters=20000, activaton_fuc=Sigmoid, loss=MSE, lr=.01):
        pass
