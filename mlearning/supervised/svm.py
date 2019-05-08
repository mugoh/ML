"""
    Support Vector Machine
"""

import numpy as np


class SVM:
    """
       Support Vector Machine Classifier

       Finds a hyperplane of N-dimension which gives a distict classification
       of the data points

       Parameters:
       -----------
        kernel: function
            The kernel function. A polynomial, Radial Basis Function, or Linear
        penalty: float
            The penalty term
        power: int
            The degree of the polynomial of the kernel
        gamma: float
            Radial Basis Function parameter
        cf: Bias term in the polynomial kernel function

    """

    def __init__(self, kernel_f=rbf, penalty=1, power=4, cf=4, gamma=None):
        self.kernel_f = kernel_f
        self.penalty = penalty
        self.gamma = gamma
        self.bias = cf
        self.power = power

    def fit(self, X, y):
        """
          Creates a hyperplane from the selected support vectors in the dataset
        """

        n_samples, n_features = np.shape(X)

        self.gamma = self.n_features if not gamma else gamma

        self.init_kernel()

    def init_kernel(self):
        """
          Initalizes the kernel function
        """
        self.kernel = self.kernel_f(power=self.power,
                                    gamma=self.gamma,
                                    bias=self.bias)
