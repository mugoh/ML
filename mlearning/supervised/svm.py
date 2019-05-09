"""
    Support Vector Machine
"""

import numpy as np
import cvxopt


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
        self.X, self.y = X, y

        self.gamma = n_features if not self.gamma else self.gamma

        self.start_optmization()

    def init_kernel(self, n_samples):
        """
          Initalizes the kernel function
        """
        self.kernel = self.kernel_f(power=self.power,
                                    gamma=self.gamma,
                                    bias=self.bias)
        kernel_matrx = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrx = self.kernel(self.X[i], self.X[j])

        return kernel_matrx

    def start_optmization(self):
        """
            Creates the quadratic optmization matrix problem
        """

        p = cvxopt.matrix(
            np.outer(
                self.y,
                self.y) *
            self.init_kernel(),
            tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

    def predict(self, X):
        """
            Iterates through samples, determing the labels
            of samples by the support vectors
        """

        y_pred = []

        for samples in X:
            pred = 0

            for mult in range(len(self.lagrng_mltpliers)):
                pred += self.lagrng_mltpliers[mult] * \
                    self.sv_labels[mult] * self.kernel(self.sv[mult], samples)

                pred += self.intercept
            y_pred.append(np.sign(pred))

        return np.array(y_pred)
