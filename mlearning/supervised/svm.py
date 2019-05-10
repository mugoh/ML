"""
    Support Vector Machine
"""

import numpy as np
import cvxopt


class SVM:
    """
        Support Vector Machine Classifier

        Finds a hyperplane of N-dimension which gives a distinct classification
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

        self.n_samples, self.n_features = np.shape(X)
        self.X, self.y = X, y

        self.gamma = self.n_features if not self.gamma else self.gamma

        minimization = self.minimize_qfunction().get('x')

        lagrng_mltpliers = np.ravel(minimization)

        # Get non-zeros -> Extract support vectors
        idx = lagrng_mltpliers > 1e-8
        self.lagrng_mltpliers = lagrng_mltpliers[idx]
        self.s_vectors = X[idx]
        self.sv_labels = y[idx]

    def init_kernel(self):
        """
          Initializes the kernel function
        """

        n_samples = self.n_samples

        self.kernel = self.kernel_f(power=self.power,
                                    gamma=self.gamma,
                                    bias=self.bias)
        kernel_matrx = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrx = self.kernel(self.X[i], self.X[j])

        return kernel_matrx

    def find_intercepts(self):
        """
            Calculates intercepts from present support vectors
        """
        self.intercept = self.sv_labels[0]

        self.intercept -= [
            multpr * self.s_vectors[i] *
            self.kernel(s_vectors[i], s_vectors[0]
                        for i, multpr in enumerate(self.lagrng_mltpliers))
        ]

    def minimize_qfunction(self):
        """
            Defines the quadratic optimization problem.
            It returns the minimized CVXOPT function solution
        """

        Q = cvxopt.matrix(
            np.outer(
                self.y, self.y) * self.init_kernel(),
            tc='d')
        p = cvxopt.matrix(np.ones(self.n_samples) * -1)
        A = cvxopt.matrix(y, (1, self.n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G_max = self.__get_stack_rows(atype='identity', -1)

        h_max = self.__get_stack_rows()

        if not self.C:
            G, h = cvxopt.matrix(G_max), cvxopt.matrix(h_max)
        elif self.C:
            G_min = self.__get_stack_rows(atype='identity')
            G = cvxopt.matrix(np.vstack((G_max, G_min)))

            h_max = self.__get_stack_rows()
            h_min = self.__get_stack_rows(multiplier=self.C)
            h = cvxopt.matrix(np.vstack((h_min, h_max)))

        return cvxopt.solvers.qp(Q, p, G, h, A, b)

        def __get_stack_rows(self, atype='zeros', multiplier=1):
            """
                Returns an array of type specified
            """
            return getattr(np, atype)(self.n_samples) * multiplier

    def predict(self, X):
        """
            Iterates through samples, determining the labels
            of samples by the support vectors
        """

        y_pred = []

        for samples in X:
            pred = 0

            for mult in range(len(self.lagrng_mltpliers)):
                pred += self.lagrng_mltpliers[mult] * \
                    self.sv_labels[mult] * \
                    self.kernel(self.s_vectors[mult], samples)

                pred += self.intercept
            y_pred.append(np.sign(pred))

        return np.array(y_pred)
