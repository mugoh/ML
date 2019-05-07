"""
    Support Vector Machine
"""


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

    def __init__(self, kernel=rbf, penalty=1, power=4, cf=4, gamma=None):
        self.kernel = kernel
        self.penalty = penalty
        self.gamma = gamma
        self.bias = cf
        self.power = power
