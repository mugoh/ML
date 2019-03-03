"""
    This module contains the regression models
"""


class Regression:
    """
    Models the relationship between a scalar independent and a dependent variable
    """

    def __init__(self, no_of_iters, step_rate):
        """
        no_of_iters: Training iterations to run on the weight
        step_rate: The length of the step to be used in updating the weights

        """
