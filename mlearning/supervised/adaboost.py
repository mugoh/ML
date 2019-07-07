"""
    Adaboost Classifier
    Boosting that makes a strong classifier from a number
    of weak classiffiers in ensemble.
"""
import numpy as np


class AdaBoost:
    """
        Decision stump [One-level tree] classifier

        Parameters:
        ----------
        classifiers: int
            Number of weak classifiers to used
    """

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        """
           Finds best threshold for prediction
        """

        self.n_samples, self.n_feat = X.shape
