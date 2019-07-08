"""
    Adaboost Classifier
    Boosting that makes a strong classifier from a number
    of weak classiffiers in ensemble.
"""
import numpy as np

from dataclasses import dataclass
from typing import Any


class AdaBoost:
    """
        Decision stump [One-level tree] classifier

        Parameters:
        ----------
        n_classifiers: int
            Number of weak classifiers to used
    """

    def __init__(self, n_classifiers):
        self.n_classifiers = n_classifiers
        self.classifiers = []

    def fit(self, X, y):
        """
           Finds best threshold for prediction
        """

        self.n_samples, self.n_feat = X.shape

        # Init weights to 1/n
        weights = np.full(self.n_samples, 1 / self.n_samples)

        for i in range(self.n_classifiers):
            clf = DecisionStump()


@dataclass
class DecisionStump:
    """
        Weak classifier for Adaboost

        Parameters
        ----------
        polarity: int
            Determines classification of sample as -1 or 1
            from threshold
        aplha: float
            Indicates classifiers accuracy
        feature_idx: int
            Index of feature used in making the classification
        threshold: int
            Threshold value against which feature is measured against
    """
    polarity: int = 1
    alpha: float = .02
    feature_idx: Any = None
    threshold: Any = None
