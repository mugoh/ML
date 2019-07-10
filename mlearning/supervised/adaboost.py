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
        self.X, self.y = X, y

        # Init weights to 1/n
        weights = np.full(self.n_samples, 1 / self.n_samples)

        for i in range(self.n_classifiers):
            clf = DecisionStump()
            min_error = float('inf')

            # Find best threshold for y prediction
            for feat_idx in range(self.n_feat):
                feature_val = np.expand_dims(X[:, feat_idx], axis=1)
                uniques = np.unique(feature_val)

                for threshold in uniques:
                    self.p = 1

                    prediction = np.ones(y.shape)
                    # Sample values below index. Label -1
                    prediction[X[:, feat_idx] < threshold] = -1

                    self.error = np.sum(weights[y != prediction])

                    new_min_err = self._reclassify_on_error(
                        min_error)
                    if new_min_err:
                        min_error = self.error
                        clf.polarity = self.p
                        clf.threshold = threshold
                        clf.feature_idx = feat_idx
            self.approximate_proficiency(clf, min_error, weights)
            self._calculate_weights(clf, weights)
            self.classifiers.append(clf)

    def approximate_proficiency(self, clf, min_error):
        """
            Calculates the value of alpha, used in updating the
            sample weights
        """
        clf.aplha = 0.5 * np.log(
            (1 - min_error) / min_error + 1e-8)

    def _calculate_weights(self, clf, weights):
        """
            Calculates new weights.
            Lower for misclassified samples, and higher for samples
            correctly classified.
        """
        preds = np.ones(self.y.shape)
        # Index for sample values below threshold
        negtv_idx = (clf.polarity *
                     self.X[:, clf.feature_idx] < clf.polarity * clf.threshold)
        preds[negtv_idx] = -1
        weights *= np.exp(-clf.alpha * self.y * preds)

        # Normalize to one
        weights /= np.sum(weights)

    def reclassify_on_error(self, min_err):
        """
            Flips polarity of samples based on error value
        """
        it_happens = self.error > .5

        if it_happens:
            self.error = 1 - err
            self.p = -1

        return False if not self.error < min_err else True


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
