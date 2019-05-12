"""
    Naive Bayes Classifier
"""

import numpy as np


class NaiveBayes:
    """
        Naive Bayes Classfier: Assumes indipendence among predictors

        Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
        Posterior = Likelihood * Prior / Scaling Factor
    """

    def __init__(self, *args, **kwargs):
        super(NaiveBayes, self).__init__(*args, **kwargs)

        self.params = []
        self.eps = kwargs.get('eps')

    def fit(self, X, y):
        """
            Finds the mean and variance of the features
        """
        self.X, self.y = X, y
        self.classes = np.unicode(y)

        self.params = np.zeros(self.classes.shape[0])

        for i, col in enumerate(self.classes):
            X_col = X[np.where(y == col)]

            self.params[i].append(
                [(col.mean(), col.var()) for col in X_col.T]
            )
