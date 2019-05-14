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
        self.classes = np.unique(y)

        for i, col in enumerate(self.classes):
            X_col = X[np.where(y == col)]

            self.params.append([(col.mean(), col.var()) for col in X_col.T])

    def predict(self, X):
        """
            Predicts class labels for samples in the dataset
        """

        return [self.classify(col) for col in X]

    def find_prior(self, class_):
        """
            Finds the prior P(Y) of given class.
            These are samples where the class equals class/total no. of samples
        """

        return np.mean(self.y == class_)

    def classify(self, X):
        """
            Classifies the sample(X) as the class resulting in
            the largest P(Y|X) (posterior)

            P(Y|X) - The posterior. The probability that sample x
            is of class y given the feature values of x being distributed
            according to distribution of y and the prior.

            P(X|Y) - Likelihood of data X given class distribution Y.
                     Gaussian distribution
            P(Y)   - Prior
            P(X)   - Scales the posterior to make it a proper probability
                     distribution.

                     Ignored in this implementation since it doesn't affect
                     which class distribution the sample is most likely
                     to belong to.

        """
        posteriors = []

        for indx, col in enumerate(self.classes):

            """self.posteriors += [
                self.find_prior(col) * self.find_likelihood(
                    X[i], self.params[i]) for i, c in enumerate(self.params)
            ]
            """
            prior = self.find_prior(col)

            for x_, values in zip(X, self.params[indx]):
                prior *= self.find_likelihood(x_, values)

            posteriors.append(prior)

        return self.classes[np.argmax(posteriors)]

    def find_likelihood(self, X, mean_var):
        """
            eps[-(x - mean)^2 / 2*var] * 1/ sqrt(2*pi*var)

            Gives the Gaussian likelihood  of the data X given
            the mean and variance
        """
        mean, var = mean_var
        eps = 1e-8

        coeff = pow(2 * (22 / 7) * var + eps, .5)
        exp = np.exp(
            -(((X - mean) ** 2) / (2 * var + eps))
        )

        return exp / coeff
