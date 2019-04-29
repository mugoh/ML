"""
    Neuroevolution Model
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import Dense, Activation
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..deep_learning.grad_optimizers import Adam

import numpy as np


class EvolvedNN:
    """
        Evolutionary optimized Neural Network model

        Parameters:
        ----------
        X: array_like
            X training dataset
        y: array_like
            Dataset labels
        n_individuals: int
            Number of neural networks to form the population
        mutation_rate: float
            Probability of mutating a given network weight
    """

    def __init__(self, X, y,
                 n_individuals=1000,
                 mutation_rate=0.01):
        self.X = X
        self.y = y
        self.mtn_rate = mutation_rate
        self.plt_size = n_individuals

    def init_population(self):
        """
            Initializes the population formed by the Neural Network
        """

        self.population = [
            self.build_model(
                np.random.randint(1000)
            ) for _ in range(self.plt_size)
        ]

    def build_model(self, id_):
        """
            Creates a new individual (net)
        """
        n_inputs, n_ouputs = self.X.shape[1], self.y.shape[1]

        clf = Neural_Network(optimizer=Adam(),
                             loss=CrossEntropyLoss)
        clf.add_layer(Dense(units=16, input_shape=(n_inputs,)))
        clf.add_layer(Activation('ReLu'))
        clf.add_layer(Dense(n_ouputs))
        clf.add_layer(Activation('softmax'))

        clf.fitness = clf.accuracy = 0
        clf.id_ = id_

        return clf

    def evolve(self, n_generations):
        """
            Evolves the population for a given number of
            generations based on the dataset X, and labels y

            Parameters
            ----------
            n_generations: int
                The number of generations for which to evolve
                the network
        """
        self.init_population()
