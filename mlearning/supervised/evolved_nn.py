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
    model_count = 0

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

        clf.show_model_details(
            'Neuroevolution') if not self.model_count else None
        self.model_count = + 1  # Show individual summary for first model

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

        # Select x % [40] highest individuals for next gen

        n_fittest = int(self.plt_size * .4)
        n_parents = self.plt_size - n_highest

        for epoch in range(n_generations):
            self.determine_fiteness()
            fittest = self.__sort_with_fitness()

            print(f'{epoch}  Fittest individual -  ' +
                  f'Fitness: {fittest:.2f}  ' +
                  f'Accuracy: { 100* fittest.accuracy:.2f }')
            self.population = [self.population[fit] for fit in n_fittest]
            total_fitness = np.sum(model.fitness for model in self.population)

            # Probability of selection of parent proportional to fitness
            parent_probs = [model.fitness /
                            total_fitness for model in self.population]

            # Without distribution - Preserve diversity
            parents = np.random.choice(self.population,
                                       size=n_parents,
                                       replace=False,
                                       p=parent_probs)
            self.produce_offsring()

    def determine_fitness(self):
        """
          Gets fitness scores from evaluation of Neural Networks
          on the test data
        """
        for indv in self.population:
            loss, acc = indv.test_on_batch(self.X, self.y)

            indv.fitness = 1 / (loss + 1e8)
            indv.accuracy = acc

    def __sort_with_fitness(self):
        """
            Orders the population starting from the fittest
            individual.
            Returns the fittest individual
        """

        ft = np.argsort(
            [individual.fitness for individual in self.population])
        self.population = [self.population[i] for i in ft]

        return self.population[0]
