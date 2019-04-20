import numpy as np

import string


class GeneticAlgorithm:
    """
        A model of the Genetic Algorithm.
        It will attempt to produce the specified
        input string

        Parameters
        -----------
        target_str: string
            The string the Algorithm should attempt to reproduce
        population: int
            Size of the population (Possible solutions)
        mutation_rate: float
            Probability at which alleles should be changed(chars)
    """

    def __init__(self, target_str, population, mutation_rate):
        self.target = target_str
        self.pltn_size = population
        self.mtn_rate = mutation_rate
        self.chars = [' '] + list(string.ascii_letters)
        self.population = []

    def init_population(self):
        """
            Initializes the population
        """
        for _ in range(self.pltn_size):
            individual = ''.join(np.random.choice(
                self.chars, size=len(self.target)))
        self.population.append(individual)
