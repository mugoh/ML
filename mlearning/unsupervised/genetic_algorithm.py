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

    def determine_fitness(self):
        """
            Calculates the fitness of an individual
            in the population
        """
        fitness = []

        # Loss taken as the alphabetical distance between
        # chars in the individual and the target string

        for individual in self.population:
            loss_ = 0
            for char in range(len(individual)):
                char_i1 = self.chars.index(individual[char])
                char_i2 = self.chars.index(self.target[char])

                loss_ += abs(char_i1 - char_i2)
            fitness.append(1 / (loss_ + 1e-6))
        return fitness
