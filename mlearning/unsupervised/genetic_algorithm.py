"""
    Implements a Genetic Algorithm which aims to produce
    a user specified target string. Candidates fitness is
    based on the alphabetical distance between candidate
    and target.
    ----
    Probability of selection as parent is dependent on the
    candidate's fitness.
    ---
    Offspring are produced as a single crossover point
    between pairs of parents, where mutation will involve
    random assignment of new characters with uniform probability.
"""

import numpy as np

import string

from terminaltables import AsciiTable


class GeneticAlgorithm:
    """
        A model of the Genetic Algorithm.
        It attempts to produce a specified input string

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

    def _mutate(self, indv):
        """
            Changes the individual's genes in a random manner
        """
        indv = list(indv)

        for i in range(len(indv)):
            # Change made with probability of mutation rate
            if np.random.random() < self.mutation_rate:
                indv[i] = np.random.choice(self.chars)
        return ''.join(indv)

    def _cross_over(self, parent_one, parent_two):
        """
            Creates offspring from parents by crossover.
            Crossover point is randomly selected.
        """
        cross_idx = np.random.randint(0, len(parent_one))
        offspring_a = parent_one[:cross_idx] + parent_two[cross_idx:]
        offspring_b = parent_two[:cross_idx] + parent_one[cross_idx:]

        return offspring_a, offspring_b

    def run(self, iters):
        """
            Starts the genetic algorithm
        """

        self.init_population()

        for epoch in range(iters):
            fitness = self.determine_fitness()
            fittest_indv = self.population[np.argmax(fitness)]
            best_fitness = max(fitness)

            if fittest_indv == self.target:  # Found individual
                break
            # Probability that individual selected as parent
            # set proportional to fitness

            parent_prob = [fit / sum(fitness) for fit in fitness]

            # Next gen
            population_ = []
            for i in np.arange(0, self.pltn_size, 2):
                parent_a, parent_b = np.random.choice(
                    self.population, size=2, replace=False, p=parent_prob)
                child_a, child_b = self._cross_over(parent_a, parent_b)
                population_ += (
                    self._mutate(child_a),
                    self._mutate(child_b)
                )

            self.population = population_
            print(f'{epoch} Closest candidate: {fittest_indv}, \
                Fitness: {best_fitness:.2f}')
        print(f'[{epoch} Answer: {fittest_indv}]')

    def summarize(self):
        """
            Outputs a summary of the model details.
        """
        print(AsciiTable([['Genetic Algorithm']]).table)

        data = [
            ['Target String', 'Mutation Rate', 'Population Size'],
            [self.target, self.mtn_rate, self.pltn_size]
        ]

        print(AsciiTable(data).table)
