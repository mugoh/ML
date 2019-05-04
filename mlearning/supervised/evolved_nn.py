"""
    Neuroevolution Model
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import Dense, Activation
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..deep_learning.grad_optimizers import Adam

import numpy as np

from terminaltables import AsciiTable


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
    model_count = True

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

        if self.model_count:  # Show individual summary for first model
            clf.show_model_details('Neuroevolution')
            self.model_count = False

        return clf

    def evolve(self, n_generations=None):
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
        EvolvedNN.summarize(self.plt_size, self.mtn_rate, n_generations)
        epoch = 0

        # Select x % [40] highest individuals for next gen

        n_fittest = int(self.plt_size * .4)
        n_parents = self.plt_size - n_fittest

        while True:
            self.determine_fitness()
            fittest = self.__sort_with_fitness()

            print(f'{epoch}  Fittest individual -  ' +
                  f'Fitness: {fittest.fitness:.5f}  ' +
                  f'Accuracy: {100* fittest.accuracy:.2f}%')

            total_fitness = np.sum(model.fitness for model in self.population)

            # Probability of selection of parent proportional to fitness
            parent_probs = [model.fitness /
                            total_fitness for model in self.population]

            # Without distribution - Preserve diversity
            self.parents = np.random.choice(self.population,
                                            size=n_parents,
                                            replace=False,
                                            p=parent_probs)
            self.produce_offspring(len(self.parents), n_fittest)
            epoch += 1

            if n_generations and epoch == n_generations:
                break
            elif fittest.accuracy >= 1:
                return fitness
                break

        return fittest

    def determine_fitness(self):
        """
          Gets fitness scores from evaluation of Neural Networks
          on the test data
        """
        for indv in self.population:
            loss, acc = indv.test_on_batch(self.X, self.y)

            indv.fitness = 1 / (loss + 1e-8)
            indv.accuracy = acc

    def produce_offspring(self, no_prnts, no_fittest):
        """
           Creates the next generation
        """

        next_population = [self.population[i]
                           for i in range(no_fittest)]

        for i in np.arange(0, no_prnts, 2):
            offspr_a, offsp_b = self.crossover(self.parents[i],
                                               self.parents[i + 1])
            next_population += [self.mutate(offspr_a),
                                self.mutate(offsp_b)
                                ]
        self.population = next_population

    def crossover(self, *parents):
        """
            Peforms crossovers between parents to
            give new offspring
        """

        children = [self.build_model(parent.id_ + 1)
                    for parent in parents]

        [
            self.__inherit_weights(child, parents[children.index(
                child)])
            for child in children
        ]
        child_a, child_b = children

        for i in range(len(child_a.input_layers)):
            if hasattr(child_a.input_layers[i], 'weight'):

                n_neurons = child_a.input_layers[i].weight.shape[1]
                limit = np.random.randint(0, n_neurons)

                self.cross_neuron_weights(child_a, parents[1], i, limit)
                self.cross_neuron_weights(child_b, parents[0], i, limit)

        return child_a, child_b

    def cross_neuron_weights(self, child, parent, indx, cut_off):
        """
            Crosses over using individuals neuron weights
        """
        expected_weights = ['weight', 'weight_out']

        for wt in expected_weights:

            getattr(child.input_layers[indx], wt)[:, cut_off:] = \
                getattr(parent.input_layers[indx], wt)[
                :, cut_off:].copy()

    def __sort_with_fitness(self):
        """
            Orders the population starting from the fittest
            individual.
            Returns the fittest individual
        """

        ft = np.argsort(
            [individual.fitness for individual in self.population])[::-1]

        self.population = [self.population[i] for i in ft]

        return self.population[0]

    def __inherit_weights(self, offspring, parent):
        """
            Inherits weights from parent, passed on to
            offspring
        """
        for i in range(len(offspring.input_layers)):
            if hasattr(offspring.input_layers[i], 'weight'):
                offspring.input_layers[
                    i].weight = parent.input_layers[i].weight.copy()
                offspring.input_layers[i].weight_out = parent.input_layers[
                    i].weight_out.copy()

        return offspring

    def generate_mutn_mask(self, size):
        """
            Generates a mutation mask as a binomial distribution
        """

        return np.random.binomial(1, p=self.mtn_rate, size=size)

    def mutate(self, model):
        """
            Adds a zero mean Gauassian noise to the layer weights
            with a probability of the mutation rate
        """
        for layer in model.input_layers:
            if hasattr(layer, 'weight'):
                layer.weight += np.random.normal(
                    size=layer.weight.shape) * \
                    self.generate_mutn_mask(size=layer.weight.shape)

                layer.weight_out += np.random.normal(
                    size=layer.weight_out.shape) * \
                    self.generate_mutn_mask(size=layer.weight_out.shape)
        return model

    @classmethod
    def summarize(clf, *data):
        """
            Gives a summary of the model details
        """
        model_info = [['Population Size', 'Mutation rate', 'No. Generations']]
        model_info.append(data)

        print('\n', AsciiTable(model_info).table, '\n')
