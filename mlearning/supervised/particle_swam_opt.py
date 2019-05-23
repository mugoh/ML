"""
    Training of Neural Network using Particle Swam Optimization
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import Dense, Activation
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..deep_learning.grad_optimizers import Adam

import numpy as np
import copy


class ParticleSwamOptimizedNN:
    """
        Optimizes Neural Network using Particle Swam Optimization

        Parameters
        ----------
        population: int
            Number of neural networks in the population
        # inertia_weight: float
        # cognitive_weight: float
        # social_weight: float
        max_velocity: int
            Maximum value for velocity
    """

    def __init__(self, population,
                 inertia_weight,
                 cognitive_weight,
                 social_weight,
                 max_velocity):
        self.pop_size = population
        self.cognitive_w = cognitive_weight
        self.inertia_w = inertia_weight
        self.social_w = social_weight

        self.max_v = max_velocity
        self.min_v = - max_velocity

        self.population = []

    def evolve(self, X, y, n_gens):
        """
            Evolves the network for n number of generations
        """
        self.X = X
        self.y = y
        self.init_population()

        self.best_individual = self.population[0]

        for epoch in range(n_gens):
            for individual in self.population:
                self.update_weights(individual)
                self.find_fitness(individual)

                if individual.fitness > individual.highst_fitns:
                    individual.best_layers = copy.copy(individual.layers)
                    individual.highst_fitns = individual.fitness

                elif individual.fitness > self.best_individual.fitness:
                    self.best_individual = copy.copy(individual)

            print(f'[{epoch} Best Individual - ID {individual.id_} Fitness : {individual.fitness}]' +
                  f'Accuracy : {100* individual.accuracy:.2f}')
        return best_individual

    def init_population(self):
        """
            Initializes the network forming the population
        """

        self.population = [
            self.build_model(i) for i in range(self.pop_size)]

    def build_model(self, id_):
        """
            Creates a new individual
        """
        clf = Neural_Network(optimizer=Adam(), loss=CrossEntropyLoss)
        clf.add_layer(Dense(units=16, input_shape=(self.X[1], )))
        clf.add_layer(Activation('relu'))
        clf.add_layer(Dense(units=self.y.shape[1]))
        clf.add_layer(Activation('softmax'))

        clf.id_ = id_
        clf.fitness = clf.highst_fitns = clf.accuracy = 0
        clf.best_layers = clf.layers.copy()

        self.model = self.init_model_velocity(clf)

    def init_model_velocity(self, model):
        """
            Sets the velocity for each individual
        """
        model.velocity = []

        for layer in model.layers:
            velocity = {'w': 0, 'w_o': 0}
            if hasattr(layer, 'weight'):
                velocity = {'w': np.zeros_like(
                    layer.weights), 'w_o': np.zeros_like(layer.weight_out)}
                model.velocity.append(velocity)

        return model

    def update_weights(self, model):
        """
            Calculates new weight velocity for model, updating weights
            in each layer
        """
        r1 = np.random.uniform()
        r2 = np.random.uniform()

        for i in len(model.layers):
            if not hasattr(model.layers[i], 'weight'):
                continue
            first_term_w = self.intertia_w * model.velocity[i]['w']
            second_term_w = self.cognitive_w * r1 * \
                (model.best_layers[i].weight - model.layers[i].weight)
            third_term_w = self.social_w * r2 * \
                (self.best_individual.layers[i].weight -
                 model.layers[i].weight)

            velocity_ = first_term_w + second_term_w + third_term_w
            model.velocity[i]['w'] = np.clip(
                velocity_, self.min_v, self.max_v)

            # Bias weight velocity
            first_term_w_o = self.intertia_w * model.velocity[i]['w_o']
            sec_term_w_o = self.cognitive_w * r1 * \
                (model.best_layers[i].weight_out - model.layers[i].weight_out)
            third_term_w_o = self.social_w * r2 * \
                (self.best_individual.layers[i].weight_out -
                 model.layers[i].weight_out)

            velocity_ = first_term_w_o + sec_term_w_o + third_term_w_o
            model.velocity[i]['w_o'] = np.clip(
                velocity_, self.min_v, self.max_v)

            model.layers[i].weight += model.velocity[i].get('w')
            model.layers[i].weight_out += model.velocity[i].get('w_o')

    def find_fitness(self, model):
        """
            Evaluates the individual on the best set for
            fitness scores
        """

        loss, model.accuracy = model.test_on_batch(self.X, self.y)
        model.fitness = 1 / (loss + 1e-8)
