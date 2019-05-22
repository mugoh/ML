"""
    Training of Neural Network using Particle Swam Optimization
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import Dense, Activation
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..deep_learning.grad_optimizers import Adam


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

        self.init_model_velocity(clf)
        self.model = clf

    def init_model_velocity(self, model):
        """
            Sets the velocity for each individual
        """

        for layer in model.layers:
            velocity = {'w': 0, 'w_o': 0}
            if hasattr(layer, 'weights'):
