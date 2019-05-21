"""
    Training of Neural Network using Particle Swam Optimization
"""


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

    def evolve(self, X, y, n_gens):
        """
            Evolves the network for n number of generations
        """
