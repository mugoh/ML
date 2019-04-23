"""
    This module contains a deep Q network model
    // https://arxiv.org/abs/1312.5602
"""

import gym


class DeepQNet:
    """
        Makes use of a deep learning neural network for prediction of
        the Q values.

        Parameters
        ----------
        env: string
            The environment to be explored by the agent
        epsilon: float
            Probability that the agent selects a random value
            and not one that maximizes the expected utility
        gamma: float
            Factors the extent the agent should consider future rewards
        decay: float
            Rate of epsilon value decay for each epoch
        min_eps: float
            Value to be approached as epsilon as training progresses
    """

    def __init__(self, env, gamma=0.9, epsilon=1, decay=0.005, min_eps=0.1):
        self.epsilon = epsilon
        self.min_epsilon = min_eps
        self.gamma = gamma
        self.decay_rate = decay

        self.memory = []
        self.mem_size = 300

        self.init_env(env)

    def init_env(self, env_name):
        """
            Initializes the specified environment
        """
        self.env = gym.make(env_name)
        self.no_states = self.env.observation_space.shape[0]
        self.no_actions = self.env.action_space.n
