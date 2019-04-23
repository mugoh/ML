"""
    This module contains a deep Q network model
    // https://arxiv.org/abs/1312.5602
"""

import gym

import numpy as np

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.loss import SquareLoss
from ..helpers.deep_learning.layers import Dense, Activation
from ..deep_learning.grad_optimizers import Adam


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
        self.build_model(self.no_states, self.no_actions)

    def init_env(self, env_name):
        """
            Initializes the specified environment
        """
        self.env = gym.make(env_name)
        self.no_states = self.env.observation_space.shape[0]
        self.no_actions = self.env.action_space.n

    def memorize(self, *args):
        """
            Appends states, action and reward to memory
        """
        self.memory.append(args)
        self.memory.pop(0) if len(self.memory > self.mem_size) else None

    def __choose_action(self, state):
        """
            Selects a random action for the agent
        """
        if np.random.random_sample() < self.epsilon:
            action = np.random.randint(self.no_actions)

        else:
            # Action with highest predicted utility in given state
            action = np.argmax(self.model.make_prediction(state), axis=1)[0]

        return action

    def build_model(self, inputs, outputs):
        """
            Builds the neural network model
        """
        clf = Neural_Network(optimizer=Adam, loss=SquareLoss)
        clf.add_layer(Dense(units=64, input_shape=(inputs, )))
        clf.add_layer(Activation('ReLu'))
        clf.add_layer(Dense(units=outputs))

        self.model = clf

    def train(self, no_of_epochs=500, batch_size=32):
        """
            Trains the model: Rewards the agent as it
            transitions between states
        """
        max_reward = 0
        epoch_loss = []
        trained = False

        for epoch in range(no_of_epochs):
            state = self.env.reset()
            summed_reward = 0

            epoch_loss.clear()

            while not trained:
                action = self.__choose_action(state)
                new_state, reward, trained, _ = self.env.step(action)
                self.memorize(state, action, reward, new_state, trained)

                batch_size_ = min(len(self.memory), batch_size)
                replay = np.random.choice(self.memory, size=batch_size_)
                X, y = self.__create_training_set(replay)

                # Learn control policy
                loss = self.model.train_on_batch(X, y)
                epoch_loss.append(loss)
                state = new_state
                summed_reward += reward

            epoch_loss = np.mean(epoch_loss)
            self.epsilon = self.min_epsilon + \
                (1 - self.min_epsilon) * np.exp(-self.decay_rate * epoch)

            max_reward = max(max_reward, summed_reward)

            print(f'{epoch} [Loss: {epoch_loss:.4f} Epsilon: {self.epsilon} \
                Reward: {total_reward}, Max Reward: {max_reward}]')
        print('\nTraining Complete')
