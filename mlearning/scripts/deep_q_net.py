"""
    This module contains a function to run
    and output the Deep Q Network
"""

from ..reinforcement_learning.q_network import DeepQNet


def start_deepq_net():
    """
        Initializes the Q network model
    """
    dq_net = DeepQNet(env='CartPole-v1',
                      epsilon=0.9,
                      discount_factor=.8)
    dq_net.model.show_model_details('Deep Q Network')
    dq_net.train()
    dq_net.shake_it(no_of_epochs=100)
