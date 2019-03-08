"""
    This module contains optimizers that utiilze gradient descents
    for finding weights at global loss minima.
"""


class Adam:
    """
        Adaptive Moment Estimation
        Maintains adaptive learning rates for each parameter (first moment)
        and an exponentially decaying average of past gradients (momentum).
    """
