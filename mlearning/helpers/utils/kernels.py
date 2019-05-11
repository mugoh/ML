"""
    This module contains clustering algorithm kernels
"""

import numpy as np


def compute_polynomial(**kwargs):
    """
        Polynomial kernel
    """
    def fun(x_one, x_two):
        return (
            np.inner(x_one, x_two) + kwargs.get('cf')
        ) ** kwargs.get('power')

    return fun
