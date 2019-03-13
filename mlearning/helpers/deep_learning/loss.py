"""
    This module contains deep learning loss functions
"""

import numpy


class CrossEntropyLoss:
    """
        Perform operationd that aid in computing the
        cross entropy loss between predicted and true data values.
    """

    def compute_loss(self, y_true, y_prediction):
        pred = numpy.clip(y_prediction, 1e-15, 1 - 1e-15)

        return -y_true * numpy.log(pred) - (1 - y_true) * numpy.log(1 - pred)
