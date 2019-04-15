"""
    This mofile contains the restricted Boltzmann Machine model
"""


class RBM:
    """
        Bearnoulli Restricted Boltzmann Machine

        Parameters:
        ------------
        hidden: int
            No of processing nodes in the hidden layer
        l_rate: float
            Step length used in updating weights
        batch_size: int
            Size of the mini-batch used to calculate each weight update
        iter: int
            Number of training iterations to tune the weight for

    """

    def __init__(self, hidden=128, l_rate=0.1, batch_size=10, iters=100):
        self.no_of_iters = iters
        self.batch_size = batch_size
        self.learning_rate = l_rate
        self.hidden = hidden
