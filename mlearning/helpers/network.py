"""
    This module contains a class that modles the Deep Learning Neural Network
"""
import numpy
import progress_bar


class Neural_Network:
    """
        Represents the modeled neural network

        optimizer: class
                The descent optimizer that tunes the weight
                for minumum loss
        loss:   class
                    Measures the performance of the model
        validation_data: tuple
                        Contains the validation data items and (X, y) labels
    """

    def __init__(self, optimizer, loss, validation_data):
        self.optimizer = optimizer
        self.loss_func = loss()
        self.input_layers = []
        self.progress_bar = progress_bar.ProgressBar(widgets=bar_widgets)
        self.errs = {'training': [],
                     'validation': []
                     }

    def set_trainable(self, trainable):
        """
            Enables freezing of weights of the network input_layers

            ###################################################################
            # Freezing the model means producing a singular file containing  #
            # information about the graph and checkpoint variables,          #
            # but saving these hyperparameters as constants                  #
            within the graph structure.
            This eliminates additional information saved in the checkpoint
            files such as the gradients at each point, which are included
            so that the model can be reloaded and training continued from where
            you left off. As this is not needed when serving a model purely for
            inference they are discarded in freezing.

        """
        for layer in self.input_layers:
            layer.trainable = trainable

    def add_layer(self, new_layer):
        """
            Adds a layer to the neural network
        """

        #  Set input shape to output shape of last layer added
        if self.input_layers:
            new_layer.set_input_shape(
                shape=self.input_layers[-1].output_shape())

        # Layer contains weights that require initialization
        if hasattr(new_layer, 'initialize'):
            new_layer.initialize(optimizer=self.optimizer)
        self.layers.append(new_layer)

    def test_on_batch(self, X, y):
        """
            Evaluates the model over samples in a single batch.
        """

        y_prediction = self.forward_pass(X, training=False)
        loss = numpy.mean(self.loss_func.loss(y, y_prediction))
        acc = self.loss_func.acc(y, y_prediction)

        return loss, acc

    def forward_pass(self, X, training=True):
        """
            Calculates the output of the neural network
        """

        output_layer = X
        for layer in self.input_layers:
            output_layer = layer.forward_pass(output_layer, training)

        return output_layer
