"""
    This module contains a Convolutional Neural Network example
"""
import matplotlib.pyplot as plot
import numpy as np

from sklearn import datasets

from ..helpers.deep_learning.network import Neural_Network
from ..deep_learning.grad_optimizers import Adam
from ..helpers.utils.data_utils import data_helper
from ..helpers.deep_learning.layers import (
    ConvolutionTwoD, Activation, DropOut, BatchNormalization,
    Flatten, Dense)
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.utils.display import plot_dimensioner


class CNN:
    """
        This class holds the implemented convolution
        neral network example
    """

    def convolute(self):
        """
            Runs convolution to a data input
        """
        #
        # Replace ascii tables with astropy.io.ascii
        optimizer = Adam()

        digits = datasets.load_digits()
        X = digits.data
        y = digits.target

        # Convert to one hot encoding

        y = data_helper.categorize(y.astype('int'))

        X_train, X_test, y_train, y_test = data_helper.split_train_test(
            X, y, test_size=0.4, seed=1)

        # Reshape to no. of samples, channels, height, width
        X_train = X_train.reshape((-1, 1, 8, 8))
        X_test = X_test.reshape((-1, 1, 8, 8))

        self.classifier = Neural_Network(
            optimizer, CrossEntropyLoss, (X_test, y_test))

        self.add_layers()

        self.classifier.show_model_details('Convolution Network')
        training_err, validation_err = self.classifier.fit(
            X_train, y_train, no_of_epochs=50, batch_size=256)

        count = len(training_err)
        self.training, = plot.plot(range(count),
                                   training_err,
                                   label='Trainig Error')
        self.validation, = plot.plot(range(count),
                                     validation_err,
                                     label='Validation Error')
        self.output()

    def add_layers(self):
        """
        Adds network layers to the classifier
        """
        self.classifier.add_layer(ConvolutionTwoD(
            no_of_filters=16, filter_shape=(3, 3), input_shape=(1, 8, 8),
            padding=True))

        self.classifier.add_layer(Activation('ReLu'))
        self.classifier.add_layer(DropOut(0.25))
        self.classifier.add_layer(BatchNormalization())

        self.classifier.add_layer(
            ConvolutionTwoD(
                no_of_filters=32,
                filter_shape=(3, 3),
                stride=1,
                padding=True))
        self.classifier.add_layer(Activation('ReLu'))
        self.classifier.add_layer(DropOut(0.25))
        self.classifier.add_layer(BatchNormalization())

        self.classifier.add_layer(Flatten())
        self.classifier.add_layer(Dense(256))
        self.classifier.add_layer(Activation('ReLu'))
        self.classifier.add_layer(DropOut(0.4))
        self.classifier.add_layer(BatchNormalization())

        self.classifier.add_layer(Dense(10))
        self.classifier.add_layer(Activation('softmax'))

    def output(self):
        """
            Displays output from the data convolution.
        """
        plot.legend(handles=[self.training, self.validation])
        plot.title('Error Plot')
        plot.ylabel('error')
        plot.xlabel('No. of iterations')
        plot.show()

        _, accuracy = self.classifier.test_on_batch(X_test, y_test)
        print(f'Accuracy: {accuracy}')

        y_prediction = np.argmax(
            self.classifier.make_prediction(X_test), axis=1)
        X_test = X_test.reshape(-1, 8 * 8)

        # Flatten dimension to Two-D
        plot_dimensioner.plot_in_two_d(X_test, y_prediction,
                                       title="Convolution Neural Network",
                                       accuracy=accuracy,
                                       legend_labels=range(10))


convolute = CNN().convolute
