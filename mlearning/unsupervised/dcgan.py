"""
    Deep Convolutional Generative Adversarial Network
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.deep_learning.layers import (
    ConvolutionTwoD, Activation, DropOut, BatchNormalization, Dense, Flatten)
from ..deep_learning.grad_optimizers import Adam


import numpy as np


class DCGAN:
    """
            Models a Deep Convolutional Generative Adversarial Network

    """

    def __init__(self, optimizer, loss_function):
        self.image_rows = 28
        self.image_cols = 28
        self.channels = 1
        self.latent_dims = 100
        self.img_shape = [self.channels, self.image_rows, self.image_cols]

        self.discriminator = self.build_discriminator(optimizer, loss_function)
        self.gen = self.build_gen(optimizer, loss_function)

    def build_discriminator(self, optimizer, loss_function
                            ):
        """
            Creates the network discriminator
        """
        model = Neural_Network(optimizer=optimizer, loss=loss_function)

        model.add_layer(ConvolutionTwoD(no_of_filters=32,
                                        filter_shape=(3, 3),
                                        stride=2, input_shape=self.img_shape))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(DropOut(p=.25))
        model.add_layer(ConvolutionTwoD(64, filter_shape=(3, 3), stride=2))
        model.add_layer(ZerosPadding2D(padding=((0, 1), (0, 1))))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(DropOut(.25))

        model.add_layer(BatchNormalization(momentum=.8))
        model.add_layer(ConvolutionTwoD(128,
                                        filter_shape=(3, 3), stride=2))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(DropOut(0.25))

        model.add_layer(BatchNormalization(momentum=.8))
        model.add_layer(ConvolutionTwoD(256, filter_shape=(3, 3),
                                        stride=1))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(DropOut(0.25))
        model.add_layer(Flatten())
        model.add_layer(Dense(units=2))
        model.add_layer(Activation('softmax'))

        return model

    def build_gen(self, optimizer, loss_function):
        """
            Builds the model discriminator
        """

        model = Neural_Network(optimizer=optimizer, loss=loss_function)

        model.add_layer(Dense(units=128 * 7 * 7, input_shape=(100,)))
        model.add_layer('leaky_relu')
        model.add_layer(Reshape((128, 7, 7)))
        model.add_layer(BatchNormalization(momentum=0.8))
        model.add_layer(UpSampling2D())

        model.add_layer(ConvolutionTwoD(
            no_of_filters=128, filter_shape=(3, 3)))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(BatchNormalization(.8))
        model.add_layer(UpSampling2D())

        model.add_layer(ConvolutionTwoD(64, filter_shape=(3, 3)))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(BatchNormalization(.8))
        model.add_layer(ConvolutionTwoD(no_of_filters=1,
                                        filter_shape=(3, 3)))
        model.add_layer(Activation('tanh'))

        return model

    def train(self, X, y, epochs, batch_size, save_interval):
        """
            Trains the model
        """

        self.X = X
        self.y = y

        for epoch in range(epochs):
            self.train_discriminator(batch_size / 2)

    def train_discriminator(self, half_batch):
        """
            Trains the discriminator
        """
        self.discriminator.set_trainable(True)

        # Random half batch of images
        idx = np.random.randint(0, self.X.shape[0], half_batch)
        images = self.X[idx]

        # Sample noise for use as generator input
        noise = np.random.normal(size=(half_batch, 100))

        # Generate a half batch of images
        gen_images = self.gen.make_prediction(noise)

        valid = np.concatenate(
            (np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis=1)
        fake = np.concatenate(
            (np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis=1)
