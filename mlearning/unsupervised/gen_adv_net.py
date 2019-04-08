"""
    Generative Adversarial Network
"""
from ..deep_learning.grad_optimizers import Adam

from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import (
    Dense, DropOut, Activation, BatchNormalization)

from sklearn.datasets import fetch_mldata
import numpy as np


class Generative_Adversarial_Net:
    """
        A Generative Adversarial Network model with deep Neural
        Networks as the Generator and Discriminator
        It makes use of MNIST handwritten digits data

        Parameters
        ----------
        rows: int
            Number of image rows
        cols: int
            Number of image columns
    """

    def __init__(self, rows=28, cols=28):
        self.img_rows = rows
        self.img_cols = cols
        self.img_dimensions = self.img_rows * self.img_cols
        self.latent_dimensions = 100

        optimizer = Adam(learning_rate=0.002, beta1=0.5)

        self.make_generative(optimizer, CrossEntropyLoss)

    def make_generative(self, optimizer, loss_funct):
        """
            Builds the Net's Generator and Discriminator
        """
        self.discriminator = self.build_discriminator(optimizer, loss_funct)
        self.generator = self.create_generator(optimizer, loss_funct)

    @classmethod
    def build_discriminator(cls, optimizer, loss):
        """
            Creates the GAN discriminator
        """
        net = Neural_Network(optimizer, loss)

        net.add_layer(Dense(512, input_shape=(cls.img_dimensions, )))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(DropOut(0.5))

        net.add_layer(Dense(256))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(DropOut(0.5))

        net.add_layer(Dense(2))
        net.add_layer(Activation('softmax'))

        return net

    @classmethod
    def create_generator(cls, optimizer, loss_func):
        """
            Creates the GAN generator
        """
        net = Neural_Network(optimizer, loss_func)

        net.add_layer(Dense(256, input_shape=(cls.latent_dimensions, )))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(512))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(1024))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(cls.img_dimensions))
        net.add_layer(Activation('tanh'))

    def train(self, no_of_epochs, batch_size=128, save_interval=50):
        """
            Trains the network
        """

        mnist = fetch_mldata('MNIST original')

        X = mnist.data
        y = mnist.target

        # Rescale data -> -1, 1
        X = (X.astype(np.float32) - 127.5) / 127.5
        half_batch = batch_size / 2

        # Train the discriminator
        for epoch in range(no_of_epochs):
            self.discriminator.set_trainable(True)

            # Select a random half-batch of images
            index = np.random.randint(0, X.shape[0], half_batch)
            images = X.get(index)