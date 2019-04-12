"""
    Generative Adversarial Network
"""
from ..deep_learning.grad_optimizers import Adam

from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.layers import (
    Dense, DropOut, Activation, BatchNormalization)

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
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
        model_data: dict
            {'no_of_epochs':222int 'batch_size': 128, 'save_interval':50}
            To be used in training the network
    """
    sample_input = {'no_of_epochs': 20000,
                    'batch_size': 64,
                    'save_interval': 400
                    }

    def __init__(self, rows=28, cols=28, model_data={}):
        self.img_rows = rows
        self.img_cols = cols
        self.img_dimensions = self.img_rows * self.img_cols
        self.latent_dimensions = 100

        optimizer = Adam(learning_rate=0.002, beta1=0.5)

        self.make_generative(optimizer, CrossEntropyLoss)
        model_data = model_data if model_data else \
            Generative_Adversarial_Net.sample_input
        self.train(**model_data)

    def make_generative(self, optimizer, loss_funct):
        """
            Builds the Net's Generator and Discriminator
        """
        self.discriminator = self.build_discriminator(optimizer, loss_funct)
        self.generator = self.create_generator(optimizer, loss_funct)

        self.combined = Neural_Network(optimizer, loss_funct)
        self.combined.input_layers.extend(self.generator.input_layers)
        self.combined.input_layers.extend(self.discriminator.input_layers)

        self.generator.show_model_details('Generator')
        self.discriminator.show_model_details('Discriminator')

    def build_discriminator(self, optimizer, loss):
        """
            Creates the GAN discriminator
        """
        net = Neural_Network(optimizer, loss)

        net.add_layer(Dense(512, input_shape=(self.img_dimensions, )))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(DropOut(0.5))

        net.add_layer(Dense(256))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(DropOut(0.5))

        net.add_layer(Dense(2))
        net.add_layer(Activation('softmax'))

        return net

    def create_generator(self, optimizer, loss_func):
        """
            Creates the GAN generator
        """
        net = Neural_Network(optimizer, loss_func)

        net.add_layer(Dense(256, input_shape=(self.latent_dimensions, )))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(512))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(1024))
        net.add_layer(Activation('leaky_relu'))
        net.add_layer(BatchNormalization(momentum=0.8))

        net.add_layer(Dense(self.img_dimensions))
        net.add_layer(Activation('tanh'))

        return net

    def train(self, **kwargs):
        """
            Trains the network
        """
        no_of_epochs = kwargs.get('no_of_epochs')
        batch_size = kwargs.get('batch_size') or 128
        save_interval = kwargs.get('save_interval') or 50

        mnist = fetch_mldata('MNIST original')

        X = mnist.data

        # Rescale data -> -1, 1
        X = (X.astype(np.float32) - 127.5) / 127.5
        half_batch = batch_size // 2

        for epoch in range(no_of_epochs):

            self.train_discriminator(X, half_batch, epoch)
            self.train_gen(X, batch_size)

            if not epoch % save_interval:
                self.save_samples(epoch)

    def save_samples(self, epoch):
        """
            Saves generated sample images at the save interval
        """
        row, col = 5, 5
        noise = np.random.normal(0, 1, (row * col, self.latent_dimensions))

        # Generate and reshape images
        gen_images = self.generator.make_prediction(
            noise).reshape((-1, self.img_rows, self.img_cols))
        # Rescale images: 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        figure, axis = plt.subplots(row, col)
        plt.suptitle('Generative Adversarial Network')
        count = 0
        for i in range(row):
            for j in range(col):
                axis[i, j].imshow(gen_images[count, :, :], cmap='gray')
                axis[i, j].axis('off')
                count += 1
        figure.savefig('mnist_{epoch}.png')
        plt.close()

    def train_discriminator(self, X, half_batch, epoch):
        """
            Trains the discriminator for each epoch
        """
        self.discriminator.set_trainable(True)

        # Select a random half-batch of images
        index = np.random.randint(0, X.shape[0], half_batch)
        images = X[index]

        # Sample noise to use as Generator input
        noise = np.random.normal(
            0, 1, (half_batch, self.latent_dimensions))

        # Generate a half batch of images
        gen_images = self.generator.make_prediction(noise)

        # valid: [1, 0] invalid: [0, 1]
        valid = np.concatenate(
            (np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis=1)
        invalid = np.concatenate(
            (np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis=1)

        # Train discriminator
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(
            images, valid)
        d_loss_invalid, d_acc_invalid = self.discriminator.train_on_batch(
            gen_images, invalid)

        d_loss = .5 * (d_loss_real + d_loss_invalid)
        d_acc = .5 * (d_acc_real + d_acc_invalid)
        print(f'{epoch} Discriminator: Loss {d_loss: .3f} ,' +
              f'acc {d_acc * 100: .3f}')

    def train_gen(self, X, batch_size):
        """
           Trains the generator for each epoch
        """

        # Train only for the  combined model
        self.generator.set_trainable(False)

        noise = np.random.normal(
            0, 1, (batch_size, self.latent_dimensions))
        # Label generated samples as valid
        valid = np.concatenate(
            (np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=1)
        gen_loss, gen_acc = self.combined.train_on_batch(noise, valid)

        print(f'Generator: loss {gen_loss:.2f}, acc {gen_acc * 100:.2f}')
