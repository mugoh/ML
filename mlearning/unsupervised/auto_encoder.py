import numpy as np
from dataclasses import dataclass, field
from typing import Any

from ..helpers.deep_learning.loss import MSE
from ..helpers.deep_learning.network import Neural_Network
from ..deep_learning.grad_optimizers import Adam
from ..helpers.deep_learning.layers import (
    Dense, Activation, BatchNormalization)


@dataclass
class AutoEncoder:
    """
        A fully connected NN encoder
    """
    image_rows:
        int = 28
    image_cols:
        int = 28
    loss_function:
        Any = MSE
    optimizer:
        Any = Adam(learning_rate=0.0002, beta1=0.5)
    img_dim:
        int = field(default=image_rows * image_cols, init=False, repr=False)

    def __post__init__(self):
        self.latent_dims = 128  # For data embedding
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = Neural_Network(
            optimizer=self.optimizer, loss=self.loss_function)
        self.extend_layers()

    def build_encoder(self):
        """
            Creates the network encoder
        """
        model = Neural_Network(
            optimizer=self.optimizer,
            loss=self.loss_function)

        model.add_layer(Dense(units=512, input_shape=self.img_dim))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(BatchNormalization(momentum=.8))

        model.add_layer(Dense(units=512))
        model.add_layer(Activation('leaky_relu'))
        model.add_layer(BatchNormalization(momentum=.8))
        model.add_layer(Dense(units=self.latent_dims))

        return model

        def build_decoder(self
                          ):
            """
                Creates the network decoder
            """
            model = Neural_Network(
                optimizer=self.optimizer,
                loss=self.loss_function)

            model.add_layer(Dense(units=512, input_shape=self.img_dim))
            model.add_layer(Activation('leaky_relu'))
            model.add_layer(BatchNormalization(momentum=.8))

            model.add_layer(Dense(units=512))
            model.add_layer(Activation('leaky_relu'))
            model.add_layer(BatchNormalization(momentum=.8))
            model.add_layer(Dense(units=self.img_dim))
            model.add_layer(Activation('tanh'))

            return model

        def extend_layers(self):
            """
                Appends the encoder and decoder layers to the
                autoencoder
            """

            self.autoencoder.layers.extend(self.encoder.layers)
            self.autoencoder.layers.extend(self.decoder.layers)

            self.autoencoder.show_model_details('Variational Autoencoder')

        def train(self, X, n_epochs=49, batch_size=128, save_interval=49):
            """
                Trains autoencoder model
            """
            for epoch in range(n_epochs):
                idx = np.random.randint(0, X.shape[0], batch_size)
                imgs = X[idx]

                loss, acc = self.autoencoder.train_on_batch(imgs, imgs)
                print(f'[ {epoch} loss: {loss}, accuracy: {acc}]  ')

                if not epoch % save_interval:
                    self.save_imgs(epoch, X)
