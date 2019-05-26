"""
    Deep Convolitional Generative Adversarial Network
"""

from ..helpers.deep_learning.network import Neural_Network
from ..helpers.deep_learning.loss import CrossEntropyLoss
from ..helpers.deep_learning.layers import (
    ConvolutionTwoD, Activation, DropOut, BatchNormalization, Dense, Flatten)
from ..deep_learning.grad_optimizers import Adam


class DCGAN:
    """
            Models a Deep Convolitional Generative Adversarial Network

    """

    def __init__(self, optimizer, loss_function):
        self.image_rows = 28
        self.image_cols = 28
        self.channels = 1
        self.latent_dims = 100
        self.img_shape = [self.channels, self.image_rows, self.image_cols]

        self.build_discriminator(optimizer, loss_function)
        self.build_gen(optimizer, loss_function)

    def build_disctiminator(self, optimizer, loss_function
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
        model.add_layer(Activation('reaky_relu'))
        model.add_layer(DropOut(.25))

        model.add_layer(BatchNormalization(momentum=.8))
        model.add_layer(ConvolutionTwoD(128,
                                        filter_shape=(3, 3), stride=2))
        model.add_layer(Activation('reaky_relu'))
        model.add_layer(DropOut(0.25))

        model.add_layer(BatchNormalization(momentum=.8))
        model.add_layer(ConvolutionTwoD(256, filter_shape=(3, 3),
                                        stride=1))
        model.add_layer(Activation('reaky_relu'))
        model.add_layer(DropOut(0.25))
        model.add_layer(Flatten())
        model.add_layer(Dense(units=2))
        model.add_layer(Activation('softmax'))

        return model
