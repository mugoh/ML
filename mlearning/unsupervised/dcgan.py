"""
    Deep Convolitional Generative Adversarial Network
"""


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
