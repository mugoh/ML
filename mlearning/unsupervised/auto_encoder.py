import numpy as np
from dataclasses import dataclass
from typing import Any

from ..helpers.deep_learning.loss import MSE
from ..helpers.deep_learning.network import Neural_Network
from ..deep_learning.grad_optimizers import Adam


@dataclass
class AutoEncoder:
    """
        A fully connected NN encoder
    """
    image_rows: int = 28
    image_cols: int = 28
    loss_function: Any = MSE
    optimizer: Any = Adam(learning_rate=0.0002, beta1=0.5)

    def __post__init__(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = Neural_Network(
            optimizer=self.optimizer, loss=self.loss_function)
