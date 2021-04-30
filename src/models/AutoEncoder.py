"""
The PyTorch module for a simple autoencoder.
"""
import torch.nn as nn
from src.utils.misc import get_sequential


class AutoEncoder(nn.Module):
    """
    A simple autoencoder.
    Initialized with Kaiming uniform weights.
    """

    def __init__(self, encoder_shape: [int]):
        """
        Initializes the Autoencoder with specified encoder shape. Decoder takes the reversed shape.

        :param encoder_shape: A list of ints representing the size of each layer in the encoder.

        >>> ae = AutoEncoder([10, 20, 30])
        >>> ae.encoder
        Sequential(
          (0): Linear(in_features=10, out_features=20, bias=True)
          (1): Sigmoid()
          (2): Linear(in_features=20, out_features=30, bias=True)
          (3): Sigmoid()
        )
        >>> ae.decoder
        Sequential(
          (0): Linear(in_features=30, out_features=20, bias=True)
          (1): Sigmoid()
          (2): Linear(in_features=20, out_features=10, bias=True)
          (3): Sigmoid()
        )
        """

        super().__init__()

        assert len(encoder_shape) > 2, "encoder_shape requires at least an input and output"

        # Encoder
        self.encoder = get_sequential(encoder_shape, backwards=False, activation=nn.Sigmoid)

        # Decoder
        self.decoder = get_sequential(encoder_shape, backwards=True, activation=nn.Sigmoid)

        # Model Name
        self.name = 'AutoEncoder'

        # Model Version (placeholder concept)
        self.version = 'V0.0'

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)

        return y
