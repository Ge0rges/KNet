"""
The PyTorch module for a simple autoencoder.
"""
import torch.nn as nn
import torch

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
        self.encoder = self.get_sequential(encoder_shape, backwards=False)

        # Decoder
        self.decoder = self.get_sequential(encoder_shape, backwards=True)

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)

        return y

    def get_sequential(self, encoder_shape: [int], backwards: bool = False):
        """
        Returns an nn.Sequential object made from nn.Linear and nn.Sigmoid. One layer for each size
        defined in encoder_shape. Set backwards to True to get decoder shape.

        :param encoder_shape: A list of ints representing the size of each layer in the encoder.
        :param backwards: Whether to init the layers backwards (first layer size is last shape in encoder_shape)
        :return: nn.Sequential of nn.Linear and nn.Sigmoid objects, one for each in encoder_shape.

        >>> ae = AutoEncoder([])
        >>> ae.get_sequential([10, 20, 30])
        Sequential(
          (0): Linear(in_features=10, out_features=20, bias=True)
          (1): Sigmoid()
          (2): Linear(in_features=20, out_features=30, bias=True)
          (3): Sigmoid()
        )

        >>> ae.get_sequential([10, 20, 30], True)
        Sequential(
          (0): Linear(in_features=30, out_features=20, bias=True)
          (1): Sigmoid()
          (2): Linear(in_features=20, out_features=10, bias=True)
          (3): Sigmoid()
        )
        """

        assert len(encoder_shape) > 2, "encoder_shape requires at least an input and output"

        layers = []
        indices = reversed(range(0, len(encoder_shape) - 1)) if backwards else range(0, len(encoder_shape) - 1)

        for i in indices:
            # nn.Linear is He/Kaiming uniform initialized.
            if backwards:
                layers.append(nn.Linear(in_features=encoder_shape[i+1], out_features=encoder_shape[i]))
            else:
                layers.append(nn.Linear(in_features=encoder_shape[i], out_features=encoder_shape[i + 1]))

            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
