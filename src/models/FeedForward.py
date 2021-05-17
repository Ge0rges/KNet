"""
A standard implementation of a feedforward neural network.
"""

import torch.nn as nn
from src.utils.misc import get_sequential


class FeedForward(nn.Module):
    """
    A standard feedforward classifier.
    """
    def __init__(self, network_shape: [int]):
        """
        Initializes the FeedForward netowkr with specified shape.

        :param network_shape: A list of ints representing the size of each layer in the classifier.

        >>> ff = FeedForward([10, 20, 30])
        >>> ff.classifier
        Sequential(
         (0): Linear(in_features=10, out_features=20, bias=True)
         (1): Sigmoid()
         (2): Linear(in_features=20, out_features=30, bias=True)
         (3): Sigmoid()
        )
        """
        super(FeedForward, self).__init__()

        self.classifier = get_sequential(network_shape, backwards=False, activation=nn.ReLU)

        self.name = 'FeedForward'
        self.version = 'V0.0'

    def forward(self, x):
        x = self.classifier(x)
        return x
