"""
A collection of short functions used in various places.
For a function to exist here it must satisfy the following requirements:
- Fully encapsulated from any class.
- Is used in more than 1 file.
- Does not fit in another file.
- Has one straightforward function, easily understandable.
- Less than 60 lines.

This is not a dumping ground for random, obsolete functions.
"""
import torch.nn as nn
from typing import Type


def get_sequential(encoder_shape: [int], backwards: bool = False, activation: Type[nn.Module] = nn.Sigmoid) -> nn.Sequential:
    """
    Returns an nn.Sequential object made from nn.Linear and nn.Sigmoid. One layer for each size
    defined in encoder_shape. Set backwards to True to get decoder shape.

    :param encoder_shape: A list of ints representing the size of each layer in the encoder.
    :param backwards: Whether to init the layers backwards (first layer size is last shape in encoder_shape)
    :param activation: The activation function after each layer. Must not be instantiated.
    :rtype: nn.Sequential()
    :return: nn.Sequential of nn.Linear and nn.Sigmoid objects, one for each in encoder_shape.

    >>> get_sequential([10, 20, 30])
    Sequential(
      (0): Linear(in_features=10, out_features=20, bias=True)
      (1): Sigmoid()
      (2): Linear(in_features=20, out_features=30, bias=True)
      (3): Sigmoid()
    )

    >>> get_sequential([10, 20, 30], True)
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

        layers.append(activation())

    return nn.Sequential(*layers)
