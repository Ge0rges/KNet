"""
A class that provides a linear plastic layer.
"""
import torch
from torch.nn.parameter import Parameter
from torch import Tensor

import torch.nn as nn
import torch.functional as F


class PlasticLinear(nn.Linear):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Has a plastic hebian component.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(PlasticLinear, self).__init__(in_features, out_features, bias)

        self.alpha = Parameter(Tensor(out_features, in_features))
        self.hebbian = Parameter(Tensor(out_features, in_features))

    def reset_parameters(self) -> None:
        super(PlasticLinear, self).reset_parameters()

        self.alpha = Parameter(torch.zeros(self.in_features, self.out_features))
        self.hebbian = Parameter(torch.zeros(self.in_features, self.out_features))

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        """
        First optional arg is hstate and second is  hebb_trace. These are REQUIRED.
        """
        assert False, "This is not implemented."
        hstate = args[0]
        hebb_trace = args[1]

        w = F.linear(input, self.weight, self.bias)
        y = w + hstate.unsqueeze(1).bmm(self.hebbian + torch.mul(self.alpha, hebb_trace))

        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, alpha,={}, bias={}'.format(
            self.in_features, self.out_features, self.alpha, self.bias is not None
        )
