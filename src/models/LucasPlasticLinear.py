"""
A class that provides a linear plastic layer.
"""
import torch
from torch.nn.parameter import Parameter
from torch import Tensor
import math
import torch.nn as nn
import torch.functional as F


class Mod(nn.Module):
    def __init__(self, size: int) -> None:
        super(Mod, self).__init__()
        self.size = size
        self.i2mod = torch.nn.Linear(size, 1)
        self.value = (torch.Tensor(1))

    def get_value(self):
        return self.value

    def update_value(self, input: Tensor):
        self.value = self.i2mod(input)


class PlasticLinear(nn.Linear):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Has a plastic hebian component.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(PlasticLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.alpha = Parameter(torch.Tensor(out_features, in_features))

        self.hebbian = Parameter(torch.zeros(self.in_features, self.out_features))  # same as the hebb_trace

        self.previous_output = Parameter(torch.Tensor(out_features))

        self.mod = None
        self.mod_fanout = nn.Linear(torch.Tensor(1, out_features))

        self.reset_parameters()

    def set_mod(self, mod: Mod):
        self.mod = mod

    def reset_parameters(self) -> None:
        super(PlasticLinear, self).reset_parameters()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.alpha = Parameter(torch.zeros(self.in_features, self.out_features))
        self.hebbian = Parameter(torch.zeros(self.in_features, self.out_features))

        self.previous_output = Parameter(torch.Tensor(self.out_features))
        self.mod_fanout = nn.Linear(torch.Tensor(1, self.out_features))

    def forward(self, input: Tensor) -> Tensor:
        """
        First optional arg is hstate and second is  hebb_trace. These are REQUIRED.
        """

        w = F.linear(input, self.weight, self.bias)
        y = w + self.previous_output.unsqueeze(1).bmm(self.weight + torch.mul(self.alpha, self.hebbian))

        delta_hebb = torch.bmm(self.previous_output.unsqueeze(2), y.unsqueeze(1))

        myeta = self.mod_fanout(self.mod.get_value())

        self.clip_val = 2.0
        self.hebbian = torch.clamp(self.hebbian + myeta * delta_hebb, min=-self.clip_val, max=self.clip_val)

        self.previous_output = y
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, alpha,={}, bias={}'.format(
            self.in_features, self.out_features, self.alpha, self.bias is not None
        )


class PlasticFeedforward(nn.Module):

    def __init__(self):
        super(PlasticFeedforward, self).__init__()
        self.p1 = PlasticLinear(784, 100)
        self.p2 = PlasticLinear(100, 10)
        self.p3 = PlasticLinear(10, 2)

        self.mod = Mod(2)  # we give it the size of the output of the layer we want to set the mod at
        # we set the mod of the layers
        self.p1.set_mod(self.mod)
        self.p2.set_mod(self.mod)
        self.p3.set_mod(self.mod)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        self.mod.update_value(x)  # we update the mod value
        return x
