"""
Our DEN function because we need to do stuff like selective retraining.
"""
from torch.autograd import Function


class DENFunction(Function):
    """
    Our custom Function class for DEN.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass
