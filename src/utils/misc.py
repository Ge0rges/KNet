import errno
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


def plot_tensor(tensor, img_size, mode=None):
    assert np.prod(tensor.size()) == np.prod(img_size)
    if mode:
        if mode == "RGB":
            data = tensor.numpy()
            data = data.reshape(img_size).astype(np.uint8)

            plt.imshow(data, interpolation="nearest")
    else:
        data = tensor.numpy()
        data = data.reshape(img_size)
        imgplot = plt.imshow(data, interpolation="nearest")
        imgplot.set_cmap('gray')

    plt.show()


def get_modules(model: torch.nn.Module) -> dict:
    modules = {}

    for name, param in model.named_parameters():
        module = name[0: name.index('.')]

        if module not in modules.keys():
            modules[module] = []

        modules[module].append((name, param))

    return modules


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super(ModuleWrapperIgnores2ndArg, self).__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x
