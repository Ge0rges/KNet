import errno
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn


def one_hot(targets, cl):
    targets_onehot = torch.zeros(targets.shape)
    for i, t in enumerate(targets):
        if int(targets[i][cl]) == 1:
            targets_onehot[i][cl] = 1
    return targets_onehot


def one_vs_all_one_hot(targets, tasks, classes):
    targets = targets.view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in tasks:
            targets_onehot[i][t] = 1

    return targets_onehot


def fft_psd(sampling_time, sample_num, data):
    """
    Get the the FFT power spectral densities for the given data
    Args:
        data (np.array): A single numpy array of data to process.

    Returns:
        list, list: FFT frequencies, FFT power spectral densities at those frequencies.
    """
    ps_densities = np.abs(np.fft.fft(data)) ** 2
    frequencies = np.fft.fftfreq(sample_num, float(sampling_time)/float(sample_num))
    idx = np.argsort(frequencies)
    return frequencies[idx], ps_densities[idx]


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def plot_tensor(tensor, img_format, mode=None):
    assert np.prod(tensor.size()) == np.prod(img_format)
    if mode:
        if mode == "RGB":
            data = tensor.numpy()
            data = data.reshape(img_format).astype(np.uint8)

            plt.imshow(data, interpolation="nearest")
    else:
        data = tensor.numpy()
        data = data.reshape(img_format)
        imgplot = plt.imshow(data, interpolation="nearest")
        imgplot.set_cmap('gray')

    plt.show()


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


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super(ModuleWrapperIgnores2ndArg, self).__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x
