import errno
import os

import torch
from torch.utils.data.sampler import Sampler

from PIL import ImageFilter

__all__ = ['mkdir_p', 'AverageMeter', 'ClassSampler', 'GaussianNoise', 'fft_psd', 'one_hot']

def one_hot(targets, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in classes:
            targets_onehot[i][classes.index(t)] = 1
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

class ClassSampler(Sampler):

    def __init__(self, labels, classes, start_from = 0, amount = None):
        self.indices = []
        start = [start_from] * len(classes)
        left = [amount] * len(classes)

        for i, label in enumerate(labels):
            if label in classes:
                idx = classes.index(label)

                if start[idx] == 0:
                    if left[idx] is None:
                        self.indices.append(i)
                    elif left[idx] > 0:
                        self.indices.append(i)
                        left[idx] -= 1
                else: 
                    start[idx] -= 1

    def __iter__(self):
        #return (i for i in range(self.prefix))
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class AESampler(Sampler):

    def __init__(self, dataset, start_from=0, amount=None):
        self.indices = []
        start = start_from
        left = amount
        for i, sample in enumerate(dataset):
            if start == 0:
                if left is None:
                    self.indices.append(i)
                elif left > 0:
                    self.indices.append(i)
                    left -= 1
            else:
                start -= 1

    def __iter__(self):
        #return (i for i in range(self.prefix))
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class GaussianNoise(object):

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, img):
        noise = img.clone()
        noise = noise.normal_(self.mean, self.stddev)
        new_img = img + noise
        new_img = torch.clamp(new_img, 0, 1)
        return new_img