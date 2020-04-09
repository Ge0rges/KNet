import errno
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


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


class BananaCarImageDataset(Dataset):
    def __init__(self, dir, name, label, all_labels):
        self.dir = dir
        self.num_files = 0
        self.name = name
        self.label = label
        self.all_labels = all_labels
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for file in filenames:
                if file.endswith(".jpg"):
                    self.num_files += 1

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, slice):
            idx = list(range(idx.stop)[idx])

        try:
            samples = []
            for i in idx:
                img = np.asarray(Image.open(self.dir + self.name + "_{}.jpg".format(i)))
                tensor_img = torch.Tensor(img)

                new_size = 1
                for j in tensor_img.size():
                    new_size *= j
                tensor_img = tensor_img.view((new_size))
                if self.name == "banana_car":
                    label = torch.Tensor([0.5, 0.5])

                else:
                    label = torch.Tensor([self.label])
                    label = one_hot(label, self.all_labels).view((len(self.all_labels)))
                label = torch.cat([tensor_img, label], 0)

                sample = (tensor_img, label)
                samples.append(sample)

            return samples

        except TypeError:
            img = np.asarray(Image.open(self.dir + self.name + "_{}.jpg".format(idx)))
            tensor_img = torch.Tensor(img)
            new_size = 1
            for j in tensor_img.size():
                new_size *= j
            tensor_img = tensor_img.view((new_size))

            if self.name == "banana_car":
                label = torch.Tensor([0.5, 0.5])

            else:
                label = torch.Tensor([self.label])
                label = one_hot(label, self.all_labels).view((len(self.all_labels)))
            label = torch.cat([tensor_img, label], 0)

            sample = (tensor_img, label)

            return sample


class DataloaderWrapper(object):
    """Wraps the Dataloader class to increase its utility in various situations"""

    def __init__(self, dataloaderfn, args=None, batch_size=256, num_workers=0):
        self.dataloader = dataloaderfn
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_loaders(self, batch_size=None, num_workers=None):
        if batch_size:
            self.batch_size = batch_size
        if num_workers:
            self.num_workers = num_workers
        if self.args:
            return self.dataloader(self.args, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            return self.dataloader(batch_size=self.batch_size, num_workers=self.num_workers)


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
        print("indices", self.indices)

    def __iter__(self):
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

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3