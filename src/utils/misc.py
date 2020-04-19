
import errno
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt


def one_hot(targets, cl):
    targets_onehot = torch.zeros(targets.shape)
    for i, t in enumerate(targets):
        if int(targets[i][cl]) == 1:
            targets_onehot[i][cl] = 1
    return targets_onehot


def one_vs_all_one_hot(targets, cls, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t == cls:
            targets_onehot[i][cls] = 1
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


class BananaCarImageDataset(Dataset):
    def __init__(self, dir, name, label, all_labels):
        assert os.path.isdir(dir)
        assert name == "banana" or name == "car" or name == "bananacar"

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

            if self.name == "bananacar":
                label = torch.Tensor([0.5, 0.5])

            else:
                label = torch.Tensor([self.label])
                label = one_hot(label, self.all_labels).view((len(self.all_labels)))
            label = torch.cat([tensor_img, label], 0)

            sample = (tensor_img, label)

            return sample


class DataloaderWrapper(object):
    """Wraps the Dataloader class to increase its utility in various situations"""

    def __init__(self, dataloader_manager,  dataloaderfn, task, category, batch_size=256, num_workers=0):
        self.dataloader_manager = dataloader_manager
        self.id = None
        self.dataloader = dataloaderfn
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers

        if category == "train":
            self.loader_index = 0
        elif category == "valid":
            self.loader_index = 1
        elif category == "test":
            self.loader_index = 2
        else:
            raise NameError
        self.length = None
        self.id = self.dataloader_manager.register(self)

    def get_loader(self):
        return self.dataloader(self.task, batch_size=self.batch_size, num_workers=self.num_workers)[self.loader_index]

    def get_id(self):
        return self.id

    def __len__(self):
        if self.length is None:
            self.length = len(self.dataloader_manager.get_loader(self))
        return self.length

    def __iter__(self):
        if self.length is None:
            dataloader = self.dataloader_manager.get_loader(self)
            self.length = len(dataloader)
            return iter(dataloader)
        else:
            return iter(self.dataloader_manager.get_loader(self))


class DataloaderManager(object):
    """Manages multiple dataloader wrappers and allocates memory for loaders"""
    def __init__(self):
        self.ids = []
        self.current_loader = None
        self.current_loader_id = None

    def register(self, dataloader_wrapper):
        id = dataloader_wrapper.get_id()
        if id is not None and id in self.ids:
            print("This dataloader_wrapper was already registered with id {}".format(id))
            return id
        else:
            num_ids = len(self.ids)
            new_id = num_ids
            self.ids.append(new_id)

            return new_id

    def get_loader(self, dataloader_wrapper):
        id = dataloader_wrapper.get_id()
        if self.current_loader_id == id:
            return self.current_loader
        else:
            self.current_loader_id = id
            self.current_loader = dataloader_wrapper.get_loader()
            return self.current_loader


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