import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from train import one_hot
import numpy as np
import os
import csv
from PIL import Image

from misc import ClassSampler, GaussianNoise, AESampler

__all__ = ['load_MNIST', 'load_CIFAR', 'load_AE_MNIST']

DATA = './data'

BANANA_RESIZED_DATA = '../data/Banana_Car/banana/1/resized/1/'
CAR_RESIZED_DATA = '../data/Banana_Car/car/1/resized/1/'
BANANACAR_RESIZED_DATA = '../data/Banana_Car/bananacar/1/resized/1/'

BANANA_LABEL = 0
CAR_LABEL = 1
BANANACAR_LABEL = 2
ALL_CUSTOM_LABELS = [BANANA_LABEL, CAR_LABEL, BANANACAR_LABEL]


def dataset_reshaping(name, directory_path, new_size=(640, 480), colors=3):
    files = []

    for (dirpath, dirnames, filenames) in os.walk(directory_path):
        print((dirpath, dirnames, filenames))
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".jfif"):
                files.append(dirpath + "/" + file)

    count = 0
    for f in files:
        img = Image.open(f)
        new_img = img.resize(new_size)
        new_img.save(directory_path + "resized/1/{}_{}.jpg".format(name, count))
        print(count)
        count += 1


class MyImageDataset(Dataset):
    def __init__(self, dir, name, label):
        self.dir = dir
        self.num_files = 0
        self.name = name
        self.label = label
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

                label = torch.Tensor([self.label])
                label = one_hot(label, ALL_CUSTOM_LABELS).view((len(ALL_CUSTOM_LABELS)))
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

            label = torch.Tensor([self.label])
            label = one_hot(label, ALL_CUSTOM_LABELS).view((len(ALL_CUSTOM_LABELS)))
            label = torch.cat([tensor_img, label], 0)

            sample = (tensor_img, label)

            return sample


def bc_loader(dir, name, label, batch_size=256, num_workers=4):
    """Loader to be used only for the car, banana and bananacar datasets"""
    if not os.path.isdir(dir):
        print("not dir")
    dataset = MyImageDataset(dir, name, label)

    num_samples = len(dataset)
    labels = [label]*num_samples

    train_size = int(num_samples*0.7)
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    valid_size = int(num_samples*0.05)
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing {} AE dataloader".format(name))

    return (trainloader, validloader, testloader)


def EEG_preprocessing(task, batch_size=256, num_workers=4):
    files = []
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/{}/".format(task)):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + "/" + file)
    data = []
    sample_n = 256
    for f in files:
        x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
        for i in range(np.shape(x)[0] - sample_n):
            pre_pro = x[i : i + sample_n]
            pre_pro = np.delete(pre_pro, 0, 1)
            pre_pro = np.delete(pre_pro, -1, 1)
            pro = _fft_psd(1, sample_n, pre_pro)
            assert np.shape(pro[1]) == (sample_n, 4)
            data.append(pro[1])

    np.save("../data/EEG_Processed/{}".format(task), data)


def _fft_psd(sampling_time, sample_num, data):
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


def EEG_loader(task_num, batch_size=256, num_workers=4):
    data = np.load("./data/EEG_Processed/task{}.npy".format(task_num))
    num_samples = len(data)
    labels = [task_num - 1]*num_samples

    data = torch.Tensor(data).view((num_samples, 256*4))

    tensor_labels = torch.Tensor(labels)
    tensor_class_labels = one_hot(tensor_labels, range(9))
    tensor_labels = torch.cat([data, tensor_class_labels], 1)

    dataset = TensorDataset(data, tensor_labels)

    labels = [i[1] for i in dataset]

    train_size = int(num_samples*0.7)
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)
    check_for_nan(trainloader)

    valid_size = int(num_samples*0.05)
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)
    check_for_nan(validloader)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)
    check_for_nan(testloader)

    print("Done preparing EEG task{} AE dataloader".format(task_num))

    return (trainloader, validloader, testloader)


def load_AE_MNIST(batch_size=256, num_workers=4):

    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.2)
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_all)
    testset = dataloader(root=DATA, train=False, download=False, transform=transform_all)

    allset = ConcatDataset([trainset, testset])

    data = torch.zeros((len(allset), 28*28))
    for i in range(0, len(allset)):
        data[i] = allset[i][0].view((28*28))

    class_labels = list(i[1] for i in allset)

    tensor_class_labels = torch.Tensor(class_labels)
    tensor_class_labels = one_hot(tensor_class_labels, range(10))

    tensor_labels = torch.cat([data, tensor_class_labels], 1)

    dataset = TensorDataset(data, tensor_labels)

    check_for_nan(dataset)

    labels = [i[1] for i in dataset]
    trainsampler = AESampler(labels, start_from=0, amount=1000)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)
    check_for_nan(trainloader)

    validsampler = AESampler(labels, start_from=1000, amount=200)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)
    check_for_nan(validloader)

    testsampler = AESampler(labels, start_from=1200, amount=5000)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)
    check_for_nan(testloader)

    print("Done preparing AE dataloader")

    return (trainloader, validloader, testloader)


def check_for_nan(dataset):
    print("checking for nan input")
    found = 0
    for idx, sample in enumerate(dataset):
        if sample != sample:
            found += 1
            print("sample #{} contains nan".format(idx))
    if found > 0:
        print("found a total of {} tensors with nan".format(found))


def load_MNIST(batch_size = 256, num_workers = 4):

    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.2)
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_all)
    testset = dataloader(root=DATA, train=False, download=False, transform=transform_all)

    allset = ConcatDataset([trainset, testset])
    labels = list(i[1] for i in allset)

    trainsampler = ClassSampler(labels, range(10), amount=1000)
    trainloader = DataLoader(allset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = ClassSampler(labels, range(10), amount=200, start_from=1000)
    validloader = DataLoader(allset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = ClassSampler(labels, range(10), amount=5000, start_from=1200)
    testloader = DataLoader(allset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


def load_CIFAR(batch_size = 256, num_workers = 4):

    dataloader = datasets.CIFAR10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = dataloader(root=DATA, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (trainloader, testloader)


if __name__ == '__main__':
    pass