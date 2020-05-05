import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import os

from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler


class DatasetType:
    train = 0
    eval = 1
    test = 2

##### MNIST
def mnist_loader(type, batch_size=256, num_workers=0, pin_memory=False):
    def one_hot_mnist(targets):
        targets_onehot = torch.zeros(10)
        targets_onehot[targets] = 1
        return targets_onehot

    dataset = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    is_train = (True if type == DatasetType.train else False)

    root = os.path.join(os.path.dirname(__file__), "../../data/MNIST")
    assert os.path.isdir(root)

    dataset = dataset(root=root, train=is_train, download=True, transform=transform_all, target_transform=one_hot_mnist)

    if is_train:
        sampler = RandomSampler(dataset)

    else:
        index = int(len(dataset) * 0.2) if (type == DatasetType.eval) else int(len(dataset) * 0.8)
        indices = list(range(index)) if (type == DatasetType.eval) else list(range(index, len(dataset)))
        sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


##### BANANA_CAR
def banana_car_loader(dataset_type, size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):

    def one_hot_bc(targets):
        targets_onehot = torch.zeros(2)
        targets_onehot[targets] = 1
        return targets_onehot

    transform_all = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    root = os.path.join(os.path.dirname(__file__), "../../data/banana_car")
    assert os.path.isdir(root)

    dataset = datasets.ImageFolder(root=root, transform=transform_all, target_transform=one_hot_bc)

    indices = np.array(list(range(len(dataset))))
    np.random.shuffle(indices)

    if dataset_type == DatasetType.train:
        indices = indices[:int(len(dataset)*0.7)]

    elif dataset_type == DatasetType.eval:
        indices = indices[int(len(dataset)*0.7):int(len(dataset)*0.8)]

    elif dataset_type == DatasetType.test:
        indices = indices[int(len(dataset)*0.8):]

    else:
        raise ReferenceError

    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def bananacar_abstract_loader(size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):
    """Loader for the images containing cars with banana shapes"""

    def one_hot_one_one(targets):
        return torch.FloatTensor([1, 1])

    transform_all = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    path = "../data/abstraction_eval_bananacar"
    assert os.path.isdir(path)

    dataset = datasets.ImageFolder(root=path, transform=transform_all,
                                   target_transform=one_hot_one_one)

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


