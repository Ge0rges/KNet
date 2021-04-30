import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import RandomSampler, DataLoader, SubsetRandomSampler
import torch

# Pytorch dataloading variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")

root = './data'
if not os.path.exists(root):
    os.mkdir(root)


def mnist_loader(batch_size=256, num_workers=0, dims=1):
    def one_hot_mnist(targets):
        targets_onehot = torch.zeros(10)
        targets_onehot[targets] = 1
        return targets_onehot

    if dims == 3:
        transform_all = transforms.Compose([
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_all = transforms.Compose([
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda a: a.view(-1))
        ])

    train_set = datasets.MNIST(root=root, train=True, transform=transform_all, download=True,
                               target_transform=one_hot_mnist)
    test_set = datasets.MNIST(root=root, train=False, transform=transform_all, download=True,
                              target_transform=one_hot_mnist)

    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)

    index = int(len(test_set) * 0.2)
    test_indices = list(range(index, len(test_set)))
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    eval_indices = list(range(index))
    eval_sampler = SubsetRandomSampler(eval_indices)
    eval_loader = DataLoader(test_set, sampler=eval_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader, eval_loader

