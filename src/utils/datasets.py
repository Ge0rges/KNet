import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from train import one_hot
import numpy as np
import os
import torchvision
from PIL import Image
import re

from misc import ClassSampler, GaussianNoise, AESampler

__all__ = ['load_MNIST', 'load_CIFAR']

DATA = './data'
AE_DATA = './data/AE'
# only contains the data since the labels are in fact just the data points themselves for the autoencodeur
AE_FILE = AE_DATA + '/data.csv.npy'
ALL_CLASSES = range(10)

MAX_FILE_SIZE = 5000000000

BANANA_PROCESSED_DATA = './data/banana/processed'
CAR_RESIZED_DATA = './data/car/resized'
BANANA_LABEL = 1
CAR_LABEL = 2
ALL_CUSTOM_LABELS = [BANANA_LABEL, CAR_LABEL]


def car_reshaping(new_size=(640, 480), colors=3):
    filepath = './data/car/raw/'
    files = []

    for (dirpath, dirnames, filenames) in os.walk(filepath):
        print((dirpath, dirnames, filenames))
        for file in filenames:
            if file.endswith(".jpg"):
                files.append(dirpath + "/" + file)
        # files.extend(filenames)

    count = 0
    for f in files:
        img = Image.open(f)
        new_img = img.resize(new_size)
        new_img.save("./data/car/resized/1/car_{}.jpg".format(count))
        print(count)
        count += 1


def car_loader(batch_size=256, num_workers=4):
    if not os.path.isdir(CAR_RESIZED_DATA):
        print("not dir")
    dataset = torchvision.datasets.ImageFolder(
        root=CAR_RESIZED_DATA,
        transform=torchvision.transforms.ToTensor(),
    )
    data = list(i[0].numpy().astype(np.float64) for i in dataset)
    num_samples = len(data)
    class_labels = [CAR_LABEL] * num_samples
    #
    # data = torch.Tensor(data)
    # print(data.size())
    # labels = torch.Tensor(labels)
    # print(labels.size())
    # labels = one_hot(labels, ALL_CUSTOM_LABELS)
    # print(labels.size())
    tensor_data = torch.Tensor(data)
    # tensor_data = tensor_data.view(-1, 28 * 28)

    tensor_labels = torch.Tensor(data)
    # tensor_labels = tensor_labels.view(-1, 28*28)

    tensor_class_labels = torch.Tensor(class_labels)
    tensor_class_labels = one_hot(tensor_class_labels, ALL_CUSTOM_LABELS)

    tensor_labels = torch.cat([tensor_labels, tensor_class_labels], 1)

    dataset = TensorDataset(tensor_data, tensor_labels)
    labels = [i[1] for i in dataset]

    train_size = int(num_samples * 0.7)
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    valid_size = int(num_samples * 0.05)
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing banana AE dataloader")

    return (trainloader, validloader, testloader)


def banana_loader(batch_size=256, num_workers=4):

    if not os.path.isdir(BANANA_PROCESSED_DATA):
        print("not dir")
    dataset = torchvision.datasets.ImageFolder(
        root=BANANA_PROCESSED_DATA,
        transform=torchvision.transforms.ToTensor(),
    )

    data = list(i[0].numpy().astype(np.float64) for i in dataset)
    num_samples = len(data)
    class_labels = [BANANA_LABEL]*num_samples

    # labels = torch.Tensor(class_labels)
    # print(labels.size())
    # labels = one_hot(labels, ALL_CUSTOM_LABELS)
    # print(labels.size())

    tensor_data = torch.Tensor(data)
    # tensor_data = tensor_data.view(-1, 28 * 28)

    tensor_labels = torch.Tensor(data)
    # tensor_labels = tensor_labels.view(-1, 28*28)

    tensor_class_labels = torch.Tensor(class_labels)
    tensor_class_labels = one_hot(tensor_class_labels, ALL_CUSTOM_LABELS)

    tensor_labels = torch.cat([tensor_labels, tensor_class_labels], 1)

    dataset = TensorDataset(tensor_data, tensor_labels)

    labels = [i[1] for i in dataset]

    train_size = int(num_samples*0.7)
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    valid_size = int(num_samples*0.05)
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing banana AE dataloader")

    return (trainloader, validloader, testloader)


def banana_and_car_loader(batch_size=256, num_workers=4):
    if not os.path.isdir(BANANA_PROCESSED_DATA):
        print("not dir")
    banana_dataset = torchvision.datasets.ImageFolder(
        root=BANANA_PROCESSED_DATA,
        transform=torchvision.transforms.ToTensor(),
    )

    if not os.path.isdir(CAR_RESIZED_DATA):
        print("not dir")
    car_dataset = torchvision.datasets.ImageFolder(
        root=CAR_RESIZED_DATA,
        transform=torchvision.transforms.ToTensor(),
    )
    banana_data = [banana[0].numpy().astype(np.float64) for banana in banana_dataset]
    car_data = []


def load_AE_MNIST(batch_size=256, num_workers=4):
    # if not os.path.isfile(AE_FILE):
    #     if not os.path.isdir(AE_DATA):
    #         os.makedirs(AE_DATA)
    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.2)
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_all)
    testset = dataloader(root=DATA, train=False, download=False, transform=transform_all)

    allset = ConcatDataset([trainset, testset])
    data = list(i[0].numpy().astype(np.float64) for i in allset)
    # np.save(AE_FILE, data)
    # print("SAVED DATA")
        # now that we saved the data, reconstruct the data set
    # else:
    #     print("FETCHING SAVED DATA")
    #     data = np.load(AE_FILE)
    #     print("LOADED SAVED DATA")
    class_labels = list(i[1] for i in allset)

    tensor_data = torch.Tensor(data)
    tensor_data = tensor_data.view(-1, 28 * 28)

    tensor_labels = torch.Tensor(data)
    tensor_labels = tensor_labels.view(-1, 28*28)

    tensor_class_labels = torch.Tensor(class_labels)
    tensor_class_labels = one_hot(tensor_class_labels, ALL_CLASSES)

    tensor_labels = torch.cat([tensor_labels, tensor_class_labels], 1)

    dataset = TensorDataset(tensor_data, tensor_labels)

    labels = [i[1] for i in dataset]
    trainsampler = AESampler(labels, start_from=0, amount=1000)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = AESampler(labels, start_from=1000, amount=200)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=1200, amount=5000)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing AE dataloader")

    return (trainloader, validloader, testloader)


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
    # car_reshaping()
    car_loader()