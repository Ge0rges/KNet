import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from train import one_hot
import numpy as np
import os
import torchvision
from PIL import Image

from misc import ClassSampler, GaussianNoise, AESampler

__all__ = ['load_MNIST', 'load_CIFAR']

DATA = './data'
AE_DATA = './data/AE'
# only contains the data since the labels are in fact just the data points themselves for the autoencodeur
AE_FILE = AE_DATA + '/data.csv.npy'
ALL_CLASSES = range(10)

MAX_FILE_SIZE = 48000000
BANANA_PROCESSED_DATA = './data/banana/processed'
BANANA_LABEL = 1
CAR_LABEL = 2
ALL_CUSTOM_LABELS = [BANANA_LABEL, CAR_LABEL]


def dataset_setup(data_path, save_path, name):
    if not os.path.isdir(data_path):
        print("not dir")
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor(),
    )
    cur = 0
    f_count = 0
    l = len(train_dataset)

    in_size = 1
    for i in train_dataset[0][0].size():
        in_size *= i

    MAX = int((MAX_FILE_SIZE - 100000)/(in_size*8))  # 100000 is for the overhead
    while cur < l:
        data = []
        count = 0
        threshold = min(MAX, l - cur)
        while count < threshold:
            data.append(train_dataset[cur + count][0].numpy().astype(np.float64))
            count += 1
        print("SAVING BATCH {} OF {} SAMPLES".format(f_count, threshold))
        np.save("{}/{}_{}".format(save_path, name, f_count), data)
        cur += count
        f_count += 1


def car_reshaping(new_size=(480, 640), colors=3):
    filepath = './data/car/raw/'
    files = []
    for (dirpath, dirnames, filenames) in os.walk(filepath):
        files.extend(filenames)
    count = 0
    for f in files:
        img = Image.open(f)
        new_img = img.resize(new_size)
        new_img.save('./data/car/processed/car_{}.jpg'.format(count))
        count += 1

def car_loader():
    pass


def banana_loader(batch_size=256, num_workers=4):
    data = []
    count = 0
    filepath = BANANA_PROCESSED_DATA + '/banana_{}.npy'.format(count)
    while os.path.isfile(filepath):
        print(filepath)
        partial = np.load(filepath)
        data.extend(partial)
        count += 1
        filepath = BANANA_PROCESSED_DATA + '/banana_{}.npy'.format(count)

    labels = [BANANA_LABEL]*len(data)

    data = torch.Tensor(data)
    print(data.size())
    labels = torch.Tensor(labels)
    print(labels.size())
    # labels = one_hot(labels, ALL_CUSTOM_LABELS)
    # print(labels.size())

    trainsampler = ClassSampler(labels, range(len(ALL_CUSTOM_LABELS)), start_from=0, amount=800)
    trainloader = DataLoader(data, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = ClassSampler(labels, range(len(ALL_CUSTOM_LABELS)), start_from=800, amount=197)
    validloader = DataLoader(data, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = ClassSampler(labels, range(len(ALL_CUSTOM_LABELS)), start_from=997, amount=300)
    testloader = DataLoader(data, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


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
    dataset_setup('./data/banana/compiled', BANANA_PROCESSED_DATA, 'banana')
    # banana_loader()
