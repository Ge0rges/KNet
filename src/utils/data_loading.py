import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import os

from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.preprocessing import normalize
from src.utils.misc import GaussianNoise, AESampler, one_hot, BananaCarImageDataset

DATA = '../data'

BANANA_RESIZED_DATA_TRAIN = '../data/Banana_Car/banana/1/resized/train/'
BANANA_RESIZED_DATA_TEST = '../data/Banana_Car/banana/1/resized/test/'
BANANA_RESIZED_DATA_VALID = '../data/Banana_Car/banana/1/resized/validation/'
BANANA_RESIZED_DATA = '../data/Banana_Car/banana/1/resized/1/'

CAR_RESIZED_DATA = '../data/Banana_Car/car/1/resized/1/'
BANANACAR_RESIZED_DATA = '../data/Banana_Car/bananacar/1/resized/1/'

BANANA_LABEL = 1
CAR_LABEL = 2
BANANACAR_LABEL = 3
ALL_BANANA_CAR_LABELS = [BANANA_LABEL, CAR_LABEL]


### Banana Cars
def bc_loader(dir, name, label, batch_size=256, num_workers=0):
    """Loader to be used only for the car, banana and bananacar datasets"""
    assert os.path.isdir(dir)

    dataset = BananaCarImageDataset(dir, name, label, ALL_BANANA_CAR_LABELS)

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


def all_bc_loader(dirs, names, labels, batch_size=256, num_workers=0):
    """Dirs must be of the shape: [[train, valid, test], [train,valid,test]]"""
    # first we do the train dataloader
    print("loading training set")
    datasets = []
    for i in range(len(dirs)):
        name = names[i]
        label = labels[i]
        assert os.path.isdir(dirs[i][0])

        d = BananaCarImageDataset(dirs[i][0], name, label, ALL_BANANA_CAR_LABELS)
        datasets.append(d)
    dataset = ConcatDataset(datasets)
    labels = [i[1] for i in dataset]

    trainsampler = AESampler(labels, start_from=0)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    print("loading validation set")
    # valid loader
    datasets = []
    for i in range(len(dirs)):
        name = names[i]
        label = labels[i]
        assert os.path.isdir(dirs[i][1])

        d = BananaCarImageDataset(dirs[i][1], name, label,ALL_BANANA_CAR_LABELS)
        datasets.append(d)
    dataset = ConcatDataset(datasets)
    labels = [i[1] for i in dataset]

    validsampler = AESampler(labels, start_from=0)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    print("loading test set")
    # test loader
    datasets = []
    for i in range(len(dirs)):
        name = names[i]
        label = labels[i]
        assert os.path.isdir(dirs[i][2])

        d = BananaCarImageDataset(dirs[i][2], name, label, ALL_BANANA_CAR_LABELS)
        datasets.append(d)
    dataset = ConcatDataset(datasets)
    labels = [i[1] for i in dataset]

    testsampler = AESampler(labels, start_from=0)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


###### EEG
def eeg_task_to_task_loader(batch_size=256, num_workers=0):
    # TODO: make is so that the testloader isn't re-using data that is being used in the train and valid loaders
    datasets = []
    for i in range(9):
        datasets.append(EEG_get_task_dataset(i))

    dataset = ConcatDataset(datasets)

    labels = [i[1] for i in dataset]

    testsampler = AESampler(labels, start_from=0)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing AE dataloader")

    return (testloader)


def EEG_get_task_dataset(task_num):
    task_data = np.load("./data/EEG_Processed/task{}_task.npy".format(task_num+1))
    num_samples = len(task_data)
    task_labels = [task_num]*num_samples
    task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='max', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(10))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    random_data = np.load("./data/EEG_Processed/task{}_random.npy".format(task_num+1))
    num_samples = len(random_data)
    random_labels = [9]*num_samples
    random_data = normalize(random_data.reshape((num_samples, 256*4)), norm='max', axis=0)
    random_data = torch.Tensor(random_data)

    random_tensor_labels = torch.Tensor(random_labels)
    random_tensor_class_labels = one_hot(random_tensor_labels, range(10))
    random_tensor_labels = torch.cat([random_data, random_tensor_class_labels], 1)

    random_dataset = TensorDataset(random_data, random_tensor_labels)

    dataset = ConcatDataset([task_dataset, random_dataset])
    return dataset


def EEG_task_loader(task_num, batch_size=256, num_workers=0):
    task_data = np.load("./data/EEG_Processed/task{}_task.npy".format(task_num+1))
    num_samples = len(task_data)
    task_labels = [task_num]*num_samples
    task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(10))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    random_data = np.load("./data/EEG_Processed/task{}_random.npy".format(task_num+1))
    num_samples = len(random_data)
    random_labels = [9]*num_samples
    random_data = normalize(random_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    random_data = torch.Tensor(random_data)

    random_tensor_labels = torch.Tensor(random_labels)
    random_tensor_class_labels = one_hot(random_tensor_labels, range(10))
    random_tensor_labels = torch.cat([random_data, random_tensor_class_labels], 1)

    random_dataset = TensorDataset(random_data, random_tensor_labels)

    dataset = ConcatDataset([task_dataset, random_dataset])

    num_samples = len(dataset)
    labels = [i[1] for i in dataset]

    train_size = int(num_samples*0.7)
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    valid_size = int(num_samples*0.05)
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)
    print("Done preparing AE dataloader")

    return (trainloader, validloader)


def EEG_raw_to_bands_loader(batch_size=256, num_workers=0):

    datasets = []

    normal_data = np.load("./data/EEG_Processed/normal.npy")
    normal_labels = np.load("./data/EEG_Processed/normal_band.npy")
    num_samples = len(normal_data)
    normal_data = normalize(normal_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets.append(normal_dataset)

    calm_data = np.load("./data/EEG_Processed/calm.npy")
    num_samples = len(calm_data)
    calm_labels = np.load("./data/EEG_Processed/calm_band.npy")
    calm_data = normalize(calm_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    calm_data = torch.Tensor(calm_data)

    calm_tensor_labels = torch.Tensor(calm_labels)
    calm_tensor_labels = torch.cat([calm_data, calm_tensor_labels], 1)

    calm_dataset = TensorDataset(calm_data, calm_tensor_labels)
    datasets.append(calm_dataset)

    traindata = []
    trainlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i))):
            traindata.append(i[j][0].numpy().astype(float))
            trainlabels.append(i[j][1].numpy().astype(float))
    traindata = torch.Tensor(traindata)
    trainlabels = torch.Tensor(trainlabels)
    traindataset = TensorDataset(traindata, trainlabels)

    validdata = []
    validlabels = []
    for i in datasets:
        for j in range(int(0.7 * len(i)), int(0.7*len(i)) + int(0.05*len(i))):
            validdata.append(i[j][0].numpy().astype(float))
            validlabels.append(i[j][1].numpy().astype(float))
    validdata = torch.Tensor(validdata)
    validlabels = torch.Tensor(validlabels)
    validdataset = TensorDataset(validdata, validlabels)

    testdata = []
    testlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i)) + int(0.05*len(i)), len(i)):
            testdata.append(i[j][0].numpy().astype(float))
            testlabels.append(i[j][1].numpy().astype(float))
    testdata = torch.Tensor(testdata)
    testlabels = torch.Tensor(testlabels)
    testdataset = TensorDataset(testdata, testlabels)

    train_labels = [i[1] for i in traindataset]
    valid_labels = [i[1] for i in validdataset]
    test_labels = [i[1] for i in testdataset]

    trainsampler = AESampler(train_labels, start_from=0)
    trainloader = DataLoader(traindataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = AESampler(valid_labels, start_from=0)
    validloader = DataLoader(validdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(test_labels, start_from=0)
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


def EEG_bands_to_binary_loader(batch_size=256, num_workers=0):
    task_data = np.load("./data/EEG_Processed/calm_band.npy")
    num_samples = len(task_data)
    task_labels = [1] * num_samples
    # task_data = normalize(task_data.reshape((num_samples, 256 * 4)), norm='l2', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(2))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    normal_data = np.load("./data/EEG_Processed/normal_band.npy")
    num_samples = len(normal_data)
    normal_labels = [0] * num_samples
    # random_data = normalize(normal_data.reshape((num_samples, 256 * 4)), norm='l2', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_class_labels = one_hot(normal_tensor_labels, range(2))
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_class_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets = [task_dataset, normal_dataset]

    traindata = []
    trainlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i))):
            traindata.append(i[j][0].numpy().astype(float))
            trainlabels.append(i[j][1].numpy().astype(float))
    traindata = torch.Tensor(traindata)
    trainlabels = torch.Tensor(trainlabels)
    traindataset = TensorDataset(traindata, trainlabels)

    validdata = []
    validlabels = []
    for i in datasets:
        for j in range(int(0.7 * len(i)), int(0.7*len(i)) + int(0.1*len(i))):
            validdata.append(i[j][0].numpy().astype(float))
            validlabels.append(i[j][1].numpy().astype(float))
    validdata = torch.Tensor(validdata)
    validlabels = torch.Tensor(validlabels)
    validdataset = TensorDataset(validdata, validlabels)

    testdata = []
    testlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i)) + int(0.1*len(i)), len(i)):
            testdata.append(i[j][0].numpy().astype(float))
            testlabels.append(i[j][1].numpy().astype(float))
    testdata = torch.Tensor(testdata)
    testlabels = torch.Tensor(testlabels)
    testdataset = TensorDataset(testdata, testlabels)

    train_labels = [i[1] for i in traindataset]
    valid_labels = [i[1] for i in validdataset]
    test_labels = [i[1] for i in testdataset]

    trainsampler = AESampler(train_labels, start_from=0)
    trainloader = DataLoader(traindataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = AESampler(valid_labels, start_from=0)
    validloader = DataLoader(validdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(test_labels, start_from=0)
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


def EEG_raw_to_binary_loader(batch_size=256, num_workers=0):

    def custom_EEG_dataset_getter(task_num, task_label, random_label):
        task_data = np.load("./data/EEG_Processed/task{}_task.npy".format(task_num+1))
        num_samples = len(task_data)
        task_labels = [task_label]*num_samples
        task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
        task_data = torch.Tensor(task_data)

        task_tensor_labels = torch.Tensor(task_labels)
        task_tensor_class_labels = one_hot(task_tensor_labels, range(2))
        task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

        task_dataset = TensorDataset(task_data, task_tensor_labels)

        random_data = np.load("./data/EEG_Processed/task{}_random.npy".format(task_num+1))
        num_samples = len(random_data)
        random_labels = [random_label]*num_samples
        random_data = normalize(random_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
        random_data = torch.Tensor(random_data)

        random_tensor_labels = torch.Tensor(random_labels)
        random_tensor_class_labels = one_hot(random_tensor_labels, range(2))
        random_tensor_labels = torch.cat([random_data, random_tensor_class_labels], 1)

        random_dataset = TensorDataset(random_data, random_tensor_labels)

        dataset = ConcatDataset([task_dataset, random_dataset])
        return dataset
    datasets = []
    datasets.append(custom_EEG_dataset_getter(0, 1, 0))


    normal_data = np.load("./data/EEG_Processed/normal.npy")
    num_samples = len(normal_data)
    normal_labels = [0]*num_samples
    normal_data = normalize(normal_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_class_labels = one_hot(normal_tensor_labels, range(2))
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_class_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets.append(normal_dataset)

    calm_data = np.load("./data/EEG_Processed/calm.npy")
    num_samples = len(calm_data)
    calm_labels = [1]*num_samples
    calm_data = normalize(calm_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    calm_data = torch.Tensor(calm_data)

    calm_tensor_labels = torch.Tensor(calm_labels)
    calm_tensor_class_labels = one_hot(calm_tensor_labels, range(2))
    calm_tensor_labels = torch.cat([calm_data, calm_tensor_class_labels], 1)

    calm_dataset = TensorDataset(calm_data, calm_tensor_labels)
    datasets.append(calm_dataset)

    traindata = []
    trainlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i))):
            traindata.append(i[j][0].numpy().astype(float))
            trainlabels.append(i[j][1].numpy().astype(float))
    traindata = torch.Tensor(traindata)
    trainlabels = torch.Tensor(trainlabels)
    traindataset = TensorDataset(traindata, trainlabels)

    validdata = []
    validlabels = []
    for i in datasets:
        for j in range(int(0.7 * len(i)), int(0.7*len(i)) + int(0.05*len(i))):
            validdata.append(i[j][0].numpy().astype(float))
            validlabels.append(i[j][1].numpy().astype(float))
    validdata = torch.Tensor(validdata)
    validlabels = torch.Tensor(validlabels)
    validdataset = TensorDataset(validdata, validlabels)

    testdata = []
    testlabels = []
    for i in datasets:
        for j in range(int(0.7*len(i)) + int(0.05*len(i)), len(i)):
            testdata.append(i[j][0].numpy().astype(float))
            testlabels.append(i[j][1].numpy().astype(float))
    testdata = torch.Tensor(testdata)
    testlabels = torch.Tensor(testlabels)
    testdataset = TensorDataset(testdata, testlabels)

    train_labels = [i[1] for i in traindataset]
    valid_labels = [i[1] for i in validdataset]
    test_labels = [i[1] for i in testdataset]

    trainsampler = AESampler(train_labels, start_from=0)
    trainloader = DataLoader(traindataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = AESampler(valid_labels, start_from=0)
    validloader = DataLoader(validdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(test_labels, start_from=0)
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return (trainloader, validloader, testloader)


##### MNIST
def mnist_loader(batch_size=256, num_workers=0):
    allset = get_mnist_dataset()

    data = torch.zeros((len(allset), 28*28))
    for i in range(0, len(allset)):
        data[i] = allset[i][0].view((28*28))

    class_labels = list(i[1] for i in allset)

    tensor_class_labels = torch.Tensor(class_labels)
    tensor_class_labels = one_hot(tensor_class_labels, range(10))

    tensor_labels = torch.cat([data, tensor_class_labels], 1)

    dataset = TensorDataset(data, tensor_labels)

    labels = [i[1] for i in dataset]

    trainsampler = AESampler(labels, start_from=0, amount=1000)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    validsampler = AESampler(labels, start_from=1000, amount=200)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=1200, amount=5000)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing AE dataloader")

    return (trainloader, validloader, testloader)


def get_mnist_dataset():
    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.2)
    ])

    trainset = dataloader(root=DATA, train=True, download=True, transform=transform_all)
    testset = dataloader(root=DATA, train=False, download=False, transform=transform_all)

    return ConcatDataset([trainset, testset])