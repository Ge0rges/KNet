import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import os

from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.preprocessing import normalize
from src.utils.misc import GaussianNoise, AESampler, one_hot, BananaCarImageDataset


### Banana Cars
BANANA_LABEL = 0
CAR_LABEL = 1
ALL_BANANA_CAR_LABELS = [BANANA_LABEL, CAR_LABEL]


def bc_loader(args, batch_size=256, num_workers=0):
    """Loader to be used only for the car, banana and banana_car datasets"""
    assert len(args) == 3
    dir = args[0]
    name = args[1]
    label = args[2]
    assert os.path.isdir(dir)
    if name == "banana_car":
        proportions = [0, 0, 1]
    else:
        proportions = [0.7, 0.5, 0.25]
    dataset = BananaCarImageDataset(dir, name, label, ALL_BANANA_CAR_LABELS)

    num_samples = len(dataset)
    if name == "banana_car":
        labels = [[0.5, 0.5]]*num_samples
    else:
        labels = [label]*num_samples

    train_size = int(num_samples*proportions[0])
    trainsampler = AESampler(labels, start_from=0, amount=train_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    valid_size = int(num_samples*proportions[1])
    validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(labels, start_from=(train_size + valid_size))
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing {} AE dataloader".format(name))

    return (trainloader, validloader, testloader)


def all_bc_loader(args, batch_size=256, num_workers=0):
    """Dirs must be of the shape: [[train, valid, test], [train,valid,test]]"""
    # first we do the train dataloader
    assert len(args) == 3
    dirs = args[0]
    names = args[1]
    labels = args[2]
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
    task_data = np.load("../data/EEG_Processed/task{}_task.npy".format(task_num+1))
    num_samples = len(task_data)
    task_labels = [task_num]*num_samples
    task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='max', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(10))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    random_data = np.load("../data/EEG_Processed/task{}_random.npy".format(task_num+1))
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


def EEG_task_loader(args, batch_size=256, num_workers=0):
    assert len(args) == 1
    task_num = args[0]

    task_data = np.load("../data/EEG_Processed/task{}_task.npy".format(task_num+1))
    num_samples = len(task_data)
    task_labels = [task_num]*num_samples
    task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(10))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    random_data = np.load("../data/EEG_Processed/task{}_random.npy".format(task_num+1))
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

    normal_data = np.load("../data/EEG_Processed/normal.npy")
    normal_labels = np.load("../data/EEG_Processed/normal_band.npy")
    num_samples = len(normal_data)
    normal_data = normalize(normal_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets.append(normal_dataset)

    calm_data = np.load("../data/EEG_Processed/calm.npy")
    num_samples = len(calm_data)
    calm_labels = np.load("../data/EEG_Processed/calm_band.npy")
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
    task_data = np.load("../data/EEG_Processed/calm_band.npy")
    num_samples = len(task_data)
    task_labels = [1] * num_samples
    # task_data = normalize(task_data.reshape((num_samples, 256 * 4)), norm='l2', axis=0)
    task_data = torch.Tensor(task_data)

    task_tensor_labels = torch.Tensor(task_labels)
    task_tensor_class_labels = one_hot(task_tensor_labels, range(2))
    task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

    task_dataset = TensorDataset(task_data, task_tensor_labels)

    normal_data = np.load("../data/EEG_Processed/normal_band.npy")
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
        task_data = np.load("../data/EEG_Processed/task{}_task.npy".format(task_num+1))
        num_samples = len(task_data)
        task_labels = [task_label]*num_samples
        task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
        task_data = torch.Tensor(task_data)

        task_tensor_labels = torch.Tensor(task_labels)
        task_tensor_class_labels = one_hot(task_tensor_labels, range(2))
        task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

        task_dataset = TensorDataset(task_data, task_tensor_labels)

        random_data = np.load("../data/EEG_Processed/task{}_random.npy".format(task_num+1))
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


    normal_data = np.load("../data/EEG_Processed/normal.npy")
    num_samples = len(normal_data)
    normal_labels = [0]*num_samples
    normal_data = normalize(normal_data.reshape((num_samples, 256*4)), norm='l2', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_class_labels = one_hot(normal_tensor_labels, range(2))
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_class_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets.append(normal_dataset)

    calm_data = np.load("../data/EEG_Processed/calm.npy")
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


#### Math equations
def simple_math_equations_loader(batch_size=256, num_workers=0):
    def equation(x):
        eq1 = np.sum(x, 1)
        eq2 = np.sum(x, 1)
        eq3 = np.average(x, 1)
        eq4 = np.sum(x[:, :3], 1) - np.sum(x[:, 3:5], 1)

        labels = []
        for en1, en2, en3, en4 in zip(eq1, eq2, eq3, eq4):
            l1, l2, l3, l4 = 0, 0, 0, 0
            if en1 > 5:
                l1 = 1
            else:
                l1 = 0
            if en2 < 5:
                l2 = 1
            else:
                l2 = 0
            if en3 > 0.5:
                l3 = 1
            else:
                l3 = 0
            if en4 < 0:
                l4 = 1
            else:
                l4 = 0

            labels.append([l1, l2, l3, l4])

        return np.asarray(labels)

    def get_data_set(num_samples):
        task_data = np.random.rand(num_samples, 5)
        pad = np.full([num_samples, 5], 0.5)
        task_data = np.concatenate([task_data, pad], 1)
        task_labels = equation(task_data)

        task_tensor_data = torch.Tensor(task_data)
        task_tensor_labels = torch.Tensor(task_labels)
        task_tensor_labels = torch.cat([task_tensor_data, task_tensor_labels], 1)

        return TensorDataset(task_tensor_data, task_tensor_labels)

    traindataset = get_data_set(50000)
    validdataset = get_data_set(2000)
    testdataset = get_data_set(1000)

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
    train, test = get_mnist_dataset()

    traindata = torch.zeros((len(train), 28*28))
    for i in range(0, len(train)):
        traindata[i] = train[i][0].view((28*28))

    trainclass_labels = list(i[1] for i in train)

    traintensor_class_labels = torch.Tensor(trainclass_labels)
    traintensor_class_labels = one_hot(traintensor_class_labels, range(10))

    traintensor_labels = torch.cat([traindata, traintensor_class_labels], 1)

    traindataset = TensorDataset(traindata, traintensor_labels)

    trainlabels = [i[1] for i in traindataset]

    trainsampler = AESampler(trainlabels, start_from=0)
    trainloader = DataLoader(traindataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    testdata = torch.zeros(len(test), 28*28)
    for i in range(0, len(test)):
        testdata[i] = test[i][0].view((28*28))

    testclass_labels = list(i[1] for i in test)

    testtensor_class_labels = torch.Tensor(testclass_labels)
    testtensor_class_labels = one_hot(testtensor_class_labels, range(10))

    testtensor_labels = torch.cat([testdata, testtensor_class_labels], 1)

    testdataset = TensorDataset(testdata, testtensor_labels)

    testlabels = [i[1] for i in testdataset]

    l = len(testlabels)

    validsampler = AESampler(testlabels, start_from=0, amount=int(l*0.2))
    validloader = DataLoader(testdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(testlabels, start_from=int(l*0.2))
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    print("Done preparing AE dataloader")

    return (trainloader, validloader, testloader)


def mnist_class_loader(args, batch_size=256, num_workers=0):
    assert len(args) == 1
    cls = args[0]

    train, test = get_mnist_dataset()
    traindata = []
    for i in range(len(train)):
        if train[i][1] == cls:
            traindata.append(train[i][0].view((28*28)).numpy())
    traindata = np.array(traindata)

    trainclass_labels = [cls]*len(traindata)

    traindata = torch.Tensor(traindata)
    traintensor_class_labels = torch.Tensor(trainclass_labels)
    traintensor_class_labels = one_hot(traintensor_class_labels, range(10))

    traintensor_labels = torch.cat([traindata, traintensor_class_labels], 1)

    traindataset = TensorDataset(traindata, traintensor_labels)

    trainlabels = [i[1] for i in traindataset]

    trainsampler = AESampler(trainlabels, start_from=0)
    trainloader = DataLoader(traindataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)

    testdata = []
    for i in range(len(test)):
        if test[i][1] == cls:
            testdata.append(train[i][0].view((28*28)).numpy())
    testdata = np.array(testdata)

    testclass_labels = [cls]*len(testdata)

    testdata = torch.Tensor(testdata)
    testtensor_class_labels = torch.Tensor(testclass_labels)
    testtensor_class_labels = one_hot(testtensor_class_labels, range(10))

    testtensor_labels = torch.cat([testdata, testtensor_class_labels], 1)

    testdataset = TensorDataset(testdata, testtensor_labels)

    testlabels = [i[1] for i in testdataset]

    l = len(testlabels)

    validsampler = AESampler(testlabels, start_from=0, amount=int(l * 0.2))
    validloader = DataLoader(testdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)

    testsampler = AESampler(testlabels, start_from=int(l * 0.2))
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)

    return trainloader, validloader, testloader


def get_mnist_dataset():
    dataloader = datasets.MNIST

    transform_all = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        GaussianNoise(0, 0.2)
    ])

    trainset = dataloader(root="../data/MNIST/", train=True, download=True, transform=transform_all)
    testset = dataloader(root="../data/MNIST/", train=False, download=False, transform=transform_all)

    return trainset, testset
