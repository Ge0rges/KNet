import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import os

from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, RandomSampler, SubsetRandomSampler
from sklearn.preprocessing import normalize
from src.utils.misc import one_hot, plot_tensor


class DatasetType:
    train = 0
    eval = 1
    test = 2

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
    dataset = dataset(root=root, train=is_train, download=True, transform=transform_all,
                      target_transform=one_hot_mnist)

    if is_train:
        sampler = RandomSampler(dataset)

    else:
        index = int(len(dataset) * 0.2) if (type == DatasetType.eval) else int(len(dataset) * 0.8)
        indices = list(range(index)) if (type == DatasetType.eval) else list(range(index, len(dataset)))
        sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


##### BANANA_CAR
def banana_car_loader(type, size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):

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
    dataset = datasets.ImageFolder(root=root, transform=transform_all, target_transform=one_hot_bc)

    indices = np.array(list(range(len(dataset))))
    np.random.shuffle(indices)

    if type == DatasetType.train:
        indices = indices[:int(len(dataset)*0.7)]
    elif type == DatasetType.eval:
        indices = indices[int(len(dataset)*0.7):int(len(dataset)*0.8)]
    else:
        indices = indices[int(len(dataset)*0.8):]

    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def bananacar_abstract_loader(size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):
    """Loader for the images containing cars with banana shapes"""

    def one_hot_half_half(targets):
        return torch.Tensor([0.5, 0.5])  # Assumption here may be false.

    transform_all = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    dataset = datasets.ImageFolder(root="../data/banana_car_abstract", transform=transform_all,
                                   target_transform=one_hot_half_half)

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


