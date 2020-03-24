import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from train import one_hot
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import normalize


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
    if len(files) != 2:
        print("THERE MUST BE EXACTLY 2 FILES")
    # Actual task data
    data = []
    sample_n = 256

    f = files[0]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i : i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)
        pro = _fft_psd(1, sample_n, pre_pro)
        assert np.shape(pro[1]) == (sample_n, 4)
        data.append(pro[1])

    np.save("../data/EEG_Processed/{}_task".format(task), data)

    # other random data
    data = []
    sample_n = 256

    f = files[1]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i: i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)
        pro = _fft_psd(1, sample_n, pre_pro)
        assert np.shape(pro[1]) == (sample_n, 4)
        data.append(pro[1])

    np.save("../data/EEG_Processed/{}_random".format(task), data)


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


def EEG_dataset_getter(task_num):
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


def EEG_loader(batch_size=256, num_workers=4):
    # TODO: make is so that the testloader isn't re-using data that is being used in the train and valid loaders
    datasets = []
    for i in range(9):
        datasets.append(EEG_dataset_getter(i))

    dataset = ConcatDataset(datasets)

    num_samples = len(dataset)
    labels = [i[1] for i in dataset]

    # train_size = int(num_samples*0.5)
    # trainsampler = AESampler(labels, start_from=0, amount=train_size)
    # trainloader = DataLoader(dataset, batch_size=batch_size, sampler=trainsampler, num_workers=num_workers)
    # check_for_nan(trainloader)
    #
    # valid_size = int(num_samples*0.05)
    # validsampler = AESampler(labels, start_from=train_size, amount=valid_size)
    # validloader = DataLoader(dataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)
    # check_for_nan(validloader)

    testsampler = AESampler(labels, start_from=0)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)
    check_for_nan(testloader)

    print("Done preparing AE dataloader")

    return (testloader)


def EEG_task_loader(task_num, batch_size=256, num_workers=4):
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

    num_samples = len(dataset)
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
    print("Done preparing AE dataloader")

    return (trainloader, validloader)


def EEG_Mediation_preprocessing():
    files = []
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/calm/"):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + file)
    print(files)
    # if len(files) != 2:
    #     print("THERE MUST BE EXACTLY 2 FILES")

    # Actual task data
    data = []
    sample_n = 256

    f = files[0]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i : i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)
        pro = _fft_psd(1, sample_n, pre_pro)
        assert np.shape(pro[1]) == (sample_n, 4)
        data.append(pro[1])

    np.save("../data/EEG_Processed/calm", data)


    files = []
    for (dirpath, dirnames, filenames) in os.walk("../data/EEG_Raw/normal/"):
        for file in filenames:
            if file.endswith(".csv"):
                files.append(dirpath + "/" + file)
    print(files)
    # if len(files) != 2:
    #     print("THERE MUST BE EXACTLY 2 FILES")

    # Actual task data
    data = []
    sample_n = 256

    f = files[0]
    x = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)
    for i in range(np.shape(x)[0] - sample_n):
        pre_pro = x[i : i + sample_n]
        pre_pro = np.delete(pre_pro, 0, 1)
        pre_pro = np.delete(pre_pro, -1, 1)
        pro = _fft_psd(1, sample_n, pre_pro)
        assert np.shape(pro[1]) == (sample_n, 4)
        data.append(pro[1])

    np.save("../data/EEG_Processed/normal", data)


def EEG_Mediation_loader(batch_size=256, num_workers=4):

    def custom_EEG_dataset_getter(task_num, task_label, random_label):
        task_data = np.load("./data/EEG_Processed/task{}_task.npy".format(task_num+1))
        num_samples = len(task_data)
        task_labels = [task_label]*num_samples
        task_data = normalize(task_data.reshape((num_samples, 256*4)), norm='max', axis=0)
        task_data = torch.Tensor(task_data)

        task_tensor_labels = torch.Tensor(task_labels)
        task_tensor_class_labels = one_hot(task_tensor_labels, range(2))
        task_tensor_labels = torch.cat([task_data, task_tensor_class_labels], 1)

        task_dataset = TensorDataset(task_data, task_tensor_labels)

        random_data = np.load("./data/EEG_Processed/task{}_random.npy".format(task_num+1))
        num_samples = len(random_data)
        random_labels = [random_label]*num_samples
        random_data = normalize(random_data.reshape((num_samples, 256*4)), norm='max', axis=0)
        random_data = torch.Tensor(random_data)

        random_tensor_labels = torch.Tensor(random_labels)
        random_tensor_class_labels = one_hot(random_tensor_labels, range(2))
        random_tensor_labels = torch.cat([random_data, random_tensor_class_labels], 1)

        random_dataset = TensorDataset(random_data, random_tensor_labels)

        dataset = ConcatDataset([task_dataset, random_dataset])
        return dataset
    #
    datasets = []
    # for i in range(1, 9):
    #     datasets.append(custom_EEG_dataset_getter(i, 0, 0))
    datasets.append(custom_EEG_dataset_getter(0, 1, 0))


    normal_data = np.load("./data/EEG_Processed/normal.npy")
    num_samples = len(normal_data)
    normal_labels = [0]*num_samples
    normal_data = normalize(normal_data.reshape((num_samples, 256*4)), norm='max', axis=0)
    normal_data = torch.Tensor(normal_data)

    normal_tensor_labels = torch.Tensor(normal_labels)
    normal_tensor_class_labels = one_hot(normal_tensor_labels, range(2))
    normal_tensor_labels = torch.cat([normal_data, normal_tensor_class_labels], 1)

    normal_dataset = TensorDataset(normal_data, normal_tensor_labels)
    datasets.append(normal_dataset)

    calm_data = np.load("./data/EEG_Processed/calm.npy")
    num_samples = len(calm_data)
    calm_labels = [1]*num_samples
    calm_data = normalize(calm_data.reshape((num_samples, 256*4)), norm='max', axis=0)
    calm_data = torch.Tensor(calm_data)

    calm_tensor_labels = torch.Tensor(calm_labels)
    calm_tensor_class_labels = one_hot(calm_tensor_labels, range(2))
    calm_tensor_labels = torch.cat([calm_data, calm_tensor_class_labels], 1)

    calm_dataset = TensorDataset(calm_data, calm_tensor_labels)
    datasets.append(calm_dataset)
    num_samples = 0
    for i in datasets:
        num_samples += len(i)

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
    check_for_nan(trainloader)

    validsampler = AESampler(valid_labels, start_from=0)
    validloader = DataLoader(validdataset, batch_size=batch_size, sampler=validsampler, num_workers=num_workers)
    check_for_nan(validloader)

    testsampler = AESampler(test_labels, start_from=0)
    testloader = DataLoader(testdataset, batch_size=batch_size, sampler=testsampler, num_workers=num_workers)
    check_for_nan(testloader)

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
    # for i in range(1, 10):
    #     EEG_preprocessing("task{}".format(i))
    # EEG_Mediation_preprocessing()
    # EEG_Mediation_loader()
    pass