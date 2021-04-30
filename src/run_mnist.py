import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import RandomSampler, DataLoader, SubsetRandomSampler
import torch
import logging
import json
# Pytorch dataloading variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")

# where to download the data files
data_root = './data'
if not os.path.exists(data_root):
    os.mkdir(data_root)

# Output and result saving part

# We define a variable that needs to be modified for every run in order to uniquely identify them, this is used
# when saving the tensorboard metrics as well as logging the results.
run_id = 0

# where to save experiment results and output metrics
result_root = './results'
if not os.path.exists(result_root):
    os.mkdir(result_root)

output_folder_path = result_root + '/output_run_' + str(run_id)
if os.path.exists(output_folder_path):
    print("THERE ALREADY EXISTS A RUN WITH THIS RUN IDENTIFIER, PLEASE MODIFY THE RUN_ID BEFORE RESTARTING")
    exit(1)
else:
    os.mkdir(output_folder_path)

# please use the logging command to log anything that you print to the standard output that we would want to save to
# look at at a later date (example, when printing the loss, accuracy etc...)
log_file_name = output_folder_path + '/output_log.log'
logging.basicConfig(filename=log_file_name, level=logging.INFO)

# setting up the tensorboard folder logic
tensorboard_log_dir = output_folder_path + '/tb_logs'

# Creating a hyperparameter dictionary, where all hyper parameters for a specific run will be stored and then written
# to a specific output file in order to save the state of the code/input parameters when the particular experiment
# was run. This is done to facilitate going back through previous result and data gathering

# SAVE ANY AND EVERY HYPERPARAMETER VALUE THAT IS RELEVANT TO THE EXPERIMENT
hyper_parameter_dictionary = {}

# Example:
epochs = 100
hyper_parameter_dictionary['epochs'] = epochs

# put the rest of the hyper_parameters below before the dictionary save

dictionary_path = output_folder_path + '/hyper_parameters.txt'
# then we save the dictionary
with open(dictionary_path, 'w') as file:
    file.write(json.dumps(hyper_parameter_dictionary))


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

    train_set = datasets.MNIST(root=data_root, train=True, transform=transform_all, download=True,
                               target_transform=one_hot_mnist)
    test_set = datasets.MNIST(root=data_root, train=False, transform=transform_all, download=True,
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

