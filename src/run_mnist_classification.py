from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SubsetRandomSampler
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from functools import partial
from src.models import FeedForward, PlasticFeedforward

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import logging
import json
import sys
import os

# Pytorch dataloading variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 0   # number of cpu threads allocated to data loading

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

mnist_folder = result_root + '/mnist_classification'
if not os.path.exists(mnist_folder):
    os.mkdir(mnist_folder)

output_folder_path = mnist_folder + '/output_run_' + str(run_id)
while os.path.exists(output_folder_path):
    run_id += 1
    output_folder_path = mnist_folder + '/output_run_' + str(run_id)
print("RUN_ID is"+ str(run_id))
os.mkdir(output_folder_path)

# logging INFO and above so we can keep ignite info
log_file_name = output_folder_path + '/logs.log'
logging.basicConfig(filename=log_file_name, level=logging.INFO)

# please write to the following file anything you print to the standard output that we would want to save to
# look at at a later date (example, when printing the loss, accuracy etc...)
std_output = output_folder_path + '/std_output.txt'
output_handle = open(std_output, 'w')

# we save the original standard output
original = sys.stdout

# setting up the tensorboard folder logic
tensorboard_log_dir = output_folder_path + '/tb_logs'

# Creating a hyperparameter dictionary, where all hyper parameters for a specific run will be stored and then written
# to a specific output file in order to save the state of the code/input parameters when the particular experiment
# was run. This is done to facilitate going back through previous result and data gathering

# SAVE ANY AND EVERY HYPERPARAMETER VALUE THAT IS RELEVANT TO THE EXPERIMENT
hyper_parameter_dictionary = {}

# These are example placeholder values TODO: change those values to appropriate ones once we start experimentation
epochs = 25
learning_rate = 0.001
batch_size = 256
weight_decay = 0.0001

hyper_parameter_dictionary['epochs'] = epochs
hyper_parameter_dictionary['learning_rate'] = learning_rate
hyper_parameter_dictionary['batch_size'] = batch_size
hyper_parameter_dictionary['weight_decay'] = weight_decay

# put the rest of the hyper_parameters below before the dictionary save
dictionary_path = output_folder_path + '/hyper_parameters.txt'

# there are still some variables that we would like to store that can't be stored in the header, so we don't save the
# dictionary just yet


def mnist_loader(dims=1):
    """ Regular MNIST loader, meant to be used to train a classifier """
    def one_hot_mnist(targets):
        # targets_onehot = torch.zeros(10)
        # targets_onehot[targets] = 1
        targets_onehot = targets
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
    print(train_set)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)

    index = int(len(test_set) * 0.2)
    test_indices = list(range(index, len(test_set)))
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=True)

    eval_indices = list(range(index))
    eval_sampler = SubsetRandomSampler(eval_indices)
    eval_loader = DataLoader(test_set, sampler=eval_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=True)

    return train_loader, test_loader, eval_loader


def setup_event_handler(trainer, evaluator, train_loader, test_loader, eval_loader):
    log_interval = 10

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        """ Logs training loss at every epoch """
        out = "Epoch[{}] Loss: {:.5f}".format(trainer.state.epoch, trainer.state.output)
        print(out)
        sys.stdout = output_handle
        print(out)
        sys.stdout = original
        # doesn't work for some reason
        # with open(std_output, 'w') as file:
        #     print("writing")
        #     file.write(out)

        writer.add_scalar("training_iteration_loss", trainer.state.output, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_training_results(trainer):
        """ Logs training metrics at every log_interval epochs """
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        out = "Training Results - Epoch: {} ".format(trainer.state.epoch)
        for key in list(metrics.keys()):
            out = out + " {}: {:.5f}".format(key, metrics[key])
        print(out)
        sys.stdout = output_handle
        print(out)
        sys.stdout = original

        for key in list(metrics.keys()):
            writer.add_scalar("training_"+key, metrics[key], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_eval_results(trainer):
        """ Logs evaluation metrics at every log_interval epochs """
        evaluator.run(eval_loader)
        metrics = evaluator.state.metrics
        out = "Validation Results - Epoch: {} ".format(trainer.state.epoch)
        for key in list(metrics.keys()):
            out = out + " {}: {:.5f}".format(key, metrics[key])
        print(out)
        sys.stdout = output_handle
        print(out)
        sys.stdout = original

        for key in list(metrics.keys()):
            writer.add_scalar("eval_"+key, metrics[key], trainer.state.epoch)

    @trainer.on(Events.COMPLETED)
    def log_test_results(trainer):
        """ Logs test metrics at the end of the run """
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        out = "Test Results - Epoch: {} ".format(trainer.state.epoch)
        for key in list(metrics.keys()):
            out = out + " {}: {:.5f}".format(key, metrics[key])
        print(out)
        sys.stdout = output_handle
        print(out)
        sys.stdout = original

        for key in list(metrics.keys()):
            writer.add_scalar("test_"+key, metrics[key], trainer.state.epoch)


def run():
    """ we assume all hyperparameters have been defined in the header, we should only have run code here """
    # define the model
    # model = FeedForward([784, 100, 10])  # placeholder for now
    model = PlasticFeedforward()
    # move the model to the correct device
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # we save model name and model version to the hyper param dictionary
    hyper_parameter_dictionary['model_name'] = model.name
    hyper_parameter_dictionary['model_version'] = model.version
    hyper_parameter_dictionary['optimizer'] = optimizer.__repr__()
    hyper_parameter_dictionary['loss'] = criterion._get_name()

    # we save the hyper parameter dictionary
    with open(dictionary_path, 'w') as file:
        file.write(json.dumps(hyper_parameter_dictionary, indent=4, sort_keys=4))

    train_loader, test_loader, eval_loader = mnist_loader()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    def acc_ot_func(output):
        """ this function is used to change the format of the model output to match the required format
        for the accuracy metric """
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=10).to(torch.float)
        return (y_pred, y)

    # here we store all the metrics we want the evaluator to look at
    # if any metrics are deleted, make sure to also remove them from the setup_event_handler function
    # if any metric is added, make sure to add it where appropriate in the setup_event_handler function otherwise
    # it won't appear in the logs
    val_metrics = {
        # We're using Accuracy for classification tasks, however it might not make sense for AutoEncoders
        "accuracy": Accuracy(output_transform=partial(acc_ot_func)),
        "nll": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader, eval_loader)

    trainer.run(train_loader, max_epochs=epochs)

    # we now save the final trained model
    model_save_path = output_folder_path + '/model'
    torch.save(model, model_save_path)
    out = "Model Saved!"
    print(out)
    output_handle.write(out)


if __name__ == '__main__':
    run()
    output_handle.close()
