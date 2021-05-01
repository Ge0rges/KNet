import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import RandomSampler, DataLoader, SubsetRandomSampler, TensorDataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import PSNR, Loss
from functools import partial
from src.models import AutoEncoder, FeedForward
import torch
import logging
import json
import sys
import numpy as np
# Pytorch dataloading variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 0   # number of cpu threads allocated to data loading
batch_size = 256

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

mnist_folder = result_root + '/mnist_unsupervised'
if not os.path.exists(mnist_folder):
    os.mkdir(mnist_folder)

output_folder_path = mnist_folder + '/output_run_' + str(run_id)
if os.path.exists(output_folder_path):
    print("THERE ALREADY EXISTS A RUN WITH THIS RUN IDENTIFIER, PLEASE MODIFY THE RUN_ID OR REMOVE THE EXISTING"
          " OUTPUT RUN FOLDER BEFORE RESTARTING")
    exit(1)
else:
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
epochs = 15
hyper_parameter_dictionary['epochs'] = epochs

learning_rate = 0.001
hyper_parameter_dictionary['learning_rate'] = learning_rate

hyper_parameter_dictionary['batch_size'] = batch_size

weight_decay = 0.0001
hyper_parameter_dictionary['weight_decay'] = weight_decay


# put the rest of the hyper_parameters below before the dictionary save
dictionary_path = output_folder_path + '/hyper_parameters.txt'

# there are still some variables that we would like to store that can't be stored in the header, so we don't save the
# dictionary just yet


def mnist_loader():
    """ Data loader for MNIST to train an AutoEncoder (or any model that uses the input as target), the dataset's
    targets are modified to be the inputs """

    transform_all = transforms.Compose([
        # transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.Lambda(lambda a: a.view(-1 ))
    ])
    original_train_set = datasets.MNIST(root=data_root, train=True, transform=transform_all, download=True)
    original_test_set = datasets.MNIST(root=data_root, train=False, transform=transform_all, download=True)

    train_tensors = []
    for i in range(len(original_train_set)):
        input_tensor = original_train_set[i][0]
        train_tensors.append(input_tensor.numpy())
    train_tensors = np.array(train_tensors)
    # this is necessary for some metrics
    min_train = train_tensors.min()
    max_train = train_tensors.max()
    train_range = max_train - min_train
    train_inputs = torch.Tensor(train_tensors)   # for some reason Pycharm doesn't like this but it works fine

    test_tensors = []
    for i in range(len(original_test_set)):
        input_tensor = original_test_set[i][0]
        test_tensors.append(input_tensor.numpy())
    test_tensors = np.array(test_tensors)
    # this is necessary for some metrics
    min_test = test_tensors.min()
    max_test = test_tensors.max()
    test_range = max_test - min_test
    test_inputs = torch.Tensor(test_tensors)     # for some reason Pycharm doesn't like this but it works fine

    # we squeeze as the previous process leaves us with an extra useless dimension
    train_inputs = torch.squeeze(train_inputs)
    test_inputs = torch.squeeze(test_inputs)

    # for some reason, the autoencoder doesn't like when it's not reshaped
    train_inputs = torch.reshape(train_inputs, (train_inputs.size()[0], train_inputs.size()[1]*train_inputs.size()[2]))
    test_inputs = torch.reshape(test_inputs, (test_inputs.size()[0], test_inputs.size()[1]*test_inputs.size()[2]))

    train_inputs = train_inputs.to(torch.float)
    test_inputs = test_inputs.to(torch.float)

    train_set = TensorDataset(train_inputs, train_inputs)
    test_set = TensorDataset(test_inputs, test_inputs)

    # the rest is as usual
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

    return train_loader, test_loader, eval_loader, max(train_range, test_range)


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
    # TODO: add the model once the model code has been implemented
    model = AutoEncoder([784, 100, 10])  # placeholder for now
    # move the model to the correct device
    model = model.to(device)
    # we save model name and model version to the hyper param dictionary
    hyper_parameter_dictionary['model_name'] = model.name
    hyper_parameter_dictionary['model_version'] = model.version

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    hyper_parameter_dictionary['optimizer'] = 'Adam'  # couldn't find a way of getting the name from the object

    criterion = torch.nn.BCELoss()
    hyper_parameter_dictionary['loss'] = criterion._get_name()

    # we save the hyper parameter dictionary
    with open(dictionary_path, 'w') as file:
        file.write(json.dumps(hyper_parameter_dictionary, indent=4, sort_keys=4))

    train_loader, test_loader, eval_loader, data_range = mnist_loader()

    # giving the data_range as a measure of reference for the psnr
    out = "Data Range: {:.5f}".format(data_range)
    print(out)
    sys.stdout = output_handle
    print(out)
    sys.stdout = original

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    # here we store all the metrics we want the evaluator to look at
    # the handler automatically adapts to the metrics put here, no need to modify it when adding/removing metrics
    val_metrics = {
        "psnr": PSNR(data_range),
        "loss": Loss(criterion)
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
