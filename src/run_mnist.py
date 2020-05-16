"""
Train CIANet on MNIST.

There is no datapreprocessing required.

To find optimal hyper_parameters run find_hypers().
To train a model, run train_model().
"""

import torch
import random
import numpy as np

from src.main_scripts.den_trainer import DENTrainer
from src.main_scripts.hyper_optimizer import OptimizerController
from src.main_scripts.train import L1L2Penalty
from src.utils.eval import build_confusion_matrix
from src.utils.data_loading import mnist_loader, DatasetType

# No need to touch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 4

# Global experiment params
criterion = torch.nn.BCELoss()  # Change to use different loss function
number_of_tasks = 10  # Dataset specific, list of classification classes
penalty = L1L2Penalty(l1_coeff=0.0001, l2_coeff=0.0001)  # Penalty for all
drift_threshold = 0.08  # Drift threshold for split in DEN
batch_size = 256

data_loaders = (mnist_loader(DatasetType.train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
                mnist_loader(DatasetType.eval, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
                mnist_loader(DatasetType.test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory))

# Set the seed
seed = None  # Change to seed random functions. None is no Seed.
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def find_hyperparameters():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    encoder_in = 28 * 28
    hidden_encoder_layers = 1
    hidden_action_layers = 1
    action_out = 10
    core_invariant_size = 405  # None is PCA

    pbt_controller = OptimizerController(device, data_loaders, criterion, penalty, error_function, number_of_tasks,
                                         drift_threshold, encoder_in, hidden_encoder_layers, hidden_action_layers,
                                         action_out, core_invariant_size)

    return pbt_controller(8)  # Number of workers


def train_model():
    """
    Trains a CIANet model on the following params.
    """

    epochs = 10
    learning_rate = 0.01
    momentum = 0.9
    expand_by_k = 10
    err_stop_threshold = 0.99
    sizes = {"encoder": [28 * 28, 50, 50, 10],
             "action": [10, 10]}

    trainer = DENTrainer(data_loaders, sizes, learning_rate, momentum, criterion, penalty, expand_by_k, device,
                         error_function, number_of_tasks, drift_threshold, err_stop_threshold)

    results = trainer.train_all_tasks_sequentially(epochs, with_den=True)
    loss, err = trainer.test_model(range(number_of_tasks), False)[0]

    print("Net has final shape:" + str(trainer.model.sizes))
    print("Done training with total net accuracy:" + str(err))
    print("Done training with results from error function:" + str(results))

    return trainer.model, results


def error_function(model, batch_loader, tasks):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Metric must be "higher is better" (eg. accuracy)

    Do not modify params. Abstract method for all experiments.
    """

    # When training sequentially, look at previous tasks as well.
    if len(tasks) == 1:
        tasks = list(range(tasks[0] + 1))

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_tasks, tasks, device)
    confusion_matrix = confusion_matrix.to(torch.device("cpu"))
    # print(np.round(confusion_matrix.numpy()))

    class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)

    score = 0
    for i in range(class_acc.shape[0]):
        score += class_acc[i]
    score /= class_acc.shape[0]

    return score


if __name__ == "__main__":
    train_model()
