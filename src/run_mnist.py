"""
Train CIANet on MNIST.

There is not datapreprocessing required.

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
penalty = L1L2Penalty(l1_coeff=1e-5, l2_coeff=0.00001)  # Penalty for all
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
                                         encoder_in, hidden_encoder_layers, hidden_action_layers, action_out,
                                         core_invariant_size)

    return pbt_controller()


def train_model():
    """
    Trains a CIANet model on the following params.
    """

    epochs = 5
    learning_rate = 1
    momentum = 0
    expand_by_k = 10
    sizes = {"encoder": [28 * 28, 312, 128, 10],
             "action": [10, 10]}

    trainer = DENTrainer(data_loaders, sizes, learning_rate, momentum, criterion, penalty, expand_by_k, device,
                         error_function, number_of_tasks)

    results = trainer.train_all_tasks_sequentially(epochs, with_den=True)

    print("Done training with results from error function:" + str(results))

    return trainer.model, results


def error_function(model, batch_loader, tasks):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_tasks, tasks, device)
    class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)

    score = 0
    for i in range(class_acc.shape[0]):
        score += class_acc[i]
    score /= class_acc.shape[0]

    return score


if __name__ == "__main__":
    train_model()