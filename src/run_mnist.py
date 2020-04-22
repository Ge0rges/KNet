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
from src.main_scripts.train import L1L2Penalty, ResourceConstrainingPenalty
from src.utils.eval import build_confusion_matrix
from src.utils.data_loading import mnist_loader

# No need to touch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 4

# Global experiment params
criterion = torch.nn.BCELoss()  # Change to use different loss function
number_of_tasks = 10  # Dataset specific, list of classification classes
penalty = L1L2Penalty(l1_coeff=1e-5, l2_coeff=0)  # Penalty for all
batch_size = 256

data_loaders = (mnist_loader(True, batch_size=256, num_workers=batch_size, pin_memory=pin_memory),
                mnist_loader(False, batch_size=256, num_workers=batch_size, pin_memory=pin_memory),
                mnist_loader(False, batch_size=256, num_workers=batch_size, pin_memory=pin_memory))

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

    pbt_controller = OptimizerController(device, data_loaders, criterion, penalty, error_function, encoder_in,
                                         hidden_encoder_layers, hidden_action_layers, action_out, core_invariant_size)

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
                         error_function, 10)

    results = trainer.train_all_tasks_sequentially(epochs)

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
    for i in tasks:
        score += class_acc[i]
    score /= len(tasks)

    return score


if __name__ == "__main__":
    coeffs = [1e-10, 1e-5, 1e-2, 1, 10, 100, 1000, 1000000]
    resources = [0.1, 10, 100, 1000, 10000, 1000000000]
    exps = [2, 3, 4, 5, 6]
    str = []
    for c in coeffs:
        for r in resources:
            for e in exps:
                penalty = ResourceConstrainingPenalty(coeff=c, resources_available=r, exponent=e)
                _, results = train_model()

                str.append("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(c, r, e, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9]))

    print(str)