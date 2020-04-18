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
from src.main_scripts.train import l1l2_penalty
from src.utils.eval import calc_avg_AE_AUROC, build_confusion_matrix, calculate_accuracy
from src.utils.data_loading import mnist_loader, mnist_proportional_class_loader
from src.utils.misc import DataloaderWrapper

# Global experiment params
device = torch.device("cpu")  # Change to "gpu" to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
number_of_tasks = range(10)  # Dataset specific, list of classification classes
penalty = l1l2_penalty(l1_coeff=1e-5, l2_coeff=0, old_model=None)  # Penalty for all

data_loaders = []
for i in number_of_tasks:
    data_loader = []
    for j in ["train", "valid", "test"]:
        data_loader.append(DataloaderWrapper(mnist_proportional_class_loader, i, j, batch_size=256, num_workers=0))

# Set the seed
seed = None  # Change to seed random functions. None is no Seed.
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # if use_cuda:
    #     torch.cuda.manual_seed_all(seed)

def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    encoder_in = 28*28
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

    epochs = 60
    learning_rate = 1
    momentum = 0
    expand_by_k = 10
    sizes = {"encoder": [28*28, 312, 128, 10],
             "action": [10, 10]}


    trainer = DENTrainer(data_loaders, sizes, learning_rate, momentum, criterion, penalty, expand_by_k, device, error_function)

    results = trainer.train_all_tasks(epochs)

    print("Done training with results from error function:" + str(results))

    return trainer.model, results


def error_function(model, batch_loader, classes_trained):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """

    # More insights
    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_tasks, device)

    print("Confusion matrix:")
    print(confusion_matrix)

    print("Per class accuracy:")
    print(confusion_matrix.diag() / confusion_matrix.sum(1))

    accuracy = calculate_accuracy(confusion_matrix)
    print("Accuracy:")
    print(accuracy)

    # Must return one global param on performance
    auroc = calc_avg_AE_AUROC(model, batch_loader, number_of_tasks, classes_trained, device)
    print("Auroc:")
    print(auroc)

    return accuracy


def prepare_experiment():
    """
    Preprocesses the data.
    """
    # Nothing to do for MNIST.
    pass


if __name__ == "__main__":
    train_model()