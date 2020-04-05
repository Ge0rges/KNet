"""
Instructions:
    1. Have your banana_car datasets in data/EEG_Raw as downloaded from drive

    - To find best hypers run "find_hypers"
    - To train one model run "train_model"
"""

import torch

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import EEG_bands_to_binary_loader
from src.utils.eval import calc_avg_AE_AUROC
from src.utils.data_preprocessing import EEG_preprocess_tasks_to_binary
import numpy as np

# Global experiment params
seed = None  # Change to seed random functions. None is no Seed.
use_cuda = False  # Change to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
classes_list = range(2)  # Dataset specific, list of classification classes
data_loader = EEG_bands_to_binary_loader  # The loader to be used for the data.
num_workers = 0  # Leave this as zero for now.


def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    autoencoder_input = 2
    hidden_autoencoder_layers = 1
    hidden_action_layers = 1
    actionnet_output = 2

    best_worker = optimize_hypers(error_function=error_function, generation_size=8, epochs=20,
                                     standard_deviation=0.1, use_cuda=use_cuda, data_loader=data_loader,
                                     num_workers=num_workers, classes_list=classes_list, criterion=criterion, seed=seed,
                                  encoder_in=autoencoder_input, hidden_encoder=hidden_autoencoder_layers,
                                  hidden_action=hidden_action_layers, action_out=actionnet_output)

    print("Got optimal worker:" + str(best_worker))


def train_model(main_hypers, split_train_new_hypers, de_train_new_hypers):
    """
    Trains a CIANet model on the following params.
    """
    # ML Hypers
    if main_hypers is None:
        main_hypers = {
            # Common
            "learning_rate": 0.2,
            "momentum": 0.0,
            "lr_drop": 0.25,
            "epochs_drop": 5,
            "max_epochs": 5,
            "l1_coeff": 1e-10,
            "l2_coeff": 1e-10,
            "zero_threshold": 1e-4,

            ## Global net size
            "sizes": {
                "encoder": [2, 2],
                "action": [2, 3, 2]
            },

            # Unique to main
            "batch_size": 256,
            "weight_decay": 0,
            "loss_threshold": 1e-2,
            "expand_by_k": 10,
        }
    if split_train_new_hypers is None:
        split_train_new_hypers = {
            # Common
            "learning_rate": 0.2,
            "momentum": 0.0,
            "lr_drop": 0.25,
            "epochs_drop": 5,
            "max_epochs": 5,
            "l1_coeff": 1e-10,
            "l2_coeff": 1e-10,
            "zero_threshold": 1e-4,

            # Unique to split
            "drift_threshold": 0.02
        }

    if de_train_new_hypers is None:
        de_train_new_hypers = {
            # Common
            "learning_rate": 0.2,
            "momentum": 0.0,
            "lr_drop": 0.25,
            "epochs_drop": 5,
            "max_epochs": 5,
            "l1_coeff": 1e-10,
            "l2_coeff": 1e-10,
            "zero_threshold": 1e-4,
        }

    # Misc Params
    save_model = None # Pass a file name to save this model as. None does not save.

    results = main_ae(main_hypers=main_hypers, split_train_new_hypers=split_train_new_hypers,
                      de_train_new_hypers=de_train_new_hypers, error_function=error_function, use_cuda=use_cuda,
                      data_loader=data_loader, num_workers=num_workers, classes_list=classes_list, criterion=criterion,
                      save_model=save_model, seed_rand=seed)

    print("Done training with results from error function:" + str(results))
    return results


def error_function(model, batch_loader, classes_trained):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """
    auroc = calc_avg_AE_AUROC(model, batch_loader, classes_list, classes_trained, use_cuda)
    return auroc["macro"]


def prepare_experiment():
    """
    Preprocesses the data.
    """
    EEG_preprocess_tasks_to_binary()


if __name__ == "__main__":
    find_hypers()
