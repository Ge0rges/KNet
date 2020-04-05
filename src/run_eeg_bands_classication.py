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

    # PBT Params
    generation_size = 8
    number_of_generations = 16
    standard_deviation = 0.1

    # Seed workers
    seed_worker1 = (0.7246460006382267, {
        'learning_rate': 0.7384640327769042, 'momentum': 0.8598320969754832, 'lr_drop': 0.3172240101431633,
        'epochs_drop': 1, 'max_epochs': 10, 'l1_coeff': 4.165433558425681e-08, 'l2_coeff': 5.8896573962450576e-09,
        'zero_threshold': 7.029369421570681e-06, 'batch_size': 122, 'weight_decay': 0.5856925737367484,
        'loss_threshold': 0.7529815160276062, 'expand_by_k': 15,
        'split_train_new_hypers': {
            'learning_rate': 0.530872924975473, 'momentum': 0.014401177866599224, 'lr_drop': 0.4071493418952058,
            'epochs_drop': 17, 'max_epochs': 8, 'l1_coeff': 8.433902091324702e-09, 'l2_coeff': 1.3126443551337794e-08,
            'zero_threshold': 7.44147066414474e-06, 'drift_threshold': 0.00862938966175561,
            'sizes': {
                'encoder': [2, 1],
                'action': [1, 2]
            }
        },
        'de_train_new_hypers': {
            'learning_rate': 1,
            'momentum': 0.5963914353652274,
            'lr_drop': 0.6826728043526626,
            'epochs_drop': 7,
            'max_epochs': 6,
            'l1_coeff': 2.423418088450744e-09,
            'l2_coeff': 4.452500546919578e-08,
            'zero_threshold': 3.3902056652424897e-06,
            'sizes': {
                'encoder': [2, 1],
                'action': [1, 2]
            }
        },
        'sizes': {'encoder': [2, 1], 'action': [1, 2], 'decoder': [1, 2]}
    })

    seed_workers = [seed_worker1]

    # ML Hyper Bounds
    params_bounds = {
        "learning_rate": (1e-10, 1, float),
        "momentum": (0, 0.99, float),
        "lr_drop": (0, 1, float),
        "epochs_drop": (0, 20, int),
        "max_epochs": (5, 25, int),
        "l1_coeff": (1e-20, 1e-7, float),
        "l2_coeff": (1e-20, 1e-7, float),
        "zero_threshold": (0, 1e-5, float),

        "batch_size": (100, 500, int),
        "weight_decay": (0, 1, float),
        "loss_threshold": (0, 1, float),
        "expand_by_k": (0, 20, int),

        "split_train_new_hypers": {
            "learning_rate": (1e-10, 1, float),
            "momentum": (0, 0.99, float),
            "lr_drop": (0, 1, float),
            "epochs_drop": (0, 20, int),
            "max_epochs": (3, 10, int),
            "l1_coeff": (1e-20, 1e-7, float),
            "l2_coeff": (1e-20, 1e-7, float),
            "zero_threshold": (0, 1e-5, float),
            "drift_threshold": (0.005, 0.05, float)
        },

        "de_train_new_hypers": {
            "learning_rate": (1e-10, 1, float),
            "momentum": (0, 0.99, float),
            "lr_drop": (0, 1, float),
            "epochs_drop": (0, 20, int),
            "max_epochs": (3, 10, int),
            "l1_coeff": (1e-20, 1e-7, float),
            "l2_coeff": (1e-20, 1e-7, float),
            "zero_threshold": (0, 1e-5, float),
        }
    }

    best_worker = optimize_hypers(error_function=error_function, generation_size=generation_size,
                                  epochs=number_of_generations, standard_deviation=standard_deviation,
                                  use_cuda=use_cuda, data_loader=data_loader, num_workers=num_workers,
                                  classes_list=classes_list, criterion=criterion, seed=seed,
                                  encoder_in=autoencoder_input, hidden_encoder=hidden_autoencoder_layers,
                                  hidden_action=hidden_action_layers, action_out=actionnet_output,
                                  params_bounds=params_bounds, workers_seed=seed_workers)

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
