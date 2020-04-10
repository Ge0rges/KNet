"""
Instructions:
    1. Have your banana_car datasets in data/EEG_Raw as downloaded from drive

    - To find best hypers run "find_hypers"
    - To train one model run "train_model"
"""

import torch

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import EEG_bands_to_binary_loader
from src.utils.eval import calc_avg_AE_AUROC, build_confusion_matrix, calculate_accuracy
from src.utils.data_preprocessing import EEG_preprocess_tasks_to_binary
from src.utils.misc import DataloaderWrapper

# Global experiment params
seed = None  # Change to seed random functions. None is no Seed.
use_cuda = False  # Change to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
classes_list = range(2)  # Dataset specific, list of classification classes
data_loader = [DataloaderWrapper(EEG_bands_to_binary_loader)]  # The loader to be used for the data.
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
    core_invariant_size = None  # None is PCA

    # PBT Params
    generation_size = 8
    number_of_generations = 16
    standard_deviation = 0.15

    # Seed workers
    seed_workers = []

    # ML Hyper Bounds
    params_bounds = {
        "learning_rate": (1e-10, 100, float),
        "momentum": (0, 0.99, float),
        "lr_drop": (0, 1, float),
        "epochs_drop": (0, 20, int),
        "max_epochs": (5, 25, int),
        "l1_coeff": (1e-20, 1e-7, float),
        "l2_coeff": (1e-20, 1e-7, float),
        "zero_threshold": (0, 1e-5, float),

        "batch_size": (100, 500, int),
        "weight_decay": (0, 1, float),
        "loss_threshold": (0, 0.5, float),
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
                                  core_invariant_size=core_invariant_size, params_bounds=params_bounds,
                                  workers_seed=seed_workers)

    print("Got optimal worker:" + str(best_worker))

    return best_worker


def train_model(main_hypers=None, split_train_new_hypers=None, de_train_new_hypers=None):
    """
    Trains a CIANet model on the following params.
    """
    # ML Hypers
    if main_hypers is None:
        main_hypers = {
            # Common
            'learning_rate': 0.5,
            'momentum': 0.0,
            'lr_drop': 0.2,
            'epochs_drop': 20,
            'max_epochs': 100,
            'l1_coeff': 0.0,
            'l2_coeff': 0.0,
            'zero_threshold': 1e-4,

            ## Global net size
            "sizes": {
                "encoder": [2, 50, 10, 1],
                "action": [1, 2]
            },

            # Unique to main
            'batch_size': 256,
            'weight_decay': 0.22590651579084853,
            'loss_threshold': 0.3770745278503695,
            'expand_by_k': 8,
        }

    if split_train_new_hypers is None:
        split_train_new_hypers = {
            # Common
            'learning_rate': 0.5,
            'momentum': 0.0,
            'lr_drop': 0.2,
            'epochs_drop': 20,
            'max_epochs': 100,
            'l1_coeff': 3.329438489585988e-08,
            'l2_coeff': 7.224999999999999e-08,
            'zero_threshold': 8.5e-06,

            # Unique to split
            "drift_threshold": 0.02394479325728904
        }

    if de_train_new_hypers is None:
        de_train_new_hypers = {
            # Common
            'learning_rate': 0.5,
            'momentum': 0.0,
            'lr_drop': 0.2,
            'epochs_drop': 20,
            'max_epochs': 100,
            'l1_coeff': 3.329438489585988e-08,
            'l2_coeff': 7.224999999999999e-08,
            'zero_threshold': 8.5e-06,
        }

    # Misc Params
    save_model = None  # Pass a file name to save this model as. None does not save.

    model, results = main_ae(main_hypers=main_hypers, split_train_new_hypers=split_train_new_hypers,
                      de_train_new_hypers=de_train_new_hypers, error_function=error_function, use_cuda=use_cuda,
                      data_loader=data_loader, num_workers=num_workers, classes_list=classes_list, criterion=criterion,
                      save_model=save_model, seed_rand=seed)

    print("Done training with results from error function:" + str(results))

    return model, results


def error_function(model, batch_loader, classes_trained):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """
    # More insights
    confusion_matrix = build_confusion_matrix(model, batch_loader, classes_list, use_cuda)

    print("Confusion matrix:")
    print(confusion_matrix)

    print("Per class accuracy:")
    print(confusion_matrix.diag()/confusion_matrix.sum(1))

    accuracy = calculate_accuracy(confusion_matrix)
    print("Accuracy:")
    print(accuracy)

    # Must return one global param on performance
    auroc = calc_avg_AE_AUROC(model, batch_loader, classes_list, classes_trained, use_cuda)
    print("Auroc:")
    print(auroc)

    score = (auroc["macro"] + accuracy)/2
    print("Score: ")
    print(score)

    return score


def prepare_experiment():
    """
    Preprocesses the data.
    """
    EEG_preprocess_tasks_to_binary()


if __name__ == "__main__":
    #find_hypers()
    train_model()