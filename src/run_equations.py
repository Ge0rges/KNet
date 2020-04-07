"""
Train CIANet on MNIST.

There is not datapreprocessing required.

To find optimal hyper_parameters run find_hypers().
To train a model, run train_model().
"""

import torch

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import simple_math_equations_loader
from src.utils.eval import calc_avg_AE_AUROC

# Global experiment params
seed = None  # Change to seed random functions. None is no Seed.
use_cuda = False  # Change to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
classes_list = range(4)  # Dataset specific, list of classification classes
data_loader = [simple_math_equations_loader()]  # The loader to be used for the data.
num_workers = 0  # Leave this as zero for now.


def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    autoencoder_input = 10
    hidden_autoencoder_layers = 2
    hidden_action_layers = 2
    actionnet_output = len(classes_list)
    core_invariant_size = None  # None is PCA

    # PBT Params
    generation_size = 8
    number_of_generations = 16
    standard_deviation = 0.1

    # Seed workers
    seed_workers = []

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
        main_hypers = {'learning_rate': 0.1025783587760153, 'momentum': 0.7382951058371613, 'lr_drop': 0.18806624433009356,
                   'epochs_drop': 0, 'max_epochs': 5, 'l1_coeff': 2.9765538262283965e-08,
                   'l2_coeff': 3.167022697095151e-08, 'zero_threshold': 6.602347377186237e-06, 'batch_size': 329,
                   'weight_decay': 0, 'loss_threshold': 0.4849395339819411, 'expand_by_k': 1,
                       'sizes' : {'encoder': [10, 10, 5], 'action': [5, 4, 4], 'decoder': [5, 10, 10]}}

    if split_train_new_hypers is None:
        split_train_new_hypers = {'learning_rate': 0.342377783532879, 'momentum': 0.7603140193361054,
                                  'lr_drop': 0.786906032737128, 'epochs_drop': 16, 'max_epochs': 3,
                                  'l1_coeff': 5.16140189773329e-08, 'l2_coeff': 7.927273938277364e-08,
                                  'zero_threshold': 7.663501673689354e-06,
                                  'drift_threshold': 0.010738526244622092}

    if de_train_new_hypers is None:
        de_train_new_hypers= {'learning_rate': 0.4642555293963474, 'momentum': 0.38298737729960264,
                               'lr_drop': 0.941087258032139, 'epochs_drop': 16, 'max_epochs': 8,
                               'l1_coeff': 3.659698326397483e-08, 'l2_coeff': 3.084589790581798e-08,
                               'zero_threshold': 2.1404714986370876e-06}

    # Misc Params
    save_model = None  # Pass a file name to save this model as. None does not save.

    results, model = main_ae(main_hypers=main_hypers, split_train_new_hypers=split_train_new_hypers,
                      de_train_new_hypers=de_train_new_hypers, error_function=error_function, use_cuda=use_cuda,
                      data_loader=data_loader, num_workers=num_workers, classes_list=classes_list, criterion=criterion,
                      save_model=save_model, seed_rand=seed)

    print("Done training with results from error function:" + str(results))

    return results, model


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
    # Nothing to do for equations.
    pass


if __name__ == "__main__":
    # find_hypers()
    train_model()