"""
Train CIANet on MNIST.

There is not datapreprocessing required.

To find optimal hyper_parameters run find_hypers().
To train a model, run train_model().
"""

import torch

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import mnist_loader, mnist_class_loader
from src.utils.eval import calc_avg_AE_AUROC
from src.utils.misc import DataloaderWrapper, plot_tensor

# Global experiment params
seed = None  # Change to seed random functions. None is no Seed.
use_cuda = False  # Change to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
classes_list = range(10)  # Dataset specific, list of classification classes
data_loaders = [DataloaderWrapper(mnist_class_loader, args=[i]) for i in classes_list]  # The loader to be used for the data.
num_workers = 0  # Leave this as zero for now.


def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    autoencoder_input = 28*28
    hidden_autoencoder_layers = 1
    hidden_action_layers = 1
    actionnet_output = 10
    core_invariant_size = None  # None is PCA

    # PBT Params
    generation_size = 8
    number_of_generations = 16
    standard_deviation = 0.1

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
                                  use_cuda=use_cuda, data_loader=data_loaders, num_workers=num_workers,
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
            "learning_rate": 20,
            "momentum": 0,
            "lr_drop": 0.5,
            "epochs_drop": 20,
            "max_epochs": 20,
            "l1_coeff": 0,
            "l2_coeff": 0,
            "zero_threshold": 1e-4,

            ## Global net size
            "sizes": {
                "encoder": [28*28, 312, 128, 32, 10],
                "action": [10, 10],
                "decoder": [10, 32, 128, 312, 28 * 28]
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
            "learning_rate": 20,
            "momentum": 0.0,
            "lr_drop": 0.5,
            "epochs_drop": 20,
            "max_epochs": 10,
            "l1_coeff": 1e-10,
            "l2_coeff": 1e-10,
            "zero_threshold": 1e-4,

            # Unique to split
            "drift_threshold": 0.02
        }

    if de_train_new_hypers is None:
        de_train_new_hypers = {
            # Common
            "learning_rate": 0.20,
            "momentum": 0.0,
            "lr_drop": 0.5,
            "epochs_drop": 20,
            "max_epochs": 10,
            "l1_coeff": 1e-10,
            "l2_coeff": 1e-10,
            "zero_threshold": 1e-4,
        }

    # Misc Params
    save_model = None  # Pass a file name to save this model as. None does not save.

    model, results = main_ae(main_hypers=main_hypers, split_train_new_hypers=split_train_new_hypers,
                      de_train_new_hypers=de_train_new_hypers, error_function=error_function, use_cuda=use_cuda,
                      data_loader=data_loaders, num_workers=num_workers, classes_list=classes_list, criterion=criterion,
                      save_model=save_model, seed_rand=seed)

    print("Done training with results from error function:" + str(results))

    return model, results


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
    # Nothing to do for MNIST.
    pass


def test_img_display():
    dataloader = data_loaders[0]
    train, valid, test = dataloader.get_loaders()
    for idx, (inputs, targets) in enumerate(train):
        plot_tensor(inputs[0], (28, 28))


if __name__ == "__main__":
    # find_hypers()
    train_model()
