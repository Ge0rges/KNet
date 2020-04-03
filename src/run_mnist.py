"""
Train CIANet on MNIST.

There is not datapreprocessing required.

To find optimal hyper_parameters run find_hypers().
To train a model, run train_model().
"""

import torch

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import mnist_loader
from src.utils.eval import calc_avg_AE_AUROC

# Global experiment params
seed = None # Change to seed random functions. None is no Seed.
use_cuda = False # Change to use CUDA
criterion = torch.nn.BCELoss() # Change to use different loss function
classes_list = range(9) # Dataset specific, list of classification classes
data_loader = mnist_loader # The loader to be used for the data.
num_workers = 0 # Leave this as zero for now.


def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    best_worker = optimize_hypers(error_function=error_function, generation_size=8, epochs=20,
                                     standard_deviation=0.1, use_cuda=use_cuda, data_loader=data_loader,
                                     num_workers=num_workers, classes_list=classes_list, criterion=criterion, seed=seed)

    print("Got optimal worker:" + str(best_worker))


def train_model():
    """
    Trains a CIANet model on the following params.
    """
    # ML Hypers
    main_hypers = {
        # Common
        "learning_rate": 0.2,
        "momentum": 0.0,
        "lr_drop": 0.25,
        "epochs_drop": 5,
        "max_epochs": 1,
        "l1_coeff": 1e-10,
        "l2_coeff": 1e-10,
        "zero_threshold": 1e-4,

        ## Global net size
        "sizes": {
            "encoder": [28*28, 10],
            "action": [10, 10]
        },

        # Unique to main
        "batch_size": 256,
        "weight_decay": 0,
        "loss_threshold": 1e-2,
        "expand_by_k": 10,
    }

    split_train_new_hypers = {
        # Common
        "learning_rate": 0.2,
        "momentum": 0.0,
        "lr_drop": 0.25,
        "epochs_drop": 5,
        "max_epochs": 1,
        "l1_coeff": 1e-10,
        "l2_coeff": 1e-10,
        "zero_threshold": 1e-4,

        # Unique to split
        "drift_threshold": 0.02
    }

    de_train_new_hypers = {
        # Common
        "learning_rate": 0.2,
        "momentum": 0.0,
        "lr_drop": 0.25,
        "epochs_drop": 5,
        "max_epochs": 1,
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


def error_function(model, batch_loader, classes_trained):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """

    return calc_avg_AE_AUROC(model, batch_loader, classes_list, classes_trained, use_cuda)


def prepare_experiment():
    """
    Preprocesses the data.
    """
    # Nothing to do for MNIST.
    pass


if __name__ == "__main__":
    prepare_experiment()
    train_model()