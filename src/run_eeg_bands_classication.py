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
    best_worker = optimize_hypers(error_function=error_function, generation_size=8, epochs=20,
                                     standard_deviation=0.1, use_cuda=use_cuda, data_loader=data_loader,
                                     num_workers=num_workers, classes_list=classes_list, criterion=criterion, seed=seed)

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
                "action": [2, 3, 3, 2]
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

    return calc_avg_AE_AUROC(model, batch_loader, classes_list, classes_trained, use_cuda)


def prepare_experiment():
    """
    Preprocesses the data.
    """
    EEG_preprocess_tasks_to_binary()


def test_hypers(lr_range, momentum_range, lr_drop_range, max_epoch_range, n=[3, 3, 3, 3]):
    lr_iter = np.linspace(lr_range[0], lr_range[-1], n[0])
    momentum_iter = np.linspace(momentum_range[0], momentum_range[-1], n[1])
    lr_drop_iter = np.linspace(lr_drop_range[0], lr_drop_range[-1], n[2])
    max_epoch_iter = np.linspace(max_epoch_range[0], max_epoch_range[-1], n[3])
    count = 1
    results = {}
    sub_count = 0
    for lr in lr_iter:
        for momemtum in momentum_iter:
            for lr_drop in lr_drop_iter:
                for max_epoch in max_epoch_iter:
                    max_epoch = int(max_epoch)
                    for i in ["MAIN", "SPLIT", "DE", "MAIN_SPLIT", "MAIN_DE", "SPLIT_DE", "MAIN_SPLIT_DE"]:

                        print("\nMODEL #", count, "OUT OF", 7*np.prod(n))
                        params = {"lr": lr, "momentum": momemtum, "lr_drop": lr_drop, "max_epoch": max_epoch}
                        print("USING PARAMS:", params, "FOR:", i, "\n")
                        main_hypers = {
                            # Common
                            "learning_rate": lr,
                            "momentum": momemtum,
                            "lr_drop": lr_drop,
                            "epochs_drop": 5,
                            "max_epochs": max_epoch,
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
                        split_train_new_hypers = {
                            # Common
                            "learning_rate": lr,
                            "momentum": momemtum,
                            "lr_drop": lr_drop,
                            "epochs_drop": 5,
                            "max_epochs": max_epoch,
                            "l1_coeff": 1e-10,
                            "l2_coeff": 1e-10,
                            "zero_threshold": 1e-4,

                            # Unique to split
                            "drift_threshold": 0.02
                        }
                        de_train_new_hypers = {
                            # Common
                            "learning_rate": lr,
                            "momentum": momemtum,
                            "lr_drop": lr_drop,
                            "epochs_drop": 5,
                            "max_epochs": max_epoch,
                            "l1_coeff": 1e-10,
                            "l2_coeff": 1e-10,
                            "zero_threshold": 1e-4,
                        }
                        if i == "MAIN":
                            r = train_model(main_hypers, None, None)
                        elif i == "SPLIT":
                            r = train_model(None, split_train_new_hypers, None)
                        elif i == "DE":
                            r = train_model(None, None, de_train_new_hypers)
                        elif i == "MAIN_SPLIT":
                            r = train_model(main_hypers, split_train_new_hypers, None)
                        elif i == "MAIN_DE":
                            r = train_model(main_hypers, None, de_train_new_hypers)
                        elif i == "SPLIT_DE":
                            r = train_model(None, split_train_new_hypers, de_train_new_hypers)
                        elif i == "MAIN_SPLIT_DE":
                            r = train_model(main_hypers, split_train_new_hypers, de_train_new_hypers)
                        else:
                            print("INVALID KEY!!!!!!!!")
                            return None, None
                        results[count] = {"auroc": r[0]["macro"], "i": i, "params": params}
                        count += 1

                        if sub_count == np.prod(n):
                            max = 0
                            best_params = {}
                            for j in range(1, count):
                                if results[j]["auroc"] > max:
                                    best_params = results[j]
                            print("\n ################################")
                            print("\n CURRENT BEST PARAMETERS:", best_params)
                            print("\n ################################\n")
                            sub_count = 0
    max = 0
    best_params = {}
    for i in range(1, count):
        if results[i]["auroc"] > max:
            best_params = results[i]
    return best_params, results


if __name__ == "__main__":
    # prepare_experiment()
    # train_model(None, None, None)
    best_params, results = test_hypers([0.05, 0.5], [0, 0.25], [0, 0.3], [1, 10])
    print("\n RESULTS:", results)
    print("\n BEST PARAMS:", best_params)
