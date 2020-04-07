"""
Instructions:
    1. Have your banana_car datasets in data/banana_car as downloaded from drive

    - To find best hypers run "find_hypers"
    - To train one model run "train_model"
"""
import torch
import os

from src.main_scripts import main_ae, optimize_hypers
from src.utils.data_loading import bc_loader
from src.utils.eval import calc_avg_AE_AUROC, build_confusion_matrix, calculate_accuracy
from src.utils.data_preprocessing import dataset_reshaping

# Global experiment params
seed = None  # Change to seed random functions. None is no Seed.
use_cuda = False  # Change to use CUDA
criterion = torch.nn.BCELoss()  # Change to use different loss function
classes_list = range(2)  # Dataset specific, list of classification classes
num_workers = 0  # Leave this as zero for now.

# The loader to be used for the data. Change the data path if necessary
filepath1 = os.path.join(os.path.dirname(__file__), "../data/banana_car/banana/resized/1/")
filepath2 = os.path.join(os.path.dirname(__file__), "../data/banana_car/car/resized/1/")

data_loader = [bc_loader(filepath1, "banana", 0, batch_size=256, num_workers=0),
               bc_loader(filepath2, "car", 1, batch_size=256, num_workers=0)]


def find_hypers():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    autoencoder_input = 640*480*3
    hidden_autoencoder_layers = 2
    hidden_action_layers = 2
    actionnet_output = 2

    # PBT Params
    generation_size = 8
    number_of_generations = 16
    standard_deviation = 0.15

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
                                  params_bounds=params_bounds, workers_seed=seed_workers)

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
        "max_epochs": 5,
        "l1_coeff": 1e-10,
        "l2_coeff": 1e-10,
        "zero_threshold": 1e-4,

        ## Global net size
        "sizes": {
            "encoder": [640*480*3, 1000, 500, 80],
            "action": [80, 30, 2]
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
        "max_epochs": 5,
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
        "max_epochs": 5,
        "l1_coeff": 1e-10,
        "l2_coeff": 1e-10,
        "zero_threshold": 1e-4,
    }

    # Misc Params
    save_model = None  # Pass a file name to save this model as. None does not save.

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
    # More insights
    confusion_matrix = build_confusion_matrix(model, batch_loader, classes_list, use_cuda)

    print("Confusion matrix:")
    print(confusion_matrix)

    print("Per class accuracy:")
    print(confusion_matrix.diag()/confusion_matrix.sum(0))

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
    banana_path = "../data/banana_car/banana/"  # the path to your banana dataset folder, include ending /
    dataset_reshaping("banana", banana_path)
    car_path = "../data/banana_car/car/"  # the path to your car dataset folder, include ending /
    dataset_reshaping("car", car_path)
    bananacar_path = "../data/banana_car/bananacar/"  # the path to your bananacar dataset folder, include ending /
    dataset_reshaping("bananacar", bananacar_path)


if __name__ == "__main__":
    find_hypers()
