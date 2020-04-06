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
    standard_deviation = 0.15

    # Seed workers
    seed_worker1 = (0.7246460006382267, {'learning_rate': 0.7384640327769042, 'momentum': 0.8598320969754832, 'lr_drop': 0.3172240101431633, 'epochs_drop': 1, 'max_epochs': 10, 'l1_coeff': 4.165433558425681e-08, 'l2_coeff': 5.8896573962450576e-09, 'zero_threshold': 7.029369421570681e-06, 'batch_size': 122, 'weight_decay': 0.5856925737367484, 'loss_threshold': 0.7529815160276062, 'expand_by_k': 15, 'split_train_new_hypers': { 'learning_rate': 0.530872924975473, 'momentum': 0.014401177866599224, 'lr_drop': 0.4071493418952058, 'epochs_drop': 17, 'max_epochs': 8, 'l1_coeff': 8.433902091324702e-09, 'l2_coeff': 1.3126443551337794e-08,'zero_threshold': 7.44147066414474e-06, 'drift_threshold': 0.00862938966175561,'sizes': {'encoder': [2, 1],'action': [1, 2]}},'de_train_new_hypers': {'learning_rate': 1, 'momentum': 0.5963914353652274, 'lr_drop': 0.6826728043526626, 'epochs_drop': 7, 'max_epochs': 6, 'l1_coeff': 2.423418088450744e-09, 'l2_coeff': 4.452500546919578e-08, 'zero_threshold': 3.3902056652424897e-06, 'sizes': {'encoder': [2, 1],'action': [1, 2]}},'sizes': {'encoder': [2, 1], 'action': [1, 2], 'decoder': [1, 2]}})
    seed_worker2 = (0.7246542999473127, {'learning_rate': 0.5404443342533098, 'momentum': 0.022898874617269166, 'lr_drop': 0.5556241661988092, 'epochs_drop': 2, 'max_epochs': 7, 'l1_coeff': 9.431408916540752e-08, 'l2_coeff': 5.433585521469062e-08, 'zero_threshold': 4.6777037659333246e-06, 'batch_size': 376, 'weight_decay': 0.2718898941367216, 'loss_threshold': 0.2479326228982458, 'expand_by_k': 8, 'split_train_new_hypers': {'learning_rate': 0.20361101107796925, 'momentum': 0.5781271380305903, 'lr_drop': 0.11056892721075391, 'epochs_drop': 12, 'max_epochs': 4, 'l1_coeff': 2.1891598517866282e-08, 'l2_coeff': 9e-08, 'zero_threshold': 9.801000000000002e-06, 'drift_threshold': 0.021300827983799883, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'de_train_new_hypers': {'learning_rate': 0.9, 'momentum': 0.8732691000000002, 'lr_drop': 0.07269849825015848, 'epochs_drop': 16, 'max_epochs': 6, 'l1_coeff': 3.294716324326585e-08, 'l2_coeff': 4.192039333662363e-08, 'zero_threshold': 4.708662664103033e-06, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'sizes': {'encoder': [2, 1], 'action': [1, 2], 'decoder': [1, 2]}})
    seed_worker3 = (0.7246714974223379, {'learning_rate': 0.43221048335119716,'momentum': 0.4130845269017391, 'lr_drop': 0.2991681528667821, 'epochs_drop': 10, 'max_epochs': 6, 'l1_coeff': 6.17552390006323e-08, 'l2_coeff': 9.505488739409965e-09, 'zero_threshold': 1e-05, 'batch_size': 112, 'weight_decay': 0.9, 'loss_threshold': 0.4595191294967011, 'expand_by_k': 6, 'split_train_new_hypers': {'learning_rate': 0.1764005904724052, 'momentum': 0.8019000000000001, 'lr_drop': 0.2690553959888499, 'epochs_drop': 6, 'max_epochs': 3, 'l1_coeff': 5.424443009997431e-08, 'l2_coeff': 8.2947105100903e-09, 'zero_threshold': 5.30550863349907e-07, 'drift_threshold': 0.027388279856092025, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'de_train_new_hypers': {'learning_rate': 0.25882372099305484, 'momentum': 0.46095404273277957, 'lr_drop': 0.27147046871931135, 'epochs_drop': 7, 'max_epochs': 3, 'l1_coeff': 1.361617364124071e-08, 'l2_coeff': 3.4749566496501905e-08, 'zero_threshold': 6.2030774501031425e-06, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'sizes': {'encoder': [2, 1], 'action': [1, 2], 'decoder': [1, 2]}})
    seed_worker4 = (0.7246232948654313, {'learning_rate': 0.335021481960109, 'momentum': 0.2447200267206516, 'lr_drop': 1, 'epochs_drop': 12, 'max_epochs': 11, 'l1_coeff': 6.555513268949992e-08, 'l2_coeff': 3.1114930854699994e-08, 'zero_threshold': 1.294197001725149e-06, 'batch_size': 488, 'weight_decay': 1, 'loss_threshold': 0.17969023854185553, 'expand_by_k': 5, 'split_train_new_hypers': {'learning_rate': 0.04248705959175594, 'momentum': 0.42005847673087343, 'lr_drop': 0.24991203340771462, 'epochs_drop': 0, 'max_epochs': 5, 'l1_coeff': 4.757081659426345e-08, 'l2_coeff': 4.11758974363732e-08, 'zero_threshold': 2.5468802740799175e-06, 'drift_threshold': 0.04828436770036001, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'de_train_new_hypers': {'learning_rate': 0.19770469779668398, 'momentum': 0.3492200437714917, 'lr_drop': 0.6003071874999998, 'epochs_drop': 13, 'max_epochs': 3, 'l1_coeff': 5.466252256957571e-08, 'l2_coeff': 7.975969234861216e-08, 'zero_threshold': 3.942938911535103e-06, 'sizes': {'encoder': [2, 1], 'action': [1, 2]}}, 'sizes': {'encoder': [2, 1], 'action': [1, 2], 'decoder': [1, 2]}})

    seed_workers = [seed_worker1, seed_worker2, seed_worker3, seed_worker4]

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
            "encoder": [2, 1],
            "action": [1, 2]
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
    EEG_preprocess_tasks_to_binary()


if __name__ == "__main__":
    find_hypers()
