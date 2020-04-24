"""
Instructions:
    1. Have your banana_car datasets in data/banana_car as downloaded from drive

    - To find best hypers run "find_hypers"
    - To train one model run "train_model"
"""
import torch

from src.main_scripts.den_trainer import DENTrainer
from src.main_scripts.hyper_optimizer import OptimizerController
from src.main_scripts.train import L1L2Penalty
from src.utils.eval import build_confusion_matrix
from src.utils.data_loading import banana_car_loader, DatasetType

# No need to touch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 8

# Global experiment params
criterion = torch.nn.BCELoss()  # Change to use different loss function
number_of_tasks = 2  # Dataset specific, list of classification classes
penalty = L1L2Penalty(l1_coeff=1e-5, l2_coeff=0)  # Penalty for all
batch_size = 256

img_size = (280, 190) # Images will be resized correctly

data_loaders = (banana_car_loader(DatasetType.train, size=img_size, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
                banana_car_loader(DatasetType.eval, size=img_size, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
                banana_car_loader(DatasetType.test, size=img_size, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory))


def find_hyperparameters():
    """
    Runs hyper_optimizer to find the best ML params.
    """
    # Net shape
    encoder_in = img_size[0] * img_size[1] * 3
    hidden_encoder_layers = 1
    hidden_action_layers = 1
    action_out = 2
    core_invariant_size = 522  # None is PCA

    pbt_controller = OptimizerController(device, data_loaders, criterion, penalty, error_function, number_of_tasks,
                                         encoder_in, hidden_encoder_layers, hidden_action_layers, action_out,
                                         core_invariant_size)

    return pbt_controller()


def train_model():
    """
    Trains a CIANet model on the following params.
    """
    epochs = 60
    learning_rate = 0.002
    momentum = 0
    expand_by_k = 10
    sizes = {"encoder": [img_size[0] * img_size[1] * 3, 1000, 522],
             "action": [522, 80, 2]}

    trainer = DENTrainer(data_loaders, sizes, learning_rate, momentum, criterion, penalty, expand_by_k, device,
                         error_function, number_of_tasks)

    results = trainer.train_all_tasks_sequentially(epochs, with_den=False)

    print("Done training with results from error function:" + str(results))

    return trainer.model, results


def error_function(model, batch_loader, tasks):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Do not modify params. Abstract method for all experiments.
    """

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_tasks, tasks, device)
    class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)

    score = 0
    for i in tasks:
        score += class_acc[i]
    score /= len(tasks)

    return score


def test_abstraction(model):
    """
    Tests to see whether the network having learned bananas and cars, can recognize a banana car.
    """
    testloader = data_loaders[2]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            print(outputs)


if __name__ == "__main__":
    train_model()
