import copy
import torch
import os
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from src.models import ActionEncoder
from src.main_scripts.train import train


class DENTrainer:
    """
    Implements DEN training.
    """

    def __init__(self, data_loaders: (DataLoader, DataLoader, DataLoader),
                 sizes: dict, learning_rate: float, momentum: float, criterion, penalty, expand_by_k: int,
                 device: torch.device, err_func: callable, number_of_tasks: int, drift_threshold: float,
                 err_stop_threshold: float = None) -> None:

        # Get the loaders by task
        self.train_loader = data_loaders[0]
        self.valid_loader = data_loaders[1]
        self.test_loader = data_loaders[2]

        # Initalize params
        self.penalty = penalty
        self.criterion = criterion
        self.device = device
        self.expand_by_k = expand_by_k
        self.error_function = err_func
        self.err_stop_threshold = err_stop_threshold if err_stop_threshold else float("inf")

        # DEN Thresholds
        self.pruning_threshold = 0.05  # Percentage of parameters to prune (lowest)
        self.drift_threshold = drift_threshold

        self.loss_threshold = 1e-2

        self.number_of_tasks = number_of_tasks  # experiment specific
        self.model = ActionEncoder(sizes=sizes, pruning_threshold=self.pruning_threshold).to(device)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.l2_coeff = self.penalty.l2_coeff if hasattr(self.penalty, "l2_coeff") else 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.l2_coeff)

        self.__epochs_to_train = None
        self.__current_tasks = None

    # Train Functions
    def train_all_tasks_sequentially(self, epochs: int, with_den: bool) -> [float]:
        errs = []
        for i in range(self.number_of_tasks):
            print("Task: [{}/{}]".format(i + 1, self.number_of_tasks))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # DEN on task 0 is ok.
            if i == 0:
                loss, err = self.train_tasks([i], epochs, False)
                errs.append(err)

            else:
                loss, err = self.train_tasks([i], epochs, True)
                errs.append(err)

            print("Task: [{}/{}] Ended with Err: {}".format(i + 1, self.number_of_tasks, err))

        return errs

    def train_tasks(self, tasks: [int], epochs: int, with_den: bool) -> (float, float):
        # train_new_neurons will need this.
        self.__epochs_to_train = epochs
        self.__current_tasks = tasks

        # Make a copy for split
        model_copy = copy.deepcopy(self.model).to(self.device) if with_den else None

        # Train
        loss, err = self.__train_tasks_for_epochs()
        print(err)

        # Do DEN.
        if with_den:
            loss, err = self.__do_den(model_copy, loss)

        # Return validation error
        err = self.error_function(self.model, self.valid_loader, tasks)
        return loss, err

    def __train_tasks_for_epochs(self):
        loss, err = None, None
        for i in range(self.__epochs_to_train):
            loss, err = self.__train_one_epoch()
            if err is not None and err >= self.err_stop_threshold:
                break

        return loss, err

    def __train_one_epoch(self) -> (float, float):
        loss = train(self.train_loader, self.model, self.criterion, self.optimizer, self.penalty, False, self.device, self.__current_tasks)

        # Compute the error if we need early stopping
        err = None
        if True or self.err_stop_threshold != float("inf"):
            err = self.error_function(self.model, self.valid_loader, self.__current_tasks)

        return loss, err

    def __do_den(self, model_copy: torch.nn.Module, starting_loss: float) -> (float, float):
        # Desaturate saturated neurons
        old_sizes, new_sizes = self.split_saturated_neurons(model_copy)
        loss, err = self.train_new_neurons(old_sizes, new_sizes)
        print(err)

        # If old_sizes == new_sizes, train_new_neurons has nothing to train => None loss.
        loss = starting_loss if loss is None else loss

        # If loss is still above a certain threshold, add capacity.
        if loss > self.loss_threshold:
            old_sizes, new_sizes = self.dynamically_expand()
            t_loss, err = self.train_new_neurons(old_sizes, new_sizes)
            print(err)

            # If old_sizes == new_sizes, train_new_neurons has nothing to train => None loss.
            loss = loss if t_loss is None else t_loss

        return loss, err


    # DEN Functions
    def split_saturated_neurons(self, model_copy: torch.nn.Module) -> (dict, dict):
        print("Splitting...")
        total_neurons_added = 0

        new_sizes, weights, biases = {}, {}, {}

        old_modules = get_modules(model_copy)
        new_modules = get_modules(self.model)

        drifts = []

        # For each module (encoder, decoder, action...)
        for (_, old_module), (dict_key, new_module), in zip(old_modules.items(), new_modules.items()):
            # Initialize the dicts
            new_sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

            # Biases needed before going through weights
            old_biases = []
            new_biases = []

            # First get all biases.
            for (_, old_param), (new_param_name, new_param) in zip(old_module, new_module):
                if "bias" in new_param_name:
                    new_biases.append(new_param)
                    old_biases.append(old_param)

            # Go through per node/weights
            biases_index = 0
            added_last_layer = 0  # Needed here, to make last layer fixed size.

            # For each layer
            for (_, old_param), (new_param_name, new_param) in zip(old_module, new_module):
                # Skip biases params
                if "bias" in new_param_name:
                    continue

                # Need input size
                if len(new_sizes[dict_key]) == 0:
                    new_sizes[dict_key].append(new_param.shape[1])

                new_layer_weights = []
                new_layer_biases = []
                new_layer_size = 0
                added_last_layer = 0

                # For each node's weights
                for j, new_weights in enumerate(new_param.detach()):
                    old_bias = old_biases[biases_index].detach()[j]
                    old_weights = old_param.detach()[j]

                    new_bias = new_biases[biases_index].detach()[j]

                    # Check drift
                    diff = old_weights - new_weights
                    drift = diff.norm(2)
                    drifts.append(drift.to(torch.device("cpu")).numpy())

                    if drift > self.drift_threshold:
                        # Split 1 neuron into 2
                        new_layer_size += 2
                        total_neurons_added += 1
                        added_last_layer += 1

                        # Add old neuron
                        new_layer_weights.append(old_weights.tolist())
                        new_layer_biases.append(old_bias)

                    else:
                        # One neuron not split
                        new_layer_size += 1

                        # Add existing neuron back
                        new_layer_weights.append(new_weights.tolist())
                        new_layer_biases.append(new_bias)

                # Update dicts
                weights[dict_key].append(new_layer_weights)
                biases[dict_key].append(new_layer_biases)
                new_sizes[dict_key].append(new_layer_size)

                biases_index += 1

            # Output must remain constant
            new_sizes[dict_key][-1] -= added_last_layer

        # Be efficient
        old_sizes = self.model.sizes
        if total_neurons_added > 0:
            self.model = ActionEncoder(new_sizes, self.pruning_threshold, oldWeights=weights, oldBiases=biases)
            self.model = self.model.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                       weight_decay=self.l2_coeff)
        print(self.model.sizes)
        print("median drift: {} \n mean drift: {}".format(np.median(drifts), np.mean(drifts)))

        return old_sizes, self.model.sizes

    def dynamically_expand(self) -> (dict, dict):
        print("Expanding...")

        sizes, weights, biases = {}, {}, {}
        modules = get_modules(self.model)

        for dict_key, module in modules.items():
            sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

            for module_name, param in module:
                if 'bias' not in module_name:
                    if len(sizes[dict_key]) == 0:
                        sizes[dict_key].append(param.shape[1])

                    sizes[dict_key].append(param.shape[0] + self.expand_by_k)
                    weights[dict_key].append(param.detach())

                elif 'bias' in module_name:
                    biases[dict_key].append(param.detach())

            # Output must remain constant
            sizes[dict_key][-1] -= self.expand_by_k

        old_sizes = self.model.sizes
        self.model = ActionEncoder(sizes, self.pruning_threshold, oldWeights=weights, oldBiases=biases)
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                   weight_decay=self.l2_coeff)

        print(self.model.sizes)

        return old_sizes, self.model.sizes

    def train_new_neurons(self, old_sizes: dict, new_sizes: dict) -> (float, float):
        if old_sizes == new_sizes:
            print("No new neurons to train.")
            return (None, None)

        print("Training new neurons...")

        # Generate hooks for each layer
        hooks = []
        modules = get_modules(self.model)

        for module_name, parameters in modules.items():
            previously_active_weights = [False] * new_sizes[module_name][0]

            for param_name, param in parameters:
                split_param_name = param_name.split(".")  # Splits action.0.weights
                param_index = int(split_param_name[1])

                # Map every two indices to one
                param_index -= param_index % 2
                param_index /= 2
                param_index = int(param_index)

                old_size = old_sizes[module_name][param_index+1]
                new_size = new_sizes[module_name][param_index+1]
                neurons_added = new_size - old_size

                # Input/Output must stay the same
                if param_index == len(old_sizes[module_name]) - 1:
                    assert old_size == new_size
                    continue

                # Freeze biases/weights
                if "bias" in param_name:
                    active_biases = [False] * old_size + [True] * neurons_added
                    hook = ActiveGradsHook(None, active_biases, bias=True)

                    hook = param.register_hook(hook)
                    hooks.append(hook)

                else:

                    active_weights = [False] * old_size + [True] * neurons_added
                    hook = ActiveGradsHook(previously_active_weights, active_weights, bias=False)

                    hook = param.register_hook(hook)
                    hooks.append(hook)

                    previously_active_weights = active_weights

        # Train until validation loss reaches a maximum
        max_model = self.model
        max_validation_loss, max_validation_err = self.eval_model(self.__current_tasks, False)[0]

        # Initial train
        for _ in range(2):
            self.__train_one_epoch()
        validation_loss, validation_error = self.eval_model(self.__current_tasks, False)[0]

        # Train till validation error stops growing
        while validation_error > max_validation_err:
            max_model = self.model
            max_validation_err = validation_error
            max_validation_loss = validation_loss

            for _ in range(2):
                self.__train_one_epoch()
            validation_loss, validation_error = self.eval_model(self.__current_tasks, False)[0]

        self.model = max_model  # Discard the last two train epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                   weight_decay=self.l2_coeff)
        # Remove hooks
        for hook in hooks:
            hook.remove()

        return max_validation_loss, max_validation_err

    # Eval Function
    def eval_model(self, tasks, sequential=False) -> [(float, float)]:
        return self.__loss_for_loader_in_eval(self.valid_loader, tasks, sequential)

    # Test function
    def test_model(self, tasks, sequential=False) -> [(float, float)]:
        return self.__loss_for_loader_in_eval(self.test_loader, tasks, sequential)

    def __loss_for_loader_in_eval(self, loader, tasks, sequential) -> [(float, float)]:
        if sequential:
            losses = []
            for t in tasks:
                loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, True, self.device,
                             [t])
                err = self.error_function(self.model, loader, range(t))
                losses.append((loss, err))

            return losses

        else:
            loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, True, self.device, tasks)
            err = self.error_function(self.model, loader, tasks)
            return [(loss, err)]

    # Misc
    def load_model(self, model_name: str) -> bool:
        filepath = os.path.join(os.path.dirname(__file__), "../../saved_models")
        filepath = os.path.join(filepath, model_name)

        self.model = ActionEncoder(self.model.sizes, self.pruning_threshold)
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)

        return isinstance(self.model, ActionEncoder)

    def save_model(self, model_name: str) -> None:
        filepath = os.path.join(os.path.dirname(__file__), "../../saved_models")
        filepath = os.path.join(filepath, model_name)
        torch.save(self.model.state_dict(), filepath)


def get_modules(model: torch.nn.Module) -> dict:
    modules = {}

    for name, param in model.named_parameters():
        module = name[0: name.index('.')]

        if module not in modules.keys():
            modules[module] = []

        modules[module].append((name, param))

    return modules


class ActiveGradsHook:
    """
    Resets the gradient according to the passed masks.
    """

    def __init__(self, previously_active: [bool], currently_active: [bool], bias=False):

        # Could be None for biases
        if previously_active is not None:
            self.previously_active = torch.BoolTensor(previously_active).long().nonzero().view(-1).numpy()

        # Should never be None
        self.currently_active = torch.BoolTensor(currently_active).long().nonzero().view(-1).numpy()

        self.is_bias = bias

        self.__name__ = None

    def __call__(self, grad):
        grad_clone = grad.clone().detach()

        if self.is_bias:
            grad_clone[self.currently_active] = 0

        else:
            grad_clone[self.currently_active, :] = 0
            grad_clone[:, self.previously_active] = 0

        return grad_clone
