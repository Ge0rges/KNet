import copy
import torch
import os
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from src.models import ActionEncoder
from src.main_scripts.train import train
from src.utils.misc import get_modules, FreezeNeuronsHook, FreezeWeightsHook

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
        self.drift_threshold = drift_threshold
        self.zero_threshold = 1e-4  # weights below this treat as 0 in selective retraining
        self.loss_threshold = 0.2  # loss above this do expand

        self.number_of_tasks = number_of_tasks  # experiment specific
        self.model = ActionEncoder(sizes=sizes).to(device)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.l2_coeff = self.penalty.l2_coeff if hasattr(self.penalty, "l2_coeff") else 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.l2_coeff)

        self.__epochs_to_train = None
        self.__current_tasks = None

    # Train Functions
    def train_all_tasks_sequentially(self, epochs: int, with_den: bool) -> [float]:
        errs = []
        # tasks = list(range(self.number_of_tasks))
        for i in range(self.number_of_tasks):
            print("Task: [{}/{}]".format(i + 1, self.number_of_tasks))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tasks = [i]

            # No DEN on t=0.
            loss, err = self.train_tasks(tasks, epochs, (with_den and i > 0))
            errs.append(err)

            print("Task: [{}/{}] Ended with Err: {}".format(i + 1, self.number_of_tasks, err))

        return errs

    def train_tasks(self, tasks: [int], epochs: int, with_den: bool) -> (float, float):
        # train_new_neurons will need this.
        self.__epochs_to_train = epochs
        self.__current_tasks = tasks

        # No DEN
        if not with_den:
            loss, err = self.__train_tasks_for_epochs()
            print(err)

        # Do DEN. Special training regime.
        else:
            self.__train_last_layer()

            # Make a copy for split
            model_copy = copy.deepcopy(self.model).to(self.device)
            with SelectiveRetraining(self.model, self.number_of_tasks, self.__current_tasks, self.zero_threshold):
                loss, err = self.__train_tasks_for_epochs()
                print(err)
                loss, err = self.__do_den(model_copy, loss)
                print(err)

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
        if self.err_stop_threshold != float("inf"):
            err = self.error_function(self.model, self.valid_loader, self.__current_tasks)

        return loss, err

    def __train_last_layer(self):
        params = list(self.model.parameters())
        for param in params[:-2]:
            param.requires_grad = False

        self.__train_tasks_for_epochs()

        for param in self.model.parameters():
            param.requires_grad = True

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
        count = 0
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
                    if count == 0:
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
            count = 1

        # Be efficient
        old_sizes = self.model.sizes
        if total_neurons_added > 0:
            self.model = ActionEncoder(new_sizes, oldWeights=weights, oldBiases=biases)
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
        self.model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)
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
                    hook = FreezeNeuronsHook(None, active_biases, bias=True)

                    hook = param.register_hook(hook)
                    hooks.append(hook)

                else:
                    active_weights = [False] * old_size + [True] * neurons_added
                    hook = FreezeNeuronsHook(previously_active_weights, active_weights, bias=False)

                    hook = param.register_hook(hook)
                    hooks.append(hook)

                    previously_active_weights = active_weights

        # Train until validation loss reaches a maximum
        max_model = self.model
        max_validation_loss, max_validation_err = self.eval_model(self.__current_tasks, False)[0]

        # Initial train
        for _ in range(3):
            self.__train_one_epoch()
        validation_loss, validation_error = self.eval_model(self.__current_tasks, False)[0]

        # Train till validation error stops growing
        while validation_error > max_validation_err:
            max_model = self.model
            max_validation_err = validation_error
            max_validation_loss = validation_loss

            for _ in range(3):
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
    def eval_model(self, tasks: list, sequential=False) -> [(float, float)]:
        return self.__loss_for_loader_in_eval(self.valid_loader, tasks, sequential)

    # Test function
    def test_model(self, tasks: list, sequential=False) -> [(float, float)]:
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

        # Else, all tasks as once
        loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, True, self.device, tasks)
        err = self.error_function(self.model, loader, tasks)
        return [(loss, err)]

    # Misc
    def load_model(self, filepath) -> bool:
        assert os.path.isdir(filepath)

        self.model = ActionEncoder(self.model.sizes, self.pruning_threshold)
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)

        return isinstance(self.model, ActionEncoder)

    def save_model(self, model_name: str, dir_path: str = "../../saved_models") -> str:
        filepath = os.path.join(os.path.dirname(__file__), dir_path)
        assert os.path.isdir(filepath)

        filepath = os.path.join(filepath, model_name)

        torch.save(self.model.state_dict(), filepath)

        return filepath


class SelectiveRetraining:
    def __init__(self, model, number_of_tasks, tasks, zero_threshold):
        self.hooks = None
        self.model = model
        self.number_of_tasks = number_of_tasks
        self.tasks = tasks
        self.zero_threshold = zero_threshold

    def __enter__(self):
        dict_keys_to_include = ['action', 'encoder']  # Modules that don't have this name will be ignored.

        modules = get_modules(self.model)
        hooks = []

        while len(dict_keys_to_include) > 0:
            for dict_key, parameters in modules.items():
                if dict_key != dict_keys_to_include[0]:
                    continue

                new_hooks = self._select_neurons(modules[dict_key])
                hooks.extend(new_hooks)
            del dict_keys_to_include[0]

        self.hooks = hooks

        return hooks

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()

    def _select_neurons(self, module_layers):
        hooks = []

        for param_name, param in reversed(module_layers):

            # Freeze biases/weights for weights under a certain value
            if "bias" in param_name:
                # Continue freeze the bias of nodes where all weights are under the threshold
                continue

            else:
                mask = torch.zeros([param.shape[0], param.shape[1]], dtype=torch.bool)

                for x in range(param.shape[0]):  # Rows is size of last layer (first row ever is output size)
                    for y in range(param.shape[1]):  # Columns is size of current layer (last column ever is input size)
                        weight = param[x, y]
                        mask[x, y] = (abs(weight) > self.zero_threshold)

                hook = FreezeWeightsHook(mask)

                hook = param.register_hook(hook)
                hooks.append(hook)

        return hooks
