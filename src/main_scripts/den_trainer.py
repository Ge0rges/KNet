import copy
import torch
import os
import torch.optim as optim

from torch.utils.data import DataLoader
from src.models import ActionEncoder
from src.main_scripts.train import train


class DENTrainer:
    """
    Implements DEN training.
    """

    def __init__(self, data_loaders: (DataLoader, DataLoader, DataLoader),
                 sizes: dict, learning_rate: float, momentum: float, criterion, penalty, expand_by_k: int,
                 device: torch.device, err_func: callable, number_of_tasks: int, err_stop_threshold:float=None) -> None:

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
        self.zero_threshold = 1e-05
        self.drift_threshold = 0.02
        self.loss_threshold = 1e-2

        self.number_of_tasks = number_of_tasks  # experiment specific
        self.model = ActionEncoder(sizes=sizes).to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self.__epochs_to_train = None

    # Train Functions
    def train_all_tasks_sequentially(self, epochs: int, with_den=True) -> [float]:
        errs = []
        for i in range(self.number_of_tasks):
            print("Task: [{}/{}]".format(i+1, self.number_of_tasks))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if i == 0:  # Never with DEN.
                loss, err = self.train_tasks([i], epochs, False)
                errs.append(err)

            else:
                loss, err = self.train_tasks([i], epochs, with_den)
                errs.append(err)

            print("Task: [{}/{}] Ended with Err: {}".format(i + 1, self.number_of_tasks, err))

        # Reset for next train call
        if hasattr(self.penalty, 'old_model'):
            self.penalty.old_model = None

        return errs

    def train_tasks(self, tasks: [int], epochs: int, with_den=True) -> (float, float):
        # train_new_neurons will need this.
        self.__epochs_to_train = epochs

        # l1l2penalty is initialized elsewhere, but we must set the old_model
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        # Make a copy for split, get train_loader.
        model_copy = copy.deepcopy(self.model) if with_den else None

        loss, err = None, None
        for i in range(epochs):
            loss, err = self.train_one_epoch(self.train_loader, self.valid_loader, tasks)
            print(err)
            # Selective retraining from t=0.
            if err >= self.err_stop_threshold:
                break

        # Do DEN.
        if with_den:
            # Desaturate saturated neurons
            old_sizes, new_sizes = self.split_saturated_neurons(model_copy)
            loss, err = self.train_new_neurons(old_sizes, new_sizes, tasks)
            self.prune_zero_nodes()

            # If loss is still above a certain threshold, add capacity.
            if loss > self.loss_threshold:
                old_sizes, new_sizes = self.dynamically_expand()
                loss, err = self.train_new_neurons(old_sizes, new_sizes, tasks)
                self.prune_zero_nodes()

            # Post-DEN error
            err = self.error_function(self.model, self.valid_loader, tasks)

            # Reset for next task
            if hasattr(self.penalty, 'old_model'):
                self.penalty.old_model = None

        # testing the model with the test set
        err = self.error_function(self.model, self.test_loader, tasks)
        return loss, err

    def train_one_epoch(self, trainloader: DataLoader, validloader: DataLoader, tasks) -> (float, float):
        # This should be already set unless absolutely only training one epoch
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        loss = train(trainloader, self.model, self.criterion, self.optimizer, self.penalty, False, self.device, tasks)
        err = self.error_function(self.model, validloader, tasks)
        return loss, err

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
                loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, True, self.device, [t])
                err = self.error_function(self.model, loader, range(t))
                losses.append((loss, err))

            return losses

        else:
            loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, True, self.device, tasks)
            err = self.error_function(self.model, loader, tasks)
            return [loss, err]

    # DEN Functions
    def split_saturated_neurons(self, model_copy: torch.nn.Module) -> (dict, dict):
        print("Splitting...")
        total_neurons_added = 0

        sizes, weights, biases = {}, {}, {}

        old_modules = get_modules(model_copy)
        new_modules = get_modules(self.model)

        # For each module (encoder, decoder, action...)
        for (_, old_module), (dict_key, new_module), in zip(old_modules.items(), new_modules.items()):
            # Initialize the dicts
            sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

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

            for (_, old_param), (new_param_name, new_param) in zip(old_module, new_module):
                # Skip biases params
                if "bias" in new_param_name:
                    continue

                # Need input size
                if len(sizes[dict_key]) == 0:
                    sizes[dict_key].append(new_param.shape[1])

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
                sizes[dict_key].append(new_layer_size)

                biases_index += 1

            # Output must remain constant
            sizes[dict_key][-1] -= added_last_layer

        # Be efficient
        old_sizes = self.model.sizes
        if total_neurons_added > 0:
            self.model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)

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
                        sizes[dict_key].append(paramshape[1])

                    weights[dict_key].append(param.detach())
                    sizes[dict_key].append(param.shape[0] + self.expand_by_k)

                elif 'bias' in module_name:
                    raise NotImplementedError  # What's active neurons suppose to be?

                    biases[dict_key].append(param.detach())

            # Output must remain constant
            sizes[dict_key][-1] -= self.expand_by_k

        old_sizes = self.model.sizes
        self.model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)

        return old_sizes, self.model.sizes

    def train_new_neurons(self, old_sizes: dict, new_sizes: dict, tasks: [int]) -> (float, float):
        # Generate hooks for each layer
        hooks = []
        modules = get_modules(self.model)

        for module_name, parameters in modules.items():
            previously_active = None

            for param_name, param in parameters:
                split_param_name = param_name.split(".")  # Splits action.0.weights
                param_index = int(split_param_name[1])

                # Map every two indices to one
                param_index -= param_index % 2
                param_index /= 2
                param_index = int(param_index)

                old_size = old_sizes[module_name][param_index]
                new_size = new_sizes[module_name][param_index]
                neurons_added = new_size - old_size

                # Input/Output must stay the same
                if param_index == 0 or param_index == len(old_sizes[module_name]) - 1:
                    assert old_size == new_size
                    previously_active = [True] * new_size

                    continue

                # Freeze biases/weights
                if "bias" in param_name:
                    active_biases = [False] * old_size + [True] * neurons_added
                    hook = ActiveGradsHook(None, active_biases, bias=True)

                    param.register_hook(hook)

                else:
                    active_weights = [False] * old_size + [True] * neurons_added
                    hook = ActiveGradsHook(previously_active, active_weights, bias=True)

                    param.register_hook(hook)

                    previously_active = active_weights

        # Train: l1l2penalty old_model should be set
        assert not hasattr(self.penalty, 'old_model') or self.penalty.old_model is not None
        loss, err = self.train_tasks(tasks, self.__epochs_to_train, with_den=False)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return loss, err

    def prune_zero_nodes(self) -> (dict, dict):
        # TODO: Fix. We just want to remove any zero nodes
        # Removes 0 weights then create new model
        new_modules = get_modules(self.model)
        new_biases = {}
        new_weights = {}
        new_sizes = {}
        for name2, layers2 in new_modules.items():
            weight_indexes = []
            added_neurons = []
            new_biases[name2] = []
            new_weights[name2] = []
            new_sizes[name2] = []
            for label2, param2 in layers2:
                param2_detached = param2.detach()

                if 'bias' in label2:
                    new_layer = []

                    # Copy over old bias
                    for i in range(param1.shape[0]):
                        new_layer.append(float(param2_detached[i]))

                    # Copy over incoming bias for new neuron for previous existing
                    for i in range(param1.shape[0], param2.shape[0]):
                        if float(param2[i].norm(1)) > self.zero_threshold:
                            new_layer.append(float(param2_detached[i]))

                    new_biases[name2].append(new_layer)

                else:
                    new_layer = []

                    # Copy over old neurons
                    for i in range(param1.shape[0]):
                        row = []
                        for j in range(param1.shape[1]):
                            row.append(float(param2_detached[i, j]))
                        new_layer.append(row)

                    # Copy over output weights for new neuron for previous existing neuron in the next layer
                    for j in range(param1.shape[1], param2.shape[1]):
                        for i in range(param1.shape[0]):
                            if j in weight_indexes:
                                new_layer[i].append(float(param2_detached[i, j]))

                    # Copy over incoming weights for new neuron for previous existing
                    weight_indexes = []  # Marks neurons with none zero incoming weights
                    for i in range(param1.shape[0], param2.shape[0]):
                        row = []
                        if float(param2[i].norm(1)) > self.zero_threshold:
                            weight_indexes.append(i)
                            for j in range(param2.shape[1]):
                                row.append(float(param2_detached[i, j]))
                        new_layer.append(row)

                    new_weights[name2].append(new_layer)
                    added_neurons.append(weight_indexes)

            new_sizes[name2] = [sizes[name2][0]]
            for i, added_weights in enumerate(added_neurons):
                new_sizes[name2].append(sizes[name2][i + 1] + len(added_weights))

        # Create new model without 0 weights
        old_sizes = self.model.sizes
        self.model = ActionEncoder(new_sizes, oldWeights=new_weights, oldBiases=new_biases)

        return old_sizes, self.model.sizes

    # Misc
    def save_model(self, model_name: str) -> None:
        filepath = os.path.join(os.path.dirname(__file__), "../../saved_models")
        filepath = os.path.join(filepath, model_name)
        torch.save({'state_dict': self.model.state_dict()}, filepath)


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

    def __init__(self, previously_active, currently_active, bias=False):

        # Could be None for biases
        if previously_active is not None:
            self.previously_active = torch.Tensor(previously_active).long().nonzero().view(-1).numpy()

        # Should never be None
        self.currently_active = torch.Tensor(currently_active).long().nonzero().view(-1).numpy()

        self.is_bias = bias

    def __call__(self, grad):
        grad_clone = grad.detach()

        if self.is_bias:
            grad_clone[self.currently_active] = 0

        else:
            grad_clone[self.previously_active, :] = 0
            grad_clone[:, self.currently_active] = 0

        return grad_clone
