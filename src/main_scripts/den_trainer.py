import copy
import torch
import os
import torch.optim as optim

from src.utils.misc import DataloaderWrapper
from src.models import ActionEncoder
from src.main_scripts.train import train


class DENTrainer:
    """
    Implements DEN training.
    """

    def __init__(self, data_loaders: [(DataloaderWrapper, DataloaderWrapper, DataloaderWrapper)],
                 sizes: dict, learning_rate: float, momentum: float, criterion, penalty, expand_by_k: int,
                 device: torch.device, err_func, err_stop_threshold=None):

        # Get the loaders by task
        self.train_loaders = []
        self.valid_loaders = []
        self.test_loaders = []

        for train_loader, eval_loader, test_loader in data_loaders:
            self.train_loaders.append(train_loader)
            self.valid_loaders.append(eval_loader)
            self.test_loaders.append(test_loader)

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

        self.number_of_tasks = len(data_loaders)
        self.model = ActionEncoder(sizes=sizes).to(device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self.__epochs_to_train = None

    # Train Functions
    def train_all_tasks_sequentially(self, epochs: int, with_den=True):
        errs = []
        for i in range(self.number_of_tasks):
            print("Task: [{}/{}]".format(i+1, self.number_of_tasks))

            loss, err = self.train_tasks([i], epochs, with_den)
            errs.append(err)

            print("Task: [{}/{}] Ended with Err: {}".format(i + 1, self.number_of_tasks, err))

        # Reset for next train call
        if hasattr(self.penalty, 'old_model'):
            self.penalty.old_model = None

        return errs

    def train_tasks(self, tasks: [int], epochs: int, with_den=True):
        # train_new_neurons will need this.
        self.__epochs_to_train = epochs

        # l1l2penalty is initialized elsewhere, but we must set the old_model
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        # Make a copy for split, get train_loader.
        model_copy = copy.deepcopy(self.model) if with_den else None
        train_loader = self.train_loaders[tasks[0]]  # Give the data loader of the first task only.

        loss, err = None, None
        for i in range(epochs):
            loss, err = self.train_one_epoch(train_loader, tasks)  # Selective retraining from t=0.
            if err >= self.err_stop_threshold:
                break

        # Do DEN.
        if (len(tasks) > 1 or tasks[0] > 0) and with_den:
            # Desaturate saturated neurons
            old_sizes, new_sizes = self.split_saturated_neurons(model_copy)
            loss, err = self.train_new_neurons(old_sizes, new_sizes, tasks)
            self.prune_zero_nodes() # TODO: Fix this call.

            # If loss is still above a certain threshold, add capacity.
            if loss > self.loss_threshold:
                old_sizes, new_sizes = self.dynamically_expand()
                loss, err = self.train_new_neurons(old_sizes, new_sizes, tasks)
                self.prune_zero_nodes()

            # Post-DEN error
            err = self.error_function(self.model, train_loader, tasks)

            # Reset for next task
            if hasattr(self.penalty, 'old_model'):
                self.penalty.old_model = None

        return loss, err

    def train_one_epoch(self, loader: DataloaderWrapper, tasks):
        # This should be already set unless absolutely only training one epoch
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, False, self.device, tasks)
        err = self.error_function(self.model, loader, tasks)
        return loss, err

    # Eval Function
    def eval_model(self, task=None):
        return self.__loss_for_loader_in_eval(self.valid_loaders, task)

    # Test function
    def test_model(self, task=None):
        return self.__loss_for_loader_in_eval(self.test_loaders, task)

    def __loss_for_loader_in_eval(self, loaders, task=None):
        if task is None:
            losses = []
            for i in range(self.number_of_tasks):
                loss = train(loaders[i], self.model, self.criterion, self.optimizer, self.penalty, True, self.device)
                err = self.error_function(self.model, loaders[i], i)
                losses.append((loss, err))

            return losses

        else:
            loss = train(loaders[task], self.model, self.criterion, self.optimizer, self.penalty, True, self.device)
            err = self.error_function(self.model, loaders[task], task)
            return loss, err

    # DEN Functions
    def split_saturated_neurons(self, model_copy: torch.nn.Module):
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
            new_layer_size = 0  # Needed here, to make last layer fixed size.

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

                        # Add old neuron
                        new_layer_weights.append(old_weights)
                        new_layer_biases.append(old_bias)

                    else:
                        # One neuron not split
                        new_layer_size += 1

                        # Add existing neuron back
                        new_layer_weights.append(new_weights)
                        new_layer_biases.append(new_bias)

                # Update dicts
                weights[dict_key].append(new_layer_weights)
                biases[dict_key].append(new_layer_biases)
                sizes[dict_key].append(new_layer_size)

                biases_index += 1

            # Output must remain constant
            sizes[dict_key][-1] -= new_layer_size

        # Be efficient
        old_sizes = self.model.sizes
        if total_neurons_added > 0:
            self.model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)

        return old_sizes, self.model.sizes

    def dynamically_expand(self):
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

    def train_new_neurons(self, old_sizes, new_sizes, tasks):
        # TODO: Freeze the old neurons in each layer except outpu/input
        raise NotImplementedError

        # TODO: All this is wrong because the params don't contain new weights + will be overwritten
        # From split
        # # Register hook to freeze param
        # active_weights = [False] * (len(new_layer_weights) - number_of_neurons_split)
        # active_weights.extend([True] * number_of_neurons_split)
        #
        # if prev_neurons is None:
        #     prev_neurons = [True] * new_neurons[0][2].shape[1]
        #
        # # All neurons belong to same param.
        # hook = new_neurons[0][2].register_hook(freeze_hook(prev_neurons, active_weights))
        # hooks.append(hook)
        # hook = new_neurons[0][3].register_hook(freeze_hook(None, active_weights, bias=True))
        # hooks.append(hook)
        #
        # # Push current layer to next.
        # prev_neurons = active_weights

        # From DE: Weights
        # # Register hook to freeze param
        # active_neurons = [False] * (param.shape[0] - self.expand_by_k)
        # active_neurons.extend([True] * self.expand_by_k)
        #
        # if prev_neurons is None:
        #     prev_neurons = [True] * param.shape[0]
        #
        # hook = param.register_hook(freeze_hook(prev_neurons, active_neurons))
        # hooks.append(hook)
        #
        # # Pushes current set of neurons to next.
        # prev_neurons = active_neurons

        # From DE: Biases
        # hook = param.register_hook(freeze_hook(None, active_neurons, bias=True))
        # hooks.append(hook)

        # Train
        # l1l2penalty old_model should be set
        assert not hasattr(self.penalty, 'old_model') or self.penalty.old_model is None

        loss, err = self.train_tasks(tasks, self.__epochs_to_train, with_den=False)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Return loss, err
        return loss, err

    # TODO: Delete this function. Currently not called
    # def select_neurons(self):
    #     modules = get_modules(self.model)
    #
    #     prev_active = [True] * self.number_of_tasks
    #     prev_active[self.task] = False
    #
    #     action_hooks, prev_active = gen_hooks(modules['action'], self.zero_threshold, prev_active)
    #     encoder_hooks, _ = gen_hooks(modules['encoder'], self.zero_threshold, prev_active)
    #
    #     hooks = action_hooks + encoder_hooks
    #     return hooks

    def prune_zero_nodes(self, modules: list, sizes: dict):
        # Removes 0 weights then create new model
        new_modules = get_modules(self.model)
        new_biases = {}
        new_weights = {}
        new_sizes = {}
        for ((name1, layers1), (name2, layers2)) in zip(modules.items(), new_modules.items()):
            weight_indexes = []
            added_neurons = []
            new_biases[name2] = []
            new_weights[name2] = []
            new_sizes[name2] = []
            for ((label1, param1), (label2, param2)) in zip(layers1, layers2):
                param2_detached = param2.detach()

                if 'bias' in label1:
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
    def save_model(self, model_name: str):
        filepath = os.path.join(os.path.dirname(__file__), "../../saved_models")
        filepath = os.path.join(filepath, model_name)
        torch.save({'state_dict': self.model.state_dict()}, filepath)


def get_modules(model):
    modules = {}

    for name, param in model.named_parameters():
        module = name[0: name.index('.')]
        if module not in modules.keys():
            modules[module] = []
        modules[module].append((name, param))

    return modules


def gen_hooks(layers, zero_threshold, prev_active=None):
    hooks = []
    selected = []

    layers = reversed(layers)

    for name, layer in layers:
        if 'bias' in name:
            h = layer.register_hook(active_grads_hook(prev_active, None, bias=True))
            hooks.append(h)
            continue

        x_size, y_size = layer.size()

        active = [True] * y_size
        data = layer.detach()

        for x in range(x_size):
            # we skip the weight if connected neuron wasn't selected
            if prev_active[x]:
                continue

            for y in range(y_size):
                weight = data[x, y]
                # check if weight is active
                if abs(weight) > zero_threshold:
                    # mark connected neuron as active
                    active[y] = False

        h = layer.register_hook(active_grads_hook(prev_active, active))

        hooks.append(h)
        prev_active = active

        selected.append((y_size - sum(active), y_size))


    return hooks, prev_active


class freeze_hook(object):
    def __init__(self, previous_neurons, active_neurons, bias=False):
        self.previous_neurons = previous_neurons
        self.active_neurons = active_neurons
        self.bias = bias

    def __call__(self, grad):

        grad_clone = grad.clone().detach()

        if self.bias:
            for i in range(grad.shape[0]):
                grad_clone[i] = 0
        else:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    if not self.active_neurons[i] and not self.previous_neurons[j]:
                        grad_clone[i, j] = 0

        return grad_clone


class active_grads_hook(object):

    def __init__(self, mask1, mask2, bias=False):
        self.__name__ = "why do i use this"

        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()
        if mask2 is not None:
            self.mask2 = torch.Tensor(mask2).long().nonzero().view(-1).numpy()
        self.bias = bias

    def __call__(self, grad):
        with torch.autograd.detect_anomaly():
            grad_clone = grad.clone().detach()

        if self.bias:
            if self.mask1.size:
                for i in self.mask1:
                    grad_clone[i] = 0
            return grad_clone
        if self.mask1.size:
            grad_clone[self.mask1, :] = 0
        if self.mask2.size:
            grad_clone[:, self.mask2] = 0
        return grad_clone
