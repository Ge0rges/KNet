import copy
import torch
import os
import torch.optim as optim

from src.models import ActionEncoder
from src.main_scripts.train import train


class DENTrainer:
    """
    Implements DEN training.
    """

    def __init__(self, data_loaders, sizes, learning_rate, momentum, criterion, penalty, expand_by_k, device, err_func):

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

        self.zero_threshold = 1e-05
        self.drift_threshold = 0.02
        self.loss_threshold = 1e-2

        self.number_of_tasks = len(data_loaders)
        self.task = 0
        self.model = ActionEncoder(sizes=sizes)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self._epochs_to_train = None

    # Train Functions
    def train_all_tasks(self, epochs):
        errs = []
        for i in range(self.number_of_tasks):
            self.task = i
            errs.append(self.train_current_task(epochs))

        return errs

    def train_current_task(self, epochs, with_den=True):
        # train_new_neurons will need this.
        self._epochs_to_train = epochs

        # l1l2penalty is initialized elsewhere, but we must set the old_model
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        # Make a copy for split, get train_loader.
        model_copy = copy.deepcopy(self.model) if with_den else None
        train_loader = self.train_loaders[self.task]

        for i in range(epochs):
            self.train_one_epoch(train_loader)  # DEN requires selective training. We do this from t=0.

        # Do DEN.
        if self.task > 0 and with_den:
            raise NotImplementedError
            # Supposedly we won't need this anymore because kevin is implementing SR in all
            # hooks = self.select_neurons()
            #
            # for i in range(epochs):
            #     self.train_one_epoch(train_loader)
            #
            # for hook in hooks:
            #     hook.remove()

            # Desaturate saturated neurons
            self.model, loss = self.split_saturated_neurons(model_copy, train_loader)

            # If loss is still above a certain threshold, add capacity.
            if loss > self.loss_threshold:
                self.model = self.dynamically_expand(train_loader)

        return self.error_function(self.model, train_loader, self.task)

    def train_one_epoch(self, loader):
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        loss = train(loader, self.model, self.criterion, self.optimizer, self.penalty, False, self.device)
        err = self.error_function
        return loss, err

    # Eval Function
    def eval_model(self, task=None):
        return self._loss_for_loader_in_eval(self.valid_loaders, task)

    # Test function
    def test_model(self, task=None):
        return self._loss_for_loader_in_eval(self.test_loaders, task)

    def _loss_for_loader_in_eval(self, loaders, task=None):
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
    def select_neurons(self):
        modules = get_modules(self.model)

        prev_active = [True] * self.number_of_tasks
        prev_active[self.task] = False

        action_hooks, prev_active = gen_hooks(modules['action'], self.zero_threshold, prev_active)
        encoder_hooks, _ = gen_hooks(modules['encoder'], self.zero_threshold, prev_active)

        hooks = action_hooks + encoder_hooks
        return hooks

    def split_saturated_neurons(self, model_copy, loader):
        sizes, weights, biases, hooks = {}, {}, {}, []

        suma = 0

        old_modules = get_modules(model_copy)
        new_modules = get_modules(self.model)
        for (_, old_module), (dict_key, new_module), in zip(old_modules.items(), new_modules.items()):
            # Initialize the dicts
            sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

            # Make  per layer bias/weight pairs
            old_layers = []  # "{action0": [(bias, weights), ...], ...}
            new_layers = []

            old_biases = []
            new_biases = []

            # First get all biases.
            for (old_param_name, old_param), (new_param_name, new_param) in zip(old_module, new_module):

                # Construct the neurons
                if "bias" in new_param_name:
                    new_biases.append(new_param)
                    old_biases.append(old_param)

            # Then match with weights.
            weight_index = 0
            for (old_param_name, old_param), (new_param_name, new_param) in zip(old_module, new_module):
                if "bias" not in new_param_name:
                    old_layers.append([])
                    new_layers.append([])

                    for j, new_weights in enumerate(new_param.data):
                        old_layers[weight_index].append(
                            (old_biases[weight_index].data[j], old_param.data[j], old_param, old_biases[weight_index]))
                        new_layers[weight_index].append(
                            (new_biases[weight_index].data[j], new_weights, new_param, new_biases[weight_index]))
                    weight_index += 1

            prev_neurons = None
            # For each layer, rebuild the weight and bias tensors.
            for old_neurons, new_neurons in zip(old_layers, new_layers):
                new_layer_weights = []
                new_layer_biases = []
                new_layer_size = 0
                append_to_end_weights = []
                append_to_end_biases = []

                # For each neuron add the weights and biases back, check drift.
                for j, (old_neuron, new_neuron) in enumerate(zip(old_neurons, new_neurons)):  # For each neuron
                    # Add existing neuron back
                    new_layer_weights.append(new_neuron[1].tolist())
                    new_layer_biases.append(new_neuron[0])

                    # Increment layer size
                    new_layer_size += 1

                    # Need input size
                    if len(sizes[dict_key]) == 0:
                        sizes[dict_key].append(len(new_neuron[1]))

                    # Check drift
                    diff = old_neuron[1] - new_neuron[1]
                    drift = diff.norm(2)

                    if drift > self.drift_threshold:
                        suma += 1
                        new_layer_size += 1  # Increment again because added neuron

                        # Modify new_param weight to split
                        new_layer_weights[j] = old_neuron[1].tolist()
                        random_weights = torch.rand(1, len(new_neuron[1]))
                        append_to_end_weights.append(random_weights.tolist()[0])  # New weights are random

                        # Modify new_param  bias to split.
                        new_layer_biases[j] = old_neuron[0]
                        append_to_end_biases.append(0)  # New bias is 0

                # Append the split weights and biases to end of layer
                new_layer_weights.extend(append_to_end_weights)
                new_layer_biases.extend(append_to_end_biases)

                # Update dicts
                weights[dict_key].append(new_layer_weights)
                biases[dict_key].append(new_layer_biases)
                sizes[dict_key].append(new_layer_size)

                # Register hook to freeze param
                active_weights = [False] * (len(new_layer_weights) - len(append_to_end_weights))
                active_weights.extend([True] * len(append_to_end_weights))

                if prev_neurons is None:
                    prev_neurons = [True] * new_neurons[0][2].shape[1]

                hook = new_neurons[0][2].register_hook(
                    freeze_hook(prev_neurons, active_weights))  # All neurons belong to same param.
                hooks.append(hook)
                hook = new_neurons[0][3].register_hook(freeze_hook(None, active_weights, bias=True))
                hooks.append(hook)

                # Push current layer to next.
                prev_neurons = active_weights

                if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
                    sizes[dict_key][-1] -= len(append_to_end_weights)

        # Be efficient
        if suma == 0:
            return self.model

        return self.train_new_neurons(new_modules, sizes, weights, biases, hooks, loader)

    def dynamically_expand(self, loader):
        sizes, weights, biases, hooks = {}, {}, {}, []
        modules = get_modules(self.model)

        for dict_key, module in modules.items():
            sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

            prev_neurons = None
            for module_name, param in module:
                if 'bias' not in module_name:
                    if len(sizes[dict_key]) == 0:
                        sizes[dict_key].append(param.data.shape[1])

                    weights[dict_key].append(param.data)
                    sizes[dict_key].append(param.data.shape[0] + self.expand_by_k)

                    # Register hook to freeze param
                    active_neurons = [False] * (param.data.shape[0] - self.expand_by_k)
                    active_neurons.extend([True] * self.expand_by_k)

                    if prev_neurons is None:
                        prev_neurons = [True] * param.data.shape[0]

                    hook = param.register_hook(freeze_hook(prev_neurons, active_neurons))
                    hooks.append(hook)

                    # Pushes current set of neurons to next.
                    prev_neurons = active_neurons

                elif 'bias' in module_name:
                    raise NotImplementedError # What's active neurons suppose to be?

                    biases[dict_key].append(param.data)
                    hook = param.register_hook(freeze_hook(None, active_neurons, bias=True))
                    hooks.append(hook)

                else:
                    raise LookupError()

            if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
                sizes[dict_key][-1] -= self.expand_by_k

        # From here, everything taken from DE. #
        return self.train_new_neurons(modules, sizes, weights, biases, hooks, loader)

    def train_new_neurons(self, modules, sizes, weights, biases, hooks, loader):
        # TODO: Make module generation dynamic
        new_model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)

        # l1l2penalty is initlaized elsewhere, but we must set the old_model
        if hasattr(self.penalty, 'old_model') and self.penalty.old_model is None:
            self.penalty.old_model = self.model

        for i in range(self._epochs_to_train):
            train(loader, new_model, self.criterion, self.optimizer, self.penalty, False, self.device)

        # Remove hooks. Hooks still needed?
        raise NotImplementedError # Do hooks even work on new_model, different params registered?
        for hook in hooks:
            hook.remove()

        new_modules = get_modules(new_model)

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
                if 'bias' in label1:
                    new_layer = []

                    # Copy over old bias
                    for i in range(param1.data.shape[0]):
                        new_layer.append(float(param2.data[i]))

                    # Copy over incoming bias for new neuron for previous existing
                    for i in range(param1.data.shape[0], param2.data.shape[0]):
                        if float(param2[i].norm(1)) > zero_threshold:
                            new_layer.append(float(param2.data[i]))

                    new_biases[name2].append(new_layer)

                else:
                    new_layer = []

                    # Copy over old neurons
                    for i in range(param1.data.shape[0]):
                        row = []
                        for j in range(param1.data.shape[1]):
                            row.append(float(param2.data[i, j]))
                        new_layer.append(row)

                    # Copy over output weights for new neuron for previous existing neuron in the next layer
                    for j in range(param1.data.shape[1], param2.data.shape[1]):
                        for i in range(param1.data.shape[0]):
                            if j in weight_indexes:
                                new_layer[i].append(float(param2.data[i, j]))

                    # Copy over incoming weights for new neuron for previous existing
                    weight_indexes = []  # Marks neurons with none zero incoming weights
                    for i in range(param1.data.shape[0], param2.data.shape[0]):
                        row = []
                        if float(param2[i].norm(1)) > zero_threshold:
                            weight_indexes.append(i)
                            for j in range(param2.data.shape[1]):
                                row.append(float(param2.data[i, j]))
                        new_layer.append(row)

                    new_weights[name2].append(new_layer)
                    added_neurons.append(weight_indexes)

            new_sizes[name2] = [sizes[name2][0]]
            for i, added_weights in enumerate(added_neurons):
                new_sizes[name2].append(sizes[name2][i + 1] + len(added_weights))

        return ActionEncoder(new_sizes, oldWeights=new_weights, oldBiases=new_biases)

    # Misc
    def save_model(self, model_name):
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
        data = layer.data

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