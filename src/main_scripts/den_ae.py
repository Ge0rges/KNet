import random
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os

from src.models import ActionEncoder
from src.utils.train import trainAE
from src.utils import l1_penalty, l1l2_penalty

def main_ae(main_hypers=None, split_train_new_hypers=None, de_train_new_hypers=None, error_function=None, use_cuda=False,
            data_loader=None, num_workers=0, classes_list=None, criterion=None, save_model=None, seed_rand=None):

    assert data_loader is not None
    assert classes_list is not None
    assert criterion is not None
    assert main_hypers is not None
    assert split_train_new_hypers is not None
    assert de_train_new_hypers is not None
    assert error_function is not None

    # Set the seed
    if seed_rand is not None:
        random.seed(seed_rand)
        torch.manual_seed(seed_rand)
        if use_cuda:
            torch.cuda.manual_seed_all(seed_rand)

    # Parse hyper-params
    learning_rate = main_hypers["learning_rate"]
    batch_size = main_hypers["batch_size"]
    loss_threshold = main_hypers["loss_threshold"]
    expand_by_k = main_hypers["expand_by_k"]
    max_epochs = main_hypers["max_epochs"]
    weight_decay = main_hypers["weight_decay"]
    lr_drop = main_hypers["lr_drop"]
    l1_coeff = main_hypers["l1_coeff"]
    zero_threshold = main_hypers["zero_threshold"]
    epochs_drop = main_hypers["epochs_drop"]
    l2_coeff = main_hypers["l2_coeff"]
    momentum = main_hypers["momentum"]
    actionencoder_sizes = main_hypers["sizes"]


    print('==> Preparing dataset')
    trainloader, validloader, testloader = data_loader[0].get_loaders(batch_size=batch_size)

    if len(data_loader) > len(classes_list):
        print("==> Preparing evaluation dataset")
        eval_testloader = data_loader[-1].get_test_loader(batch_size=batch_size)
    else:
        eval_testloader = None

    print("==> Creating model")
    model = ActionEncoder(sizes=actionencoder_sizes)

    # Use Cuda
    if use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
        criterion = criterion.to("cuda")

    # initialize parameters
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.data.zero_()
        elif 'weight' in name:
            param.data.normal_(-0.05, 0.05)

    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000))

    errors = []

    for t, cls in enumerate(classes_list):

        print('\nTask: [%d | %d]\n' % (t + 1, len(classes_list)))

        if len(data_loader) >= len(classes_list) and t > 0:
            trainloader, validloader, testloader = data_loader[t].get_loaders(batch_size=batch_size)

        if t == 0:
            print("==> Learning")

            optimizer = optim.SGD(model.parameters(),
                                  lr=learning_rate,
                                  momentum=momentum,
                                  weight_decay=weight_decay
                                  )

            penalty = l1_penalty(coeff=l1_coeff)

            for epoch in range(max_epochs):

                # decay learning rate
                if not epochs_drop == 0 and (epoch + 1) % epochs_drop == 0:
                    learning_rate *= lr_drop
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))

                train_loss = trainAE(trainloader, model, criterion, optimizer=optimizer, penalty=penalty, use_cuda=use_cuda)
                # test_loss = trainAE(validloader, model, criterion, cl=t, test=True, penalty=penalty, use_cuda=use_cuda)

            if eval_testloader is not None:
                print("USING EVAL LOADER")
                err = error_function(model, eval_testloader, classes_list[:t+1])
            else:
                err = error_function(model, testloader, classes_list[:t+1])
            errors.append(err)
        else:
            # copy model
            model_copy = copy.deepcopy(model)

            print("==> Selective Retraining")

            # freeze all layers except the last one (last 2 parameters)
            params = list(model.parameters())
            for param in params[:-2]:
                param.requires_grad = False

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )

            penalty = l1l2_penalty(model=model, l1_coeff=l1_coeff, l2_coeff=l2_coeff)

            for epoch in range(max_epochs):

                # decay learning rate
                if not epochs_drop == 0 and (epoch + 1) % epochs_drop == 0:
                    learning_rate *= lr_drop
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))

                trainAE(trainloader, model, criterion, optimizer=optimizer, penalty=penalty, use_cuda=use_cuda)
                # trainAE(validloader, model, criterion, test=True, penalty=penalty, use_cuda=use_cuda)

            for param in model.parameters():
                param.requires_grad = True

            print("==> Selecting Neurons")
            hooks = select_neurons(model, t, zero_threshold, classes_list)

            print("==> Training Selected Neurons")

            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=1e-4
            )

            for epoch in range(max_epochs):

                # decay learning rate
                if not epochs_drop == 0 and (epoch + 1) % epochs_drop == 0:
                    learning_rate *= lr_drop
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))
                train_loss = trainAE(trainloader, model, criterion, optimizer=optimizer, use_cuda=use_cuda)
                # trainAE(validloader, model, criterion, test=True, penalty=penalty, use_cuda=use_cuda)

            # remove hooks
            for hook in hooks:
                hook.remove()

            # Note: In ICLR 18 paper the order of these steps are switched, we believe this makes more sense.
            # print("==> Splitting Neurons")
            model = split_neurons(model_copy, model, trainloader, validloader, criterion, split_train_new_hypers, use_cuda)

            # Could be train_loss or test_loss
            if train_loss > loss_threshold:
                # print("==> Dynamic Expansion")
                model = dynamic_expansion(expand_by_k, model, trainloader, validloader, criterion, de_train_new_hypers, use_cuda)

            #   add k neurons to all layers.
            #   optimize training on those weights with l1 regularization, and an addition cost based on
            #   the norm_2 of the weights of each individual neuron.
            #
            #   remove all neurons which have no weights that are non_zero
            #   save network.
            if eval_testloader is not None:
                print("USING EVAL LOADER")
                err = error_function(model, eval_testloader, classes_list[:t+1])
            else:
                err = error_function(model, testloader, classes_list[:t+1])
            errors.append(err)
            print(errors)

    if save_model is not None:
        filepath = os.path.join(os.path.dirname(__file__), "../../saved_models")
        filepath = os.path.join(filepath, save_model)
        torch.save({'state_dict': model.state_dict()}, filepath)

    return (model, errors)


def dynamic_expansion(expand_by_k, model, trainloader, validloader, criterion, de_train_new_hypers, cuda):
    sizes, weights, biases, hooks = {}, {}, {}, []
    modules = get_modules(model)
    for dict_key, module in modules.items():
        sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []

        prev_neurons = None
        for module_name, param in module:
            if 'bias' not in module_name:
                if len(sizes[dict_key]) == 0:
                    sizes[dict_key].append(param.data.shape[1])

                weights[dict_key].append(param.data)
                sizes[dict_key].append(param.data.shape[0] + expand_by_k)

                # Register hook to freeze param
                active_neurons = [False] * (param.data.shape[0] - expand_by_k)
                active_neurons.extend([True] * expand_by_k)

                if prev_neurons is None:
                    prev_neurons = [True] * param.data.shape[0]

                hook = param.register_hook(freeze_hook(prev_neurons, active_neurons))
                hooks.append(hook)

                # Pushes current set of neurons to next.
                prev_neurons = active_neurons

            elif 'bias' in module_name:
                biases[dict_key].append(param.data)
                hook = param.register_hook(freeze_hook(None, active_neurons, bias=True))
                hooks.append(hook)

            else:
                raise LookupError()

        if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
            sizes[dict_key][-1] -= expand_by_k

    # From here, everything taken from DE. #
    return train_new_neurons(model, modules, trainloader, validloader, criterion, sizes, weights, biases, hooks, de_train_new_hypers, cuda)


def get_modules(model):
    modules = {}

    for name, param in model.named_parameters():
        module = name[0: name.index('.')]
        if module not in modules.keys():
            modules[module] = []
        modules[module].append((name, param))

    return modules


def select_neurons(model, task, zero_threshold, classes_list):
    modules = get_modules(model)

    prev_active = [True] * len(classes_list)
    prev_active[task] = False

    action_hooks, prev_active = gen_hooks(modules['action'], zero_threshold, prev_active)
    encoder_hooks, _ = gen_hooks(modules['encoder'], zero_threshold, prev_active)

    hooks = action_hooks + encoder_hooks
    return hooks


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

    # for nr, (sel, neurons) in enumerate(reversed(selected)):
    #     print("layer %d: %d / %d" % (nr + 1, sel, neurons))

    return hooks, prev_active


def split_neurons(old_model, new_model, trainloader, validloader, criterion, split_train_new_hypers, cuda=False):
    sizes, weights, biases, hooks = {}, {}, {}, []

    suma = 0

    drift_threshold = split_train_new_hypers["drift_threshold"]

    old_modules = get_modules(old_model)
    new_modules = get_modules(new_model)
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
                    old_layers[weight_index].append((old_biases[weight_index].data[j], old_param.data[j], old_param, old_biases[weight_index]))
                    new_layers[weight_index].append((new_biases[weight_index].data[j], new_weights, new_param, new_biases[weight_index]))
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

                if drift > drift_threshold:
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

            hook = new_neurons[0][2].register_hook(freeze_hook(prev_neurons, active_weights))  # All neurons belong to same param.
            hooks.append(hook)
            hook = new_neurons[0][3].register_hook(freeze_hook(None, active_weights, bias=True))
            hooks.append(hook)

            # Push current layer to next.
            prev_neurons = active_weights

            if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
                sizes[dict_key][-1] -= len(append_to_end_weights)

    # How many split?
    # print("# Number of neurons split: %d" % suma)

    # Be efficient
    if suma == 0:
        return new_model

    # From here, everything taken from DE. #
    return train_new_neurons(new_model, new_modules, trainloader, validloader, criterion, sizes, weights, biases, hooks, split_train_new_hypers, cuda)


def train_new_neurons(model, modules, trainloader, validloader, criterion, sizes, weights, biases, hooks, hypers, cuda=False):
    # Get params
    learning_rate = hypers["learning_rate"]
    max_epochs = hypers["max_epochs"]
    lr_drop = hypers["lr_drop"]
    l1_coeff = hypers["l1_coeff"]
    zero_threshold = hypers["zero_threshold"]
    epochs_drop = hypers["epochs_drop"]
    l2_coeff = hypers["l2_coeff"]
    momentum = hypers["momentum"]

    # TODO: Make module generation dynamic
    new_model = ActionEncoder(sizes, oldWeights=weights, oldBiases=biases)

    optimizer = optim.SGD(
        new_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=1e-4
    )

    learning_rate = lr_drop

    for epoch in range(max_epochs):

        # decay learning rate
        if epochs_drop > 0 and (epoch + 1) % epochs_drop == 0:
            learning_rate *= lr_drop
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))

        penalty = l1l2_penalty(l1_coeff, l2_coeff, model)
        trainAE(trainloader, new_model, criterion, penalty=penalty,  optimizer=optimizer, use_cuda=cuda)

    # Remove hooks
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
        for i, weights in enumerate(added_neurons):
            new_sizes[name2].append(sizes[name2][i + 1] + len(weights))

    return ActionEncoder(new_sizes, oldWeights=new_weights, oldBiases=new_biases)


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
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()
        if mask2 is not None:
            self.mask2 = torch.Tensor(mask2).long().nonzero().view(-1).numpy()
        self.bias = bias

    def __call__(self, grad):
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
