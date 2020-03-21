from __future__ import print_function
from utils.datasets import load_AE_MNIST
from models import ActionEncoder
from utils.train import trainAE
from utils.eval import calc_avg_AE_AUROC
from utils import l1_penalty, l2_penalty, l1l2_penalty

import random
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

# Non-ML Hyperparams
ALL_CLASSES = range(10)
NUM_WORKERS = 0
CUDA = False
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)


def main_ae(main_hypers=None, split_train_new_hypers=None, de_train_new_hypers=None):

    # Default hypers for training
    learning_rate = 0.2
    batch_size = 256
    loss_threshold = 1e-2
    expand_by_k = 10
    max_epochs = 10
    weight_decay = 0
    lr_drop = 0.5
    l1_coeff = 1e-10
    zero_threshold = 1e-4
    epochs_drop = 10
    l2_coeff = 1e-10
    momentum = 0.0

    if main_hypers is not None:
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

    if split_train_new_hypers is None:
        split_train_new_hypers = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "lr_drop": lr_drop,
            "epochs_drop": epochs_drop,
            "max_epochs": max_epochs,
            "l1_coeff": l1_coeff,
            "l2_coeff": l2_coeff,
            "zero_threshold": zero_threshold,
            "drift_threshold": 0.02
        }

    if de_train_new_hypers is None:
        de_train_new_hypers = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "lr_drop": lr_drop,
            "epochs_drop": epochs_drop,
            "max_epochs": max_epochs,
            "l1_coeff": l1_coeff,
            "l2_coeff": l2_coeff,
            "zero_threshold": zero_threshold,
        }

    print('==> Preparing dataset')
    trainloader, validloader, testloader = load_AE_MNIST(batch_size=batch_size, num_workers=NUM_WORKERS)

    print("==> Creating model")
    model = ActionEncoder()

    if CUDA:
        model = model.cuda()
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    # initialize parameters
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.data.zero_()
        elif 'weight' in name:
            param.data.normal_(0, 0.005)

    print('    Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000))

    criterion = nn.BCELoss()

    CLASSES = []
    AUROCs = []

    for t, cls in enumerate(ALL_CLASSES):

        print('\nTask: [%d | %d]\n' % (t + 1, len(ALL_CLASSES)))

        CLASSES.append(cls)

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

                train_loss = trainAE(trainloader, model, criterion, optimizer=optimizer, penalty=penalty, use_cuda=CUDA)
                test_loss = trainAE(validloader, model, criterion, test=True, penalty=penalty, use_cuda=CUDA)

                # save model
                # is_best = test_loss < best_loss
                # best_loss = min(test_loss, best_loss)
                # save_checkpoint({'state_dict': model.state_dict()}, CHECKPOINT, is_best)

                suma = 0
                for p in model.parameters():
                    p = p.data.cpu().numpy()
                    suma += (abs(p) < zero_threshold).sum()
                print("Number of zero weights: %d" % (suma))

        else:
            # copy model
            model_copy = copy.deepcopy(model)

            print("==> Selective Retraining")

            # Solve Eq.3

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

            penalty = l1_penalty(coeff=l1_coeff)

            for epoch in range(max_epochs):

                # decay learning rate
                if not epochs_drop == 0 and (epoch + 1) % epochs_drop == 0:
                    learning_rate *= lr_drop
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))

                trainAE(trainloader, model, criterion, optimizer=optimizer, penalty=penalty,
                      use_cuda=CUDA)
                trainAE(validloader, model, criterion, test=True, penalty=penalty, use_cuda=CUDA)

            for param in model.parameters():
                param.requires_grad = True

            print("==> Selecting Neurons")
            hooks = select_neurons(model, t, zero_threshold)

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

                train_loss = trainAE(trainloader, model, criterion, optimizer=optimizer,
                                   use_cuda=CUDA)
                test_loss = trainAE(validloader, model, criterion, test=True, use_cuda=CUDA)

                # save model
                # is_best = test_loss < best_loss
                # best_loss = min(test_loss, best_loss)
                # save_checkpoint({'state_dict': model.state_dict()}, CHECKPOINT, is_best)

            # remove hooks
            for hook in hooks:
                hook.remove()

            # Note: In ICLR 18 paper the order of these steps are switched, we believe this makes more sense.
            print("==> Splitting Neurons")
            model = split_neurons(model_copy, model, trainloader, validloader, split_train_new_hypers)

            # Could be train_loss or test_loss
            if train_loss > loss_threshold:
                print("==> Dynamic Expansion")
                model = dynamic_expansion(expand_by_k, model, trainloader, validloader, de_train_new_hypers)

            #   add k neurons to all layers.
            #   optimize training on those weights with l1 regularization, and an addition cost based on
            #   the norm_2 of the weights of each individual neuron.
            #
            #   remove all neurons which have no weights that are non_zero
            #   save network.

        print("==> Calculating AUROC")

        # filepath_best = os.path.join(CHECKPOINT, "best.pt")
        # checkpoint = torch.load(filepath_best)
        # model.load_state_dict(checkpoint['state_dict'])

        auroc = calc_avg_AE_AUROC(model, testloader, ALL_CLASSES, CLASSES, CUDA)

        print('AUROC: {}'.format(auroc))

        AUROCs.append(auroc)

    print('\nAverage Per-task Performance over number of tasks')
    for i, p in enumerate(AUROCs):
        print("%d: %f" % (i + 1, p[i]))

    micros = [x["micro"] for x in AUROCs]

    return micros


def dynamic_expansion(expand_by_k, model, trainloader, validloader, de_train_new_hypers):
    sizes, weights, biases, hooks = {}, {}, {}, []

    modules = get_modules(model)
    for dict_key, module in modules.items():
        sizes[dict_key], weights[dict_key], biases[dict_key] = [], [], []
        for module_name, param in module:
            if 'bias' not in module_name:
                if len(sizes[dict_key]) == 0:
                    sizes[dict_key].append(param.data.shape[1])

                weights[dict_key].append(param.data)
                sizes[dict_key].append(param.data.shape[0] + expand_by_k)

                # Register hook to freeze param
                active_weights = [False] * (param.data.shape[0] - expand_by_k)
                active_weights.extend([True] * expand_by_k)
                hook = param.register_hook(freeze_hook(active_weights))
                hooks.append(hook)

            elif 'bias' in module_name:
                biases[dict_key].append(param.data)

            else:
                raise LookupError()

        if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
            sizes[dict_key][-1] -= expand_by_k

    # From here, everything taken from DE. #
    return train_new_neurons(model, modules, trainloader, validloader, sizes, weights, biases, hooks, de_train_new_hypers)


def get_modules(model):
    modules = {}

    for name, param in model.named_parameters():
        module = name[0: name.index('.')]
        if module not in modules.keys():
            modules[module] = []
        modules[module].append((name, param))

    return modules


def select_neurons(model, task, zero_threshold):
    modules = get_modules(model)

    prev_active = [True] * 10
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
                if weight > zero_threshold:
                    # mark connected neuron as active
                    active[y] = False

        h = layer.register_hook(active_grads_hook(prev_active, active))

        hooks.append(h)
        prev_active = active

        selected.append((y_size - sum(active), y_size))

    for nr, (sel, neurons) in enumerate(reversed(selected)):
        print("layer %d: %d / %d" % (nr + 1, sel, neurons))

    return hooks, prev_active


def split_neurons(old_model, new_model, trainloader, validloader, split_train_new_hypers):
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
                    old_layers[weight_index].append((old_biases[weight_index].data[j], old_param.data[j], old_param))
                    new_layers[weight_index].append((new_biases[weight_index].data[j], new_weights, new_param))
                weight_index += 1

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
            hook = new_neurons[0][2].register_hook(freeze_hook(active_weights))  # All neurons belong to same param.
            hooks.append(hook)

            if dict_key in sizes.keys() and len(sizes[dict_key]) > 0:
                sizes[dict_key][-1] -= len(append_to_end_weights)

    # How many split?
    print("# Number of neurons split: %d" % suma)

    # Be efficient
    if suma == 0:
        return new_model

    # From here, everything taken from DE. #
    return train_new_neurons(new_model, new_modules, trainloader, validloader, sizes, weights, biases, hooks, split_train_new_hypers)


def train_new_neurons(model, modules, trainloader, validloader, sizes, weights, biases, hooks, hypers):
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
    criterion = nn.BCELoss()

    for epoch in range(max_epochs):

        # decay learning rate
        if (epoch + 1) % epochs_drop == 0:
            learning_rate *= lr_drop
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        print('Epoch: [%d | %d]' % (epoch + 1, max_epochs))

        penalty = l1l2_penalty(l1_coeff, l2_coeff, model)
        train_loss = trainAE(trainloader, new_model, criterion,
                             penalty=penalty,
                             optimizer=optimizer, use_cuda=CUDA)
        test_loss = trainAE(validloader, new_model, criterion, penalty=penalty, test=True,
                            use_cuda=CUDA)

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
    def __init__(self, active_weights):
        self.active_weights = active_weights

    def __call__(self, grad):

        grad_clone = grad.clone()

        for i in range(grad.shape[0]):
            if not self.active_weights[i]:
                grad_clone[i] = 0

        return grad_clone


class active_grads_hook(object):

    def __init__(self, mask1, mask2):
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()
        self.mask2 = torch.Tensor(mask2).long().nonzero().view(-1).numpy()

    def __call__(self, grad):

        grad_clone = grad.clone()
        if self.mask1.size:
            grad_clone[self.mask1, :] = 0
        if self.mask2.size:
            grad_clone[:, self.mask2] = 0
        return grad_clone


if __name__ == '__main__':
    main_ae()
