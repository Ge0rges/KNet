from __future__ import print_function

import os
import random
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models import FeedForward
from utils import *

# PATHS
CHECKPOINT = "./checkpoints/mnist-den"

# BATCH
BATCH_SIZE = 256
NUM_WORKERS = 4

# SGD
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0

# Step Decay
LR_DROP = 0.5
EPOCHS_DROP = 20

# MISC
MAX_EPOCHS = 20
CUDA = False

# L1 REGULARIZATION
L1_COEFF = 1e-5

# L2 REGULARIZATION
L2_COEFF = 1e-5

#LOSS_THRE
LOSS_THRESHOLD = 1e-2

# Dynamic Expansion
EXPAND_BY_K = 10

# weight below this value will be considered as zero
ZERO_THRESHOLD = 1e-4

# Classes
ALL_CLASSES = range(10)

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)


def main():
    # if not os.path.isdir(CHECKPOINT):
    #     os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    trainloader, validloader, testloader = load_MNIST(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    print("==> Creating model")
    model = FeedForward()

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
                                  lr=LEARNING_RATE,
                                  momentum=MOMENTUM,
                                  weight_decay=WEIGHT_DECAY
                                  )

            penalty = l1_penalty(coeff=L1_COEFF)
            best_loss = 1e10
            learning_rate = LEARNING_RATE
            # epochs = 10

            for epoch in range(MAX_EPOCHS):

                # decay learning rate
                if (epoch + 1) % EPOCHS_DROP == 0:
                    learning_rate *= LR_DROP
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, MAX_EPOCHS))

                train_loss = train(trainloader, model, criterion, ALL_CLASSES, [cls], optimizer=optimizer, penalty=penalty, use_cuda=CUDA)
                test_loss = train(validloader, model, criterion, ALL_CLASSES, [cls], test=True, penalty=penalty, use_cuda=CUDA)

                # save model
                # is_best = test_loss < best_loss
                # best_loss = min(test_loss, best_loss)
                # save_checkpoint({'state_dict': model.state_dict()}, CHECKPOINT, is_best)

                suma = 0
                for p in model.parameters():
                    p = p.data.cpu().numpy()
                    suma += (abs(p) < ZERO_THRESHOLD).sum()
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
                lr=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY
            )

            penalty = l1_penalty(coeff=L1_COEFF)
            best_loss = 1e10
            learning_rate = LEARNING_RATE

            for epoch in range(MAX_EPOCHS):

                # decay learning rate
                if (epoch + 1) % EPOCHS_DROP == 0:
                    learning_rate *= LR_DROP
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, MAX_EPOCHS))

                train(trainloader, model, criterion, ALL_CLASSES, [cls], optimizer=optimizer, penalty=penalty,
                      use_cuda=CUDA)
                train(validloader, model, criterion, ALL_CLASSES, [cls], test=True, penalty=penalty, use_cuda=CUDA)

            for param in model.parameters():
                param.requires_grad = True

            print("==> Selecting Neurons")
            hooks = select_neurons(model, t)

            print("==> Training Selected Neurons")

            optimizer = optim.SGD(
                model.parameters(),
                lr=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=1e-4
            )

            best_loss = 1e10
            learning_rate = LEARNING_RATE

            for epoch in range(MAX_EPOCHS):

                # decay learning rate
                if (epoch + 1) % EPOCHS_DROP == 0:
                    learning_rate *= LR_DROP
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                print('Epoch: [%d | %d]' % (epoch + 1, MAX_EPOCHS))

                train_loss = train(trainloader, model, criterion, ALL_CLASSES, [cls], optimizer=optimizer,
                                   use_cuda=CUDA)
                test_loss = train(validloader, model, criterion, ALL_CLASSES, [cls], test=True, use_cuda=CUDA)

                # save model
                # is_best = test_loss < best_loss
                # best_loss = min(test_loss, best_loss)
                # save_checkpoint({'state_dict': model.state_dict()}, CHECKPOINT, is_best)

            # remove hooks
            for hook in hooks:
                hook.remove()

            print("==> Splitting Neurons")
            model = split_neurons(model_copy, model, trainloader, validloader, cls)

            # Could be train_loss or test_loss
            if train_loss > LOSS_THRESHOLD:
                print("==> Dynamic Expansion")
                model = dynamic_expansion(model, trainloader, validloader, cls, t)

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

        auroc = calc_avg_AUROC(model, testloader, ALL_CLASSES, CLASSES, CUDA)

        print('AUROC: {}'.format(auroc))

        AUROCs.append(auroc)

    print('\nAverage Per-task Performance over number of tasks')
    for i, p in enumerate(AUROCs):
        print("%d: %f" % (i + 1, p))


class my_hook(object):

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


def dynamic_expansion(model, trainloader, validloader, cls, task):
    sizes, weights, biases, hooks = [], [], [], []
    for name, param in model.named_parameters():
        if 'bias' not in name:
            if len(sizes) == 0:
                sizes.append(param.data.shape[1])

            weights.append(param.data)
            sizes.append(param.data.shape[0] + EXPAND_BY_K)

            # Register hook to freeze param
            active_weights = [False] * (param.data.shape[0] - EXPAND_BY_K)
            active_weights.extend([True] * EXPAND_BY_K)
            hook = param.register_hook(freeze_hook(active_weights))
            hooks.append(hook)

        elif 'bias' in name:
            biases.append(param.data)

    sizes[-1] -= EXPAND_BY_K

    # TODO: Make module generation dynamic
    new_model = FeedForward(sizes, oldWeights=np.asarray(weights, dtype=object), oldBiases=np.asarray(biases, dtype=object))

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=1e-4
    )

    learning_rate = LR_DROP
    criterion = nn.BCELoss()

    for epoch in range(MAX_EPOCHS):

        # decay learning rate
        if (epoch + 1) % EPOCHS_DROP == 0:
            learning_rate *= LR_DROP
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        print('Epoch: [%d | %d]' % (epoch + 1, MAX_EPOCHS))

        penalty = l1l2_penalty(L1_COEFF, L2_COEFF, model)
        train_loss = train(trainloader, new_model, criterion, ALL_CLASSES, [cls], penalty=penalty, optimizer=optimizer, use_cuda=CUDA)
        test_loss = train(validloader, new_model, criterion, ALL_CLASSES, [cls], penalty=penalty, test=True, use_cuda=CUDA)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    new_biases = []
    new_weights = []
    weight_indexes = []
    added_neurons = []
    for ((name1, param1), (name2, param2)) in zip(model.named_parameters(), new_model.named_parameters()):
        if 'bias' in name1:
            new_layer = []

            # Copy over old bias
            for i in range(param1.data.shape[0]):
                new_layer.append(float(param2.data[i]))

            # Copy over incoming bias for new neuron for previous existing
            for i in range(param1.data.shape[0], param2.data.shape[0]):
                if float(param2[i].norm(1)) > ZERO_THRESHOLD:
                    new_layer.append(float(param2.data[i]))

            new_biases.append(new_layer)

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
                if float(param2[i].norm(1)) > ZERO_THRESHOLD:
                    weight_indexes.append(i)
                    for j in range(param2.data.shape[1]):
                        row.append(float(param2.data[i, j]))
                new_layer.append(row)

            new_weights.append(new_layer)
            added_neurons.append(weight_indexes)

    new_sizes = [sizes[0]]
    for i, weights in enumerate(added_neurons):
        new_sizes.append(sizes[i+1] + len(weights))

    return FeedForward(new_sizes, oldWeights=np.asarray(new_weights, dtype=object), oldBiases=np.asarray(new_biases, dtype=object))


def select_neurons(model, task):
    prev_active = [True] * len(ALL_CLASSES)
    prev_active[task] = False

    layers = []
    for name, param in model.named_parameters():
        if 'bias' not in name:
            layers.append(param)
    layers = reversed(layers)

    hooks = []
    selected = []

    for layer in layers:

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
                if weight > ZERO_THRESHOLD:
                    # mark connected neuron as active
                    active[y] = False

        h = layer.register_hook(active_grads_hook(prev_active, active))

        hooks.append(h)
        prev_active = active

        selected.append((y_size - sum(active), y_size))

    for nr, (sel, neurons) in enumerate(reversed(selected)):
        print("layer %d: %d / %d" % (nr + 1, sel, neurons))

    return hooks


def split_neurons(old_model, new_model, trainloader, validloader, cls):
    old_biases = []
    old_layers = []
    for name, param in old_model.named_parameters():
        if 'bias' not in name:
            old_layers.append(param)

        elif 'bias' in name:
            old_biases.append(param)

    new_biases = []
    new_layers = []
    for name, param in new_model.named_parameters():
        if 'bias' not in name:
            new_layers.append(param)

        elif 'bias' in name:
            new_biases.append(param)

    suma = 0
    sizes = []
    weights = []
    bias = []

    sizes.append(new_layers[0].data.shape[1])
    for old_layer_weights, new_layer_weights, old_layer_bias, new_layer_bias, layer_index in zip(old_layers, new_layers, old_biases, new_biases, range(len(new_layers))):  # For each layer

        # Don't split first and last layer
        if layer_index == 0 or layer_index == len(new_layers)-1:
            sizes.append(new_layer_weights.data.shape[0])
            weights.append(new_layer_weights.data)
            bias.append(new_layer_bias.data)
            continue

        for old_weights, new_weights, old_bias, new_bias, node_index in zip(old_layer_weights.data, new_layer_weights.data, old_layer_bias, new_layer_bias, range(len(new_layer_weights.data))):  # For each neuron
            diff = old_weights - new_weights
            drift = diff.norm(2)

            if drift > 0.02:
                suma += 1

                # Copy neuron i into i' (w' introduction of edges or i')
                # new_layer.data append data2
                # new_layer.data replace old data2 with data1
                reshaped_weight = new_weights.unsqueeze(0)
                new_layer_weights_data = torch.cat([new_layer_weights.data, reshaped_weight], dim=0)
                new_layer_weights_data[node_index] = old_weights
                new_layer_weights.data = new_layer_weights_data

                new_layer_bias_data = torch.cat([new_layer_bias.data, new_bias.data], dim=0)
                new_layer_bias_data[node_index] = old_bias.data[0]
                new_layer_bias.data = new_layer_bias_data

                print("In layer %d split neuron %d" % (layer_index, node_index))

        sizes.append(new_layer_weights.data.shape[0])
        weights.append(new_layer_weights.data)
        bias.append(new_layer_bias.data)

    print("# Number of neurons split: %d" % suma)

    # Be efficient.
    if suma == 0:
        return new_model

    # No need to pad, get_layer does that for us.
    w_n = np.asarray(weights, dtype=object)
    b_n = np.asarray(new_biases, dtype=object)

    # Train newly added nodes
    new_model = FeedForward(sizes, oldWeights=w_n, oldBiases=b_n)

    optimizer = optim.SGD(
        new_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=1e-4
    )

    criterion = nn.BCELoss()
    penalty = l1l2_penalty(L1_COEFF, L2_COEFF, new_model)

    train_loss = train(trainloader, new_model, criterion, ALL_CLASSES, [cls], penalty=penalty, optimizer=optimizer,
                       use_cuda=CUDA)
    test_loss = train(validloader, new_model, criterion, ALL_CLASSES, [cls], penalty=penalty, test=True, use_cuda=CUDA)

    return new_model


class freeze_hook(object):
    def __init__(self, active_weights):
        self.active_weights = active_weights

    def __call__(self, grad):

        grad_clone = grad.clone()

        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):

                if not self.active_weights[i, j]:
                    grad_clone[i, j] = 0

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
    main()
