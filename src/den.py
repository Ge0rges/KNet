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
MAX_EPOCHS = 1
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

            # Could be train_loss or test_loss
            if train_loss > LOSS_THRESHOLD:
                print("==> Dynamic Expansion")
                dynamic_expansion(model, trainloader, validloader, cls, t)

            #   add k neurons to all layers.
            #   optimize training on those weights with l1 regularization, and an addition cost based on
            #   the norm_2 of the weights of each individual neuron.
            #
            #   remove all neurons which have no weights that are non_zero
            #   save network.

            print("==> Splitting Neurons")
            model = split_neurons(model_copy, model)

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
    # k = EXPAND_BY_K

    layers = []
    for name, param in model.named_parameters():
        if 'bias' not in name:
            layers.append(param)

    sizes = []
    weights = []
    sizes.append(layers[0].data.shape[1])
    for layer in layers:
        weights.append(layer.data)
        sizes.append(layer.data.shape[0] + EXPAND_BY_K)
    sizes[-1] -= EXPAND_BY_K

    # TODO: Make module generation dynamic.
    new_model = FeedForward(sizes, oldWeights=weights)

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

    new_weights = []
    weight_indexes = []
    added_neurons = []
    for ((name1, param1), (name2, param2)) in zip(model.named_parameters(), new_model.named_parameters()):
        if 'bias' in name1:
            continue

        for i in range(param1.data.shape[0]):
            row = []
            for j in range(param1.data.shape[1]):
                row.append(float(param2.data[i, j]))
            new_weights.append(row)

        for j in range(param1.data.shape[1], param2.data.shape[1]):
            for i in range(param1.data.shape[0]):
                if j in weight_indexes:
                    new_weights[i].append(float(param2.data[i, j]))

        weight_indexes = []
        for i in range(param1.data.shape[0], param2.data.shape[0]):
            row = []
            if float(param2[i].norm(1)) > ZERO_THRESHOLD:
                weight_indexes.append(i)
                for j in range(param2.data.shape[1]):
                    row.append(float(param2.data[i, j]))
            new_weights.append(row)

        added_neurons.append(weight_indexes)

    new_sizes = [sizes[0]]
    index = 1
    for weights in added_neurons:
        new_sizes.append(sizes[index] + len(weights))
        index += 1

    # Merge sublists of same lentghs into a matrix, ie this is a list of matrices
    # TODO: Will definitely break if two or more consecutive layers have the same size
    three_d_weights = []
    size = -1
    for row in new_weights:
        if not size == len(row):
            size = len(row)
            three_d_weights.append([])

        if size == len(row):
            three_d_weights[-1].append(row)

    return FeedForward(new_sizes, oldWeights=three_d_weights)

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
                if (weight > ZERO_THRESHOLD):
                    # mark connected neuron as active
                    active[y] = False

        h = layer.register_hook(my_hook(prev_active, active))

        hooks.append(h)
        prev_active = active

        selected.append((y_size - sum(active), y_size))

    for nr, (sel, neurons) in enumerate(reversed(selected)):
        print("layer %d: %d / %d" % (nr + 1, sel, neurons))

    return hooks


def split_neurons(old_model, new_model):
    old_layers = []
    for name, param in old_model.named_parameters():
        if 'bias' not in name:
            old_layers.append(param)

    new_layers = []
    for name, param in new_model.named_parameters():
        if 'bias' not in name:
            new_layers.append(param)

    suma = 0
    sizes = []
    weights = []

    sizes.append(new_layers[0].data.shape[1])
    for old_layer, new_layer, layer_index in zip(old_layers, new_layers, range(len(new_layers))):

        for data1, data2, node_index in zip(old_layer.data, new_layer.data, range(len(new_layer.data))):
            diff = data1 - data2
            drift = diff.norm(2)

            if (drift > 0.02):
                suma += 1

                # Copy neuron i into i' (w' introduction of edges or i')
                # new_layer.data append data2
                # new_layer.data replace old data2 with data1
                reshaped_data2 = data2.unsqueeze(0)
                new_layer_data = torch.cat([new_layer.data, reshaped_data2], dim=0)
                new_layer_data[node_index] = data1
                new_layer.data = new_layer_data

                print("In layer %d split neuron %d" % (layer_index, node_index))

        sizes.append(new_layer.data.shape[0])
        weights.append(new_layer.data)

    print("# Number of neurons split: %d" % (suma))

    return FeedForward(sizes, oldWeights=weights)


if __name__ == '__main__':
    main()
