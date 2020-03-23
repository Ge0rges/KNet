import time
import os
import shutil
import random

import torch
from torch.autograd import Variable
from progress.bar import Bar

from .misc import AverageMeter

__all__ = ['train', 'save_checkpoint', 'l1_penalty', 'l2_penalty', 'l1l2_penalty', 'trainAE']

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def one_hot(targets, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in classes:
            targets_onehot[i][classes.index(t)] = 1
    return targets_onehot


def train(batchloader, model, criterion, all_classes, classes, optimizer = None, penalty = None, test = False, use_cuda = False, freeze=None):
    # switch to train or evaluate mode
    if test:
        model.eval()
    else:
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if test:
        bar = Bar('Testing', max=len(batchloader))
    else:
        bar = Bar('Training', max=len(batchloader))

    for batch_idx, (inputs, targets) in enumerate(batchloader):

        # measure data loading time
        data_time.update(time.time() - end)

        # convert labels into one hot vectors
        targets_onehot = one_hot(targets, classes)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_onehot = targets_onehot.cuda()

        inputs = Variable(inputs)
        targets_onehot = Variable(targets_onehot)

        # compute output
        outputs = model(inputs)

        # calculate loss
        loss = 0
        for i, cls in enumerate(classes):
            loss = loss + criterion(outputs[:, all_classes.index(cls)], targets_onehot[:, i])
        if penalty is not None:
            loss = loss + penalty(model)

        # record loss
        losses.update(loss.data[0], inputs.size(0))

        if not test:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            if freeze:
                freeze(model)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(batchloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    return losses.avg


def trainAE(batchloader, model, criterion, optimizer=None, penalty=None, test=False, use_cuda=False):
    # switch to train or evaluate mode
    if test:
        model.eval()
        bar = Bar('Testing', max=len(batchloader))
    else:
        model.train()
        bar = Bar('Training', max=len(batchloader))

    # Progress bar stuff
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(batchloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        # compute output
        model.float()
        model.phase = "GENERATE"
        generate_output = model(inputs)

        model.phase = "ACTION"
        action_output = model(inputs)

        # calculate loss
        generate_loss = torch.nn.MSELoss()(generate_output, targets[:, :generate_output.size()[1]])
        action_loss = criterion(action_output, targets[:, generate_output.size()[1]:])

        if penalty is not None:
            generate_loss = generate_loss + penalty(model)
            action_loss = action_loss + penalty(model)

        total_loss = action_loss + generate_loss

        # record loss
        losses.update(total_loss.data[0], inputs.size(0))

        if not test:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Gen Loss: {gen_loss: .4f} | Action Loss: {action_loss: .4f}'.format(
            batch=batch_idx + 1,
            size=len(batchloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            loss=losses.avg,
            gen_loss=float(generate_loss),
            action_loss=float(action_loss)
        )
        bar.next()

    bar.finish()
    return losses.avg


def save_checkpoint(state, path, is_best = False):
    filepath = os.path.join(path, "last.pt")
    torch.save(state, filepath)
    if is_best:
        filepath_best = os.path.join(path, "best.pt")
        shutil.copyfile(filepath, filepath_best)


class l2_penalty(object):
    def __init__(self, model, coeff=5e-2):
        self.old_model = model
        self.coeff = coeff

    def __call__(self, new_model):
        # l2 = torch.nn.MSELoss()
        # return l2(self.old_model, self.new_model)*self.coeff

        penalty = 0
        for ((name1, param1), (name2, param2)) in zip(self.old_model.named_parameters(), new_model.named_parameters()):
            if name1 != name2 or param1.shape != param2.shape:
                raise Exception("model parameters do not match!")

            # get only weight parameters
            if 'bias' not in name1:
                diff = param1 - param2
                penalty = penalty + diff.norm(2)

        return self.coeff * penalty


class l1_penalty(object):
    def __init__(self, coeff=5e-2):
        self.coeff = coeff

    def __call__(self, model):
        penalty = 0
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                penalty = penalty + param.norm(1)
        return self.coeff * penalty


class l1l2_penalty(object):
    def __init__(self, l1_coeff, l2_coeff, model):
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.old_model = model

    def __call__(self, new_model):
        return self.l1(new_model) + self.l2(new_model)

    def l1(self, new_model):
        penalty = 0
        for (name, param) in new_model.named_parameters():
            if 'bias' not in name:
                penalty += param.norm(1)

        return self.l1_coeff * penalty

    def l2(self, new_model):
        penalty = 0
        for ((name1, param1), (name2, param2)) in zip(self.old_model.named_parameters(), new_model.named_parameters()):
            if 'bias' in name1:
                continue

            for i in range(param1.data.shape[0], param2.data.shape[0]):
                row = torch.zeros(param2.data.shape[1])
                for j in range(param2.data.shape[1]):
                    row[j] = param2.data[i, j]
                penalty += row.norm(2)

        return self.l2_coeff * penalty
