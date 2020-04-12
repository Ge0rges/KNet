import time
import os
import shutil
import random
import torch
import numpy as np

from torch.autograd import Variable
from progress.bar import Bar
from src.utils.misc import AverageMeter

def one_hot(targets, cl):
    targets_onehot = torch.zeros(targets.shape)
    for i, t in enumerate(targets):
        if int(targets[i][cl]) == 1:
            targets_onehot[i][cl] = 1
    return targets_onehot


def trainAE(batchloader, model, criterion, optimizer=None, penalty=None, test=False, use_cuda=False, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

        inputs = Variable(inputs)
        targets = Variable(targets)

        # compute output
        model.phase = "GENERATE"
        generate_output = model(inputs)

        model.phase = "ACTION"
        action_output = model(inputs)

        generate_targets = targets[:, :generate_output.size()[1]]
        action_target = targets[:, generate_output.size()[1]:]

        # if cls is not None:
        #     action_one_hot = one_hot(action_target, cls)
        #     action_target = Variable(action_one_hot)

        # calculate loss
        mse_loss = torch.nn.MSELoss()
        generate_loss = mse_loss(generate_output, generate_targets)
        action_loss = criterion(action_output, action_target)

        if penalty is not None:
            generate_loss = generate_loss + penalty(model)
            action_loss = action_loss + penalty(model)

        total_loss = action_loss + generate_loss# TODO: add back gen in phases

        # record loss
        losses.update(total_loss.data.item(), inputs.size(0))

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


class ResourceConstrainingPenalty:
    def __init__(self, coeff=1, bytes_available=16000000000):
        self.coeff = coeff
        self.resources = bytes_available

    def __call__(self, model):
        penalty = 0
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                penalty += -np.abs(1/param) + self.resources
        return self.coeff * penalty
