import time
import torch
import numpy as np

from progress.bar import Bar
from src.utils.misc import AverageMeter
from src.utils.misc import DataloaderWrapper


def train(batch_loader: DataloaderWrapper, model: torch.nn.Module, criterion, optimizer, penalty, testing: bool, device: torch.device, tasks: [int]):

    # switch to train or evaluate mode
    if testing:
        model.eval()
        bar = Bar('Testing', max=len(batch_loader))
    else:
        model.train()
        bar = Bar('Training', max=len(batch_loader))

    # Progress bar stuff
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        with torch.autograd.detect_anomaly():
            # model.phase = "GENERATE"
            # generate_output = model(inputs)

            model.phase = "ACTION"
            action_output = model(inputs)
            generate_targets = targets[:, :model.input_size]
            action_target = targets[:, model.input_size:]

            # calculate loss
            optimizer.zero_grad()

            action_output = action_output[:, tasks]
            action_target = action_target[:, tasks]

            # encoder_loss = torch.nn.BCELoss()
            # generate_loss = encoder_loss(generate_output, generate_targets) + penalty(model)
            action_loss = criterion(action_output, action_target) + penalty(model)

            total_loss = action_loss #+ generate_loss  # TODO: add back gen in phases

            # record loss
            losses.update(total_loss.item(), inputs.size(0))

            if not testing:
                # compute gradient and do SGD step
                total_loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Gen Loss: {gen_loss: .4f} | Action Loss: {action_loss: .4f}'.format(
            batch=batch_idx + 1,
            size=len(batch_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            loss=losses.avg,
            gen_loss=float(0),
            action_loss=float(action_loss)
        )
        bar.next()

    bar.finish()
    return losses.avg


class l1l2_penalty:
    """
    Does not account biases. See paper.
    """
    def __init__(self, l1_coeff, l2_coeff, old_model):
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.old_model = old_model

    def __call__(self, new_model):
        return self.l1(new_model) + self.l2(new_model)

    def l1(self, new_model):
        penalty = 0
        for (name, param) in new_model.named_parameters():
            if 'bias' not in name:
                penalty += param.norm(1)

        return self.l1_coeff * penalty

    def l2(self, new_model):
        assert self.old_model is not None
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
