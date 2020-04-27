import time
import torch

from torch.utils.data import DataLoader
from progress.bar import Bar
from src.utils.misc import AverageMeter


def train(batch_loader: DataLoader, model: torch.nn.Module, criterion, optimizer, penalty, testing: bool, device: torch.device, tasks: [int]):

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

    for batch_idx, (inputs, action_target) in enumerate(batch_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        action_target = action_target.to(device)

        # compute output
        with torch.autograd.detect_anomaly():
            # model.phase = "GENERATE"
            # generate_output = model(inputs)
            # generate_targets = inputs according to Lucas

            model.phase = "ACTION"
            action_output = model(inputs)

            action_output = action_output[:, tasks]
            action_target = action_target[:, tasks]

            # encoder_loss = torch.nn.BCELoss()
            penalty_val = penalty(model) if penalty else 0

            # generate_loss = encoder_loss(generate_output, generate_targets) + penalty_val
            action_loss = criterion(action_output, action_target) + penalty_val

            total_loss = action_loss #+ generate_loss  # TODO: add back gen in phases

            # record loss
            losses.update(total_loss.item(), inputs.size(0))

            if not testing:
                # Compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # Measure elapsed time
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


class L1L2Penalty:
    """
    Does not account biases. See paper.
    """
    def __init__(self, l1_coeff, l2_coeff):
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.old_model = None

    def __call__(self, new_model):
        return self.l1(new_model) + self.l2(new_model)

    def l1(self, new_model):
        if self.l1_coeff == 0:  # Be efficient
            return 0

        penalty = 0
        for (name, param) in new_model.named_parameters():
            if 'bias' not in name:
                penalty += param.norm(1)

        return self.l1_coeff * penalty

    def l2(self, new_model):
        if self.l2_coeff == 0:  # Be efficient
            return 0

        assert self.old_model is not None

        penalty = 0
        for ((name1, param1), (name2, param2)) in zip(self.old_model.named_parameters(), new_model.named_parameters()):
            if 'bias' in name1:
                continue

            for i in range(param1.shape[0], param2.shape[0]):
                row = torch.zeros(param2.shape[1])
                for j in range(param2.shape[1]):
                    row[j] = param2[i, j]
                penalty += row.norm(2)

        return self.l2_coeff * penalty

