import time
import torch

from torch.utils.data import DataLoader
from progress.bar import Bar
from src.utils.misc import AverageMeter


def train(batch_loader: DataLoader, model: torch.nn.Module, criterion, optimizer, penalty, testing: bool, device: torch.device, tasks: [int], seen_tasks: [int]):

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
        # model.phase = "GENERATE"
        # generate_output = model(inputs)
        # generate_targets = inputs according to Lucas

        model.phase = "ACTION"

        action_output = model(inputs)

        action_output = action_output[:, tasks+seen_tasks]
        action_target = action_target[:, tasks+seen_tasks]

        # For seen tasks, set target = output
        # action_target[:, seen_tasks] = action_output.clone().detach()[:, seen_tasks]
        if len(seen_tasks) > 0:
            seen_outputs = action_output.clone().detach()[:, seen_tasks]
            seen_outputs_current_tasks = torch.cat([seen_outputs, action_target[:, tasks]], dim=1)
            _, indices = torch.max(seen_outputs_current_tasks, 1)
            binary_seen_outputs = torch.nn.functional.one_hot(indices, num_classes=len(seen_tasks)+len(tasks))

            action_target[:, seen_tasks] = binary_seen_outputs.float()[:, seen_tasks]

        # encoder_loss = torch.nn.BCELoss()
        penalty_val = penalty(model, device) if penalty else 0

        # generate_loss = encoder_loss(generate_output, generate_targets) + penalty_val
        action_loss = criterion(action_output, action_target) + penalty_val
        total_loss = action_loss  # + generate_loss  # TODO: add back gen in phases

        # Record loss
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
        self.l2_coeff = l2_coeff  # Used by DENTrainer to set SGD weight_decay.

    def __call__(self, new_model, device):
        return self.l1(new_model, device) + self.l2(new_model, device)

    def l1(self, model, device):
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        if self.l1_coeff == 0:  # Be efficient
            return l1_reg

        modules = model.get_used_keys()

        for name, param in model.named_parameters():
            module = name[0: name.index('.')]
            if 'weight' in name and module in modules:
                l1_reg = l1_reg + torch.norm(param, 1)

        return torch.mul(self.l1_coeff, l1_reg)

    def l2(self, model, device):
        l2_reg = torch.tensor(0., requires_grad=True).to(device)
        if self.l2_coeff == 0:  # Be efficient
            return l2_reg

        modules = model.get_used_keys()

        for name, param in model.named_parameters():
            module = name[0: name.index('.')]
            if 'weight' in name and module in modules:
                l2_reg = l2_reg + torch.norm(param, 2)

        return torch.mul(self.l2_coeff, l2_reg)
