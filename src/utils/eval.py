from __future__ import print_function, absolute_import

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

__all__ = ['accuracy', 'calc_avg_AUROC', 'AUROC', 'calc_avg_AE_AUROC']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calc_avg_AUROC(model, batchloader, all_classes, classes, use_cuda, num_classes = 10):
    """Calculates average of the AUROC for selected classes in the dataset
    """
    sum_targets = torch.cuda.LongTensor() if use_cuda else torch.LongTensor()
    sum_outputs = torch.cuda.FloatTensor() if use_cuda else torch.FloatTensor()

    for batch_idx, (inputs, targets) in enumerate(batchloader):

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        outputs = model(inputs).data

        sum_targets = torch.cat((sum_targets, targets), 0)
        sum_outputs = torch.cat((sum_outputs, outputs), 0)

    sum_area = 0
    for cls in classes:
        scores = sum_outputs[:, all_classes.index(cls)]
        sum_area += AUROC(scores.cpu().numpy(), (sum_targets == cls).cpu().numpy())

    return (sum_area / len(classes))


def calc_avg_AE_AUROC(model, batchloader, all_classes, cls, use_cuda, num_classes = 10):
    """Calculates average of the AUROC for the autoencoder
    """

    # TODO: still doesn't work

    binary_targets = []
    binary_output = []

    for idx, (inp, target) in enumerate(batchloader):
        target = target[:, target.size()[1] - len(all_classes):]

        # transform into binary classification:
        for y_true in target:
            binary_y = argmax(y_true)
            if binary_y != cls:
                binary_y = 0
            else:
                binary_y = 1
            binary_targets.append(binary_y)

        inp = Variable(inp)
        model.phase = "ACTION"
        output = model(inp).data

        for y_score in output:
            binary_y = argmax(y_score)
            if binary_y != cls:
                binary_y = 0
            else:
                binary_y = 1
            binary_output.append(binary_y)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(binary_targets)):
        if binary_targets[i] == 1 and binary_output[i] == 1:
            tp += 1
        elif binary_targets[i] == 0 and binary_output[i] == 0:
            tn += 1
        elif binary_targets[i] == 0 and binary_output[i] == 1:
            fp += 1
        elif binary_targets[i] == 1 and binary_output[i] == 0:
            fn += 1
        else:
            print("ERROR WHILE COMPUTING AUROC, BINARY TARGETS/OUTPUTS NOT SETUP PROPERLY")

    print("tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)
    if tp == 0 and fn == 0:
        if tp != 0:
            tpr = 1
        else:
            tpr = 0
    else:
        tpr = float(float(tp)/(float(tp) + float(fn)))
    if fp == 0 and tn == 0:
        if fp != 0:
            fpr = 1
        else:
            fpr = 0
    else:
        fpr = float(float(fp)/(float(fp) + float(tn)))

    cr = float(tp + tn)/float(len(binary_targets))

    return {"False Positive Rate": fpr, "True Positive Rate": tpr, "Classification Rate": cr}


def argmax(y):
    """No argmax function for pytorch in 0.3.1 so implementing my own"""
    l = len(y)
    max = 0
    for i in range(1, l):
        if y[max] < y[i]:
            max = i
    return max


def AUROC(scores, targets):
    """Calculates the Area Under the Curve.
    Args:
        scores: Probabilities that target should be possitively classified.
        targets: 0 for negative, and 1 for positive examples.
    """
    # case when number of elements added are 0
    if scores.shape[0] == 0:
        return 0.5

    # sorting the arrays
    scores, sortind = torch.sort(torch.from_numpy(
        scores), dim=0, descending=True)
    scores = scores.numpy()
    sortind = sortind.numpy()

    # creating the roc curve
    tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
    fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

    for i in range(1, scores.size + 1):
        if targets[sortind[i - 1]] == 1:
            tpr[i] = tpr[i - 1] + 1
            fpr[i] = fpr[i - 1]
        else:
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1] + 1

    tpr /= (targets.sum() * 1.0)
    fpr /= ((targets - 1.0).sum() * -1.0)

    # calculating area under curve using trapezoidal rule
    n = tpr.shape[0]
    h = fpr[1:n] - fpr[0:n - 1]
    sum_h = np.zeros(fpr.shape)
    sum_h[0:n - 1] = h
    sum_h[1:n] += h
    area = (sum_h * tpr).sum() / 2.0

    return area
