from __future__ import print_function, absolute_import

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

__all__ = ['accuracy', 'calc_avg_AUROC', 'AUROC']


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


def calc_avg_AE_AUROC(model, batchloader, all_classes, classes, use_cuda, num_classes = 10):
    """Calculates average of the AUROC for the autoencoder
    """
    sum_targets = torch.cuda.LongTensor() if use_cuda else torch.LongTensor()
    sum_outputs = torch.cuda.FloatTensor() if use_cuda else torch.FloatTensor()

    for idx, (input, target) in enumerate(batchloader):
        target = target[:, target.size()[1] - len(all_classes):]
        target = label_binarize(target, all_classes)
        input = Variable(input)
        model.phase = "ACTION"
        output = model(input).data

        target = torch.LongTensor(target)
        sum_targets = torch.cat((sum_targets, target), 0)
        sum_outputs = torch.cat((sum_outputs, output), 0)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in classes:
        fpr[i], tpr[i], _ = roc_curve(sum_targets[:, i], sum_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(np.ravel(sum_targets.numpy()), np.ravel(sum_outputs.numpy()))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc


def calc_acc(model, batchloader, all_classes, cls):

    binary_targets = []
    binary_output = []

    for idx, (inp, target) in enumerate(batchloader):
        target = target[:, target.size()[1] - len(all_classes):]
        target = target.numpy()

        # transform into binary classification:
        for y_true in target:
            binary_y = argmax(y_true)
            binary_targets.append(binary_y)

        inp = Variable(inp)
        model.phase = "ACTION"
        output = model(inp)
        # output = torch.nn.functional.softmax(output, dim=0)
        output = output.data.numpy()

        for y_score in output:
            binary_y = argmax(y_score)
            binary_output.append(binary_y)

    correctly_classified = 0
    for i in range(len(binary_targets)):
        if binary_targets[i] == binary_output[i]:
            correctly_classified += 1

    cr = float(correctly_classified)/float(len(binary_targets))

    return {"Classification Rate": cr}


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
