import torch
import numpy as np

from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable
from src.utils.misc import one_vs_all_one_hot


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


def calc_avg_AUROC(model, batchloader, number_of_tasks, classes, device):
    """Calculates average of the AUROC for selected classes in the dataset
    """

    sum_targets = torch.LongTensor().to(device)
    sum_outputs = torch.FloatTensor().to(device)

    for batch_idx, (inputs, targets) in enumerate(batchloader):
        input = input.to(device)
        target = target.to(device)

        outputs = model(inputs)

        sum_targets = torch.cat((sum_targets, targets), 0)
        sum_outputs = torch.cat((sum_outputs, outputs), 0)

    sum_area = 0
    for cls in classes:
        scores = sum_outputs[:, cls]
        sum_area += AUROC(scores.cpu().numpy(), (sum_targets == cls).cpu().numpy())

    return (sum_area / len(classes))


def calc_avg_AE_AUROC(model, batchloader, number_of_tasks, number_of_tasks_trained, device):
    """Calculates average of the AUROC for the autoencoder
    """

    sum_targets = torch.LongTensor().to(device)
    sum_outputs = torch.FloatTensor().to(device)

    for idx, (input, target) in enumerate(batchloader):
        input = input.to(device)
        target = target.to(device)

        target = target[:, - number_of_tasks:]
        target = label_binarize(target, range(number_of_tasks))
        model.phase = "ACTION"
        output = model(input)

        target = torch.LongTensor(target).to(device)
        sum_targets = torch.cat((sum_targets, target), 0)
        sum_outputs = torch.cat((sum_outputs, output), 0)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # switching back to cpu for roc computation otherwise it breaks
    sum_targets = sum_targets.to('cpu')
    sum_outputs = sum_outputs.to('cpu')
    for i in range(number_of_tasks_trained):
        fpr[i], tpr[i], _ = roc_curve(sum_targets[:, i], sum_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(np.ravel(sum_targets.numpy()), np.ravel(sum_outputs.numpy()))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_tasks_trained)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(number_of_tasks_trained):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= number_of_tasks_trained

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc


def calc_avg_AE_band_error(model, batchloader, device):

    errors = []

    for idx, (input, target) in enumerate(batchloader):
        input = input.to(device)
        target = target.to(device)

        target = target[:, target.size()[1] - 2:]
        target = target.numpy()

        model.phase = "ACTION"
        output = model(input).numpy()

        errors.extend(np.abs((target - output)/target))

    errors = np.array(errors)

    alpha_error = np.mean(errors[:, 0])
    beta_error = np.mean(errors[:, 1])
    return {"alpha_error": alpha_error, "beta_error": beta_error}


def calculate_accuracy(confusion_matrix):
    assert confusion_matrix is not None
    return confusion_matrix.diag().sum()/confusion_matrix.sum()


def build_confusion_matrix(model, dataloader, number_of_tasks, tasks, device):
    untrained_tasks = list(range(number_of_tasks))
    for t in tasks:
        untrained_tasks.remove(t)

    matrix_size = number_of_tasks if len(untrained_tasks) == 0 else len(tasks) + 1

    confusion_matrix = torch.zeros(matrix_size, matrix_size).to(device)

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.phase = "ACTION"
        outputs = model(inputs)

        # Go from probabilities to classification
        _, indices = torch.max(outputs, 1)
        binary_outputs = torch.nn.functional.one_hot(indices, num_classes=number_of_tasks)
        binary_targets = targets

        # Create the column for all untrained tasks if necessary
        if len(untrained_tasks) > 0:
            trained_outputs = binary_outputs[:, tasks]
            trained_targets = binary_targets[:, tasks]

            untrained_outputs = torch.sum(binary_outputs[:, untrained_tasks], dim=1).unsqueeze(dim=1)
            untrained_targets = torch.sum(binary_targets[:, untrained_tasks], dim=1).unsqueeze(dim=1)

            binary_outputs = torch.cat([trained_outputs, untrained_outputs], dim=1)
            binary_targets = torch.cat([trained_targets, untrained_targets], dim=1)

        for t, p in zip(binary_targets.view(-1), binary_outputs.view(-1)):
            confusion_matrix[p.long(), t.long()] += 1

    return confusion_matrix


def build_multilabel_confusion_matrix(model, dataloader, tasks, device):
    raise NotImplementedError

    all_targets = None
    all_predictions = None
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.phase = "ACTION"
        outputs = model(inputs)

        predictions = np.where(outputs > 0.5, 1, 0)

        if all_targets is None:
            all_targets = np.asarray(targets)
            all_predictions = predictions

        else:
            # Concatenate by row
            all_targets = np.concatenate((all_targets, np.asarray(targets)), axis=0)
            all_predictions = np.concatenate((all_predictions, predictions), axis=0)

    confusion_matrix = multilabel_confusion_matrix(all_targets, all_predictions)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)

    return confusion_matrix


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
    scores, sortind = torch.sort(torch.from_numpy(scores), dim=0, descending=True)
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