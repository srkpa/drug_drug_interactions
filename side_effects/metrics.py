import numpy as np
import sklearn.metrics as skm
import ivbase.utils.metrics as ivbm
from functools import partial, update_wrapper


def wrapped_partial(func, **kwargs):
    partial_func = partial(func, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def roc_auc_score(y_pred, y_true, average):
    assert y_true.shape == y_pred.shape
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    if average == 'macro':
        out = []
        for i in range(y_true.shape[1]):
            roc_score = skm.roc_auc_score(y_true[:, i], y_pred[:, i], average='micro') if len(
                np.unique(y_true[:, i])) > 1 else 0.5
            out.append(roc_score)
        return np.mean(out)
    return ivbm.roc_auc_score(y_pred, y_true, average=average)


def auprc_score(y_pred, y_true, average):
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    if average == 'macro':
        out = []
        for i in range(y_true.shape[1]):
            roc_score = skm.average_precision_score(y_true[:, i], y_pred[:, i], average='micro') if len(
                np.unique(y_true[:, i])) > 1 else 0.5
            out.append(roc_score)
        return np.mean(out)
    return skm.average_precision_score(y_true, y_pred, average=average)


def model_classification_report(y_pred, y_true, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    return skm.classification_report(y_true, y_pred, output_dict=True)


def precision_score(y_pred, y_true, average, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    return skm.precision_score(y_true, y_pred, average=average)


def recall_score(y_pred, y_true, average, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    skm.recall_score(y_true, y_pred, average=average)


def accuracy_score(y_pred, y_true, average, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred, y_true = ivbm.torch_to_numpy(y_pred), ivbm.torch_to_numpy(y_true)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    if average == "macro":
        return np.mean(
            np.array([skm.accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))
    return skm.accuracy_score(y_true, y_pred)


def apk(predicted, actual, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(predicted, actual, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
