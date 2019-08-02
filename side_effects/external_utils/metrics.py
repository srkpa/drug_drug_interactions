import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve


def auroc(actual, scores):
    assert scores.shape == actual.shape
    n_classes = actual.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], scores[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(actual.ravel(), scores.ravel(), pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def auprc(actual, scores):
    assert scores.shape == actual.shape
    n_classes = actual.shape[1]
    average_precisions, precision, recall, thresholds = {}, {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(actual[:, i], scores[:, i])
        average_precisions[i] = average_precision_score(actual[:, i], scores[:, i])
    precision["micro"], recall["micro"], thresholds["micro"] = precision_recall_curve(actual.ravel(), scores.ravel())
    average_precisions["micro"] = average_precision_score(actual, scores, average="micro")
    return thresholds, average_precisions, precision, recall


def predict(y_scores, threshold):
    y_pred = np.where(y_scores >= threshold, 1, 0)
    return y_pred


def model_classification_report(actual, predicted):
    return classification_report(actual, predicted, output_dict=True)


def acc_precision_f1_recall(actual, predicted, average='macro'):
    return dict(
        acc=_accuracy(actual, predicted, average=average),
        f1=f1_score(actual, predicted, average=average),
        prc=precision_score(actual, predicted, average=average),
        rec=recall_score(actual, predicted, average=average))


def _accuracy(actual, predicted, average=None):
    assert actual.shape == predicted.shape
    if average == "macro":
        return np.mean(
            np.array([accuracy_score(actual[:, i], predicted[:, i]) for i in range(actual.shape[1])])),
    return accuracy_score(actual, predicted)


def apk(actual, predicted, k=10):
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


def mapk(actual, predicted, k=10):
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
