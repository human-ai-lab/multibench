import sklearn.metrics
import numpy as np


def ptsort(tu):
    return tu[0]


def AUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth.detach().cpu().numpy(), pred.detach().cpu().numpy(), average=average)


def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth.detach().cpu().numpy(), pred.detach().cpu().numpy())


def unweighted_accuracy(truth, pred):
    """Unweighted accuracy: macro-averaged per-class recall.

    Also commonly reported as UAR (Unweighted Average Recall) in the
    speech/affect emotion recognition literature - both names refer to the
    same computation (sklearn's balanced accuracy).
    """
    return sklearn.metrics.balanced_accuracy_score(
        truth.detach().cpu().numpy(), pred.detach().cpu().numpy())


_METRIC_FUNCS = {
    "ua": unweighted_accuracy,
    "uar": unweighted_accuracy,
    "wa": accuracy,
    "accuracy": accuracy,
    "f1": lambda truth, pred: f1_score(truth, pred, average="macro"),
}


def compute_metrics(truth, pred, names):
    """Compute a dict of named classification metrics for label tensors.

    :param truth: ground truth class labels, shape (N,)
    :param pred: predicted class labels, shape (N,)
    :param names: iterable of metric names (case-insensitive). Supported:
        "UA"/"UAR" (unweighted accuracy, macro-averaged recall),
        "WA"/"Accuracy" (weighted accuracy, i.e. standard accuracy),
        "F1" (macro-averaged F1 score).
    :return: dict mapping each requested name to its computed value.
    """
    results = {}
    for name in names:
        func = _METRIC_FUNCS.get(name.lower())
        if func is None:
            raise ValueError(
                f"Unknown evaluation metric '{name}'. Supported: {sorted(_METRIC_FUNCS)}")
        results[name] = func(truth, pred)
    return results


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return sklearn.metrics.accuracy_score(binary_truth, binary_preds)
