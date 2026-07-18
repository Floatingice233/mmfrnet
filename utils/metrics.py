"""Evaluation metrics: Accuracy and AUC."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_metrics(outputs, targets, num_classes):
    """Compute ACC and AUC.

    Args:
        outputs: (N, num_classes) logits or probabilities
        targets: (N,) class labels
        num_classes: number of classes

    Returns:
        acc: accuracy score
        auc: macro-average AUC (one-vs-rest for multi-class)
    """
    probs = softmax(outputs)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(targets, preds)

    if num_classes == 2:
        auc = roc_auc_score(targets, probs[:, 1])
    else:
        targets_onehot = np.eye(num_classes)[targets]
        try:
            auc = roc_auc_score(targets_onehot, probs, multi_class="ovr",
                                average="macro")
        except ValueError:
            auc = 0.0

    return acc, auc


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
