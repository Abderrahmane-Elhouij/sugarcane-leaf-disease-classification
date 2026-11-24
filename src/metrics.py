import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(labels, probs):
    predictions = probs.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
    }


def per_class_metrics(labels, probs, class_names):
    predictions = probs.argmax(1)
    indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )
    aucs = [roc_auc_score(labels == k, probs[:, k]) for k in indices]
    return pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": aucs,
        "support": support,
    })


def normalized_confusion(labels, probs, n_classes):
    counts = confusion_matrix(labels, probs.argmax(1), labels=list(range(n_classes)))
    return counts, counts / counts.sum(axis=1, keepdims=True)


def one_vs_rest_roc(labels, probs):
    curves = {}
    for k in range(probs.shape[1]):
        fpr, tpr, _ = roc_curve(labels == k, probs[:, k])
        curves[k] = (fpr, tpr, auc(fpr, tpr))
    grid = np.linspace(0.0, 1.0, 2001)
    mean_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr, _ in curves.values()], axis=0)
    mean_tpr[0] = 0.0
    curves["macro"] = (grid, mean_tpr, auc(grid, mean_tpr))
    return curves
