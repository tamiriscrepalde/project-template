"""Evaluation class and functions."""

from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def get_binary_results(
    y_pred_proba: pd.Series,
    threshold: float = 0.6
) -> NDArray:
    """Get binary results from probabilistic results based on a threshold.

    Args:
        y_pred_proba (pd.Series): Pandas Series of probabilistic results.
        threashold (float, optional): Threshold to be considered. Defaults to
        0.5.

    Returns:
        NDArray: Array of binary results.
    """
    y_pred = np.array([1 if i >= threshold else 0 for i in y_pred_proba[:, 1]])

    return y_pred


def compute_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
    """Generate dict of metric values.

    Args:
        y_test (pd.Series): Pandas Series of true values.
        y_pred (pd.Series): Pandas Series of predicted values.

    Returns:
        dict: Dictionary containing metric values.
    """
    recall_pos = recall_score(y_test, y_pred, average='binary')
    recall_neg = recall_score(y_test, y_pred, average='binary', pos_label=0)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_pos = f1_score(y_test, y_pred, average='binary')
    f1_neg = f1_score(y_test, y_pred, average='binary', pos_label=0)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accur = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        'recall_pos': recall_pos,
        'recall_neg': recall_neg,
        'recall_macro': recall_macro,
        'f1_pos': f1_pos,
        'f1_neg': f1_neg,
        'f1_macro': f1_macro,
        'accur': accur,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
