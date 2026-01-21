import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(normal_errors, anomaly_errors):
    """
    Compute ROC-AUC given reconstruction errors for
    normal and anomalous samples.
    """
    y_true = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(anomaly_errors))
    ])

    y_scores = np.concatenate([
        normal_errors,
        anomaly_errors
    ])

    return roc_auc_score(y_true, y_scores)
