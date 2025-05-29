from pytorch_lightning import Callback
import torch
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
) 
import numpy as np
 


def find_best_threshold(y_true, y_score, steps=100):
    """Finds the best threshold for binary classification based on Half Total Error Rate (HTER).

    :param y_true: Ground truth binary labels (0 or 1).
    :param y_score: Predicted scores or probabilities for the positive class.
    :param steps: Number of threshold steps to evaluate, defaults to 100.
    :return: A tuple containing the best threshold and the corresponding HTER.
    """
    thr = np.linspace(0, 1, steps)
    best_t, best_hter = 0.5, 1.0
    for t in thr:
        preds = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        far = fp / (fp + tn + 1e-9)  # False Acceptance Rate
        frr = fn / (fn + tp + 1e-9)  # False Rejection Rate
        hter = 0.5 * (far + frr)  # Half Total Error Rate
        if hter < best_hter:
            best_hter, best_t = hter, t
    return best_t, best_hter


class MetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Calculate scores using sigmoid activation
        scores = torch.sigmoid(torch.tensor(pl_module.val_outputs))
        # Find the optimal threshold and HTER
        t_opt, hter = find_best_threshold(pl_module.val_labels, scores.numpy())
        # Log HTER and optimal threshold
        pl_module.log("val_HTER", hter, prog_bar=True)
        pl_module.log("val_threshold", t_opt, prog_bar=True)

        # Calculate and log AUC-ROC
        auc_roc = roc_auc_score(pl_module.val_labels, scores.numpy())
        pl_module.log("val_AUC_ROC", auc_roc, prog_bar=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate scores using sigmoid activation
        scores = torch.sigmoid(torch.tensor(pl_module.train_outputs))
        # Find the optimal threshold and HTER
        t_opt, hter = find_best_threshold(pl_module.train_labels, scores.numpy())
        # Log HTER and optimal threshold
        pl_module.log("train_HTER", hter, prog_bar=True)
        pl_module.log("train_threshold", t_opt, prog_bar=True)

        # Calculate and log AUC-ROC
        auc_roc = roc_auc_score(pl_module.train_labels, scores.numpy())
        pl_module.log("train_AUC_ROC", auc_roc, prog_bar=True)