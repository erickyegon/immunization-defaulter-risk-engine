from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def evaluate_predictions(y_true, y_prob) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }
