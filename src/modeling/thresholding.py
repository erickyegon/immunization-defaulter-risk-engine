from __future__ import annotations

import numpy as np


def top_k_threshold(probabilities, fraction: float = 0.2) -> float:
    probs = np.asarray(probabilities)
    if len(probs) == 0:
        return 0.5
    k = max(1, int(len(probs) * fraction))
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[min(k - 1, len(sorted_probs) - 1)])
