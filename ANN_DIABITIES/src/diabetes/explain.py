from math import factorial
import numpy as np
import pandas as pd
from .data import FEATURES


def exact_shapley(model, row, background):
    """Exact interventional Shapley values over eight raw input features.

    Enumerate all 256 coalitions. Missing features are replaced jointly with
    each background row, averaging predicted positive-class probabilities.
    Exactness is relative to this finite background, not the population.
    Feature dependence is not modeled and contributions are not causal.
    """
    x = np.asarray(row, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    m = len(FEATURES)
    if len(x) != m or bg.ndim != 2 or bg.shape[1] != m or len(bg) == 0:
        raise ValueError("Expected one eight-feature row and a nonempty eight-feature background.")
    masks = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(bool)
    batch = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :])
    p = model.predict_proba(pd.DataFrame(batch.reshape(-1, m), columns=FEATURES))[:, 1]
    values = p.reshape(2 ** m, len(bg)).mean(axis=1)
    phi = np.zeros(m)
    for j in range(m):
        for mask in range(2 ** m):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    return {"baseline": float(values[0]), "probability": float(values[-1]),
            "contributions": {name: float(value) for name, value in zip(FEATURES, phi)},
            "additivity_error": float(abs(values[0] + phi.sum() - values[-1])),
            "background_size": len(bg), "method": "exact interventional Shapley (256 coalitions)"}
