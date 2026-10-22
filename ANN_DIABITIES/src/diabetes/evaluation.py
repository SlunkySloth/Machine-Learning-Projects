import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss)


def metrics(y, probabilities, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)), roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), ece_10_bins=float(ece),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))


def choose_threshold(y, p):
    """Maximize validation F1 on a prespecified grid; ties prefer 0.5."""
    grid = np.linspace(0.10, 0.90, 81)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(grid[tied[np.argmin(abs(grid[tied] - 0.5))]])


def bootstrap_intervals(y, p, threshold, iterations=1000, seed=42):
    """Stratified resampling: sampling uncertainty conditional on the fitted model."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    groups = [np.flatnonzero(y == v) for v in [0, 1]]
    draws = []
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws.append(metrics(y[idx], p[idx], threshold))
    names = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "brier"]
    return {k: [float(v) for v in np.quantile([d[k] for d in draws], [0.025, 0.975])] for k in names}


def paired_comparison(y, a, b, threshold_a=0.5, threshold_b=0.5, iterations=1000, seed=42):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    correct_a, correct_b = (a >= threshold_a) == y, (b >= threshold_b) == y
    n_ab = int((correct_a & ~correct_b).sum())
    n_ba = int((~correct_a & correct_b).sum())
    pvalue = float(binomtest(n_ab, n_ab + n_ba, 0.5).pvalue) if n_ab + n_ba else 1.0
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == v) for v in [0, 1]]
    deltas = []
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {"mcnemar_exact_p": pvalue, "a_only_correct": n_ab, "b_only_correct": n_ba,
            "auc_difference_a_minus_b": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "paired_auc_difference_95_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "note": "Single prespecified comparison; not proof of equivalence or clinical superiority."}
