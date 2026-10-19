"""Metrics that stay honest under heavy class imbalance. Label 1 = cancer, 0 = healthy."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def choose_threshold(y_true, scores) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on validation scores."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = np.argmax(tpr - fpr)
    return float(min(thresholds[best], 1.0))


def confusion_counts(y_true, scores, threshold) -> dict:
    y_true = np.asarray(y_true)
    predicted = np.asarray(scores) >= threshold
    return {"tp": int(np.sum(predicted & (y_true == 1))), "fn": int(np.sum(~predicted & (y_true == 1))),
            "tn": int(np.sum(~predicted & (y_true == 0))), "fp": int(np.sum(predicted & (y_true == 0)))}


def sensitivity_at_specificity(y_true, scores, specificity=0.95) -> float:
    """Highest share of cancers caught while still clearing at least `specificity` of healthy people."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= 1 - specificity].max())


def point_metrics(y_true, scores, threshold) -> dict:
    c = confusion_counts(y_true, scores, threshold)
    sensitivity = c["tp"] / (c["tp"] + c["fn"])
    specificity = c["tn"] / (c["tn"] + c["fp"])
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_cancer": 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        "sensitivity_at_95_specificity": sensitivity_at_specificity(y_true, scores, 0.95),
        **c,
    }


def wilson_interval(successes, n, z=1.96) -> tuple:
    """95% Wilson score interval for a proportion; well behaved for small n such as 41 healthy people."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (float(centre - half), float(centre + half))


def bootstrap_interval(y_true, scores, metric, n_resamples=1000, seed=42) -> tuple:
    """Stratified bootstrap 95% interval for a threshold-free metric such as ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    groups = [np.flatnonzero(y_true == label) for label in (0, 1)]
    values = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values.append(metric(y_true[idx], scores[idx]))
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def held_out_metrics(y_true, scores, threshold) -> dict:
    """Point metrics plus 95% intervals, for the single held-out test evaluation."""
    m = point_metrics(y_true, scores, threshold)
    m["sensitivity_ci"] = wilson_interval(m["tp"], m["tp"] + m["fn"])
    m["specificity_ci"] = wilson_interval(m["tn"], m["tn"] + m["fp"])
    m["roc_auc_ci"] = bootstrap_interval(y_true, scores, roc_auc_score)
    return m


def ppv_at_prevalence(sensitivity, specificity, prevalence) -> float:
    """Share of positive calls that are real cancers at a given prevalence (Bayes' rule)."""
    true_positive = sensitivity * prevalence
    false_positive = (1 - specificity) * (1 - prevalence)
    return true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
