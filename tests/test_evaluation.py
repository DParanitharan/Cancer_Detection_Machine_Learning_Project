import numpy as np
import pytest

from cancer_detection.evaluation import (
    choose_threshold,
    confusion_counts,
    held_out_metrics,
    point_metrics,
    ppv_at_prevalence,
    wilson_interval,
)


def test_wilson_interval_matches_known_value():
    low, high = wilson_interval(23, 41)
    assert low == pytest.approx(0.410, abs=0.002)
    assert high == pytest.approx(0.701, abs=0.002)


def test_choose_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = choose_threshold(y, scores)
    assert confusion_counts(y, scores, threshold) == {"tp": 3, "fn": 0, "tn": 3, "fp": 0}


def test_majority_guess_has_chance_balanced_accuracy_but_high_f1():
    y = np.array([0] * 4 + [1] * 96)
    metrics = point_metrics(y, np.ones(100), threshold=0.5)
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1_cancer"] > 0.97


def test_held_out_metrics_reports_intervals():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 70)
    scores = np.clip(y * 0.3 + rng.uniform(0, 0.7, 100), 0, 1)
    m = held_out_metrics(y, scores, 0.5)
    assert m["specificity_ci"][0] <= m["specificity"] <= m["specificity_ci"][1]
    assert m["roc_auc_ci"][0] <= m["roc_auc"] <= m["roc_auc_ci"][1]


def test_ppv_collapses_at_screening_prevalence():
    assert ppv_at_prevalence(0.93, 0.56, 0.01) == pytest.approx(0.0209, abs=1e-3)
