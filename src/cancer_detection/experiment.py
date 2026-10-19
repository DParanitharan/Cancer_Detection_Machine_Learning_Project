"""Nested cross-validation and the single held-out test evaluation for one experiment spec."""

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imblearn
import joblib
import numpy as np
import sklearn
import yaml
from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from cancer_detection import __version__
from cancer_detection.data import DEFAULT_DATA_DIR, FEATURE_COLUMNS, SPLIT_FILES, fingerprint, load_task
from cancer_detection.evaluation import choose_threshold, held_out_metrics, point_metrics, ppv_at_prevalence
from cancer_detection.models import PARAM_GRIDS, build_pipeline

SCREENING_PREVALENCE = 0.01  # illustrative cancer prevalence in a general screening population


@dataclass(frozen=True)
class Spec:
    name: str
    task: str
    model: str
    features: str = "all"
    selector: str = "none"
    sampler: str = "none"
    class_weight: str = None
    log_transform: bool = False

    def pipeline(self, seed):
        return build_pipeline(self.model, self.features, self.selector, self.sampler,
                              self.class_weight, self.log_transform, seed)


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 2
    inner_splits: int = 3
    seed: int = 42

    def inner(self):
        return StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.seed)

    def outer(self):
        return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed)


def load_config(path):
    """Expand the YAML variants x tasks matrix into Specs."""
    config = yaml.safe_load(Path(path).read_text())
    cv = CVConfig(**config.get("cv", {}))
    variants = config["variants"]
    specs = [Spec(name=f"{task}_{variant}", task=task, **variants[variant])
             for group in config["groups"] for task in group["tasks"] for variant in group["variants"]]
    return specs, cv


def fit_tuned(spec, x, y, cv):
    """Tune hyperparameters by inner-CV ROC-AUC (threshold-free); return the refitted best pipeline."""
    grid = PARAM_GRIDS[spec.model]
    if not grid:
        return spec.pipeline(cv.seed).fit(x, y), {}
    search = GridSearchCV(spec.pipeline(cv.seed), grid, scoring="roc_auc", cv=cv.inner(), n_jobs=-1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_


def out_of_fold_scores(estimator, x, y, cv):
    return cross_val_predict(clone(estimator), x, y, cv=cv.inner(), method="predict_proba", n_jobs=-1)[:, 1]


def nested_cv(spec, x, y, cv):
    """Outer folds estimate performance; tuning and threshold choice happen inside each outer training fold."""
    folds = []
    for train_idx, val_idx in cv.outer().split(x, y):
        x_tr, y_tr, x_val, y_val = x.iloc[train_idx], y.iloc[train_idx], x.iloc[val_idx], y.iloc[val_idx]
        best, _ = fit_tuned(spec, x_tr, y_tr, cv)
        threshold = choose_threshold(y_tr, out_of_fold_scores(best, x_tr, y_tr, cv))
        folds.append(point_metrics(y_val, best.predict_proba(x_val)[:, 1], threshold))
    keys = [k for k, v in folds[0].items() if isinstance(v, float)]
    return {k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))}
            for k in keys}


def fit_calibrator(y, oof_scores):
    """Platt scaling on out-of-fold scores: maps resampled/weighted scores back to real probabilities."""
    features = logit(np.clip(oof_scores, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return LogisticRegression().fit(features, y)


def calibrated(calibrator, scores):
    return calibrator.predict_proba(logit(np.clip(scores, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]


def final_model(spec, x_train, y_train, cv):
    """Tune on all training data, then choose the calibrator and threshold from out-of-fold predictions."""
    best, params = fit_tuned(spec, x_train, y_train, cv)
    oof = out_of_fold_scores(best, x_train, y_train, cv)
    calibrator = fit_calibrator(y_train, oof)
    threshold = choose_threshold(y_train, calibrated(calibrator, oof))
    return best, calibrator, threshold, params


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(data_dir):
    return {
        "package_version": __version__,
        "git_commit": git_commit(),
        "data_sha256": {split: fingerprint(Path(data_dir) / name) for split, name in SPLIT_FILES.items()},
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "imblearn": imblearn.__version__,
    }


def run(spec, cv, data_dir=DEFAULT_DATA_DIR, models_dir=None):
    """Nested CV on training data, then one evaluation of the final model on the untouched test set."""
    started = time.time()
    x_train, y_train, x_test, y_test = load_task(spec.task, data_dir)
    cv_summary = nested_cv(spec, x_train, y_train, cv)
    best, calibrator, threshold, params = final_model(spec, x_train, y_train, cv)
    test_probability = calibrated(calibrator, best.predict_proba(x_test)[:, 1])
    test = held_out_metrics(y_test, test_probability, threshold)
    test["brier"] = float(brier_score_loss(y_test, test_probability))
    test["ppv_at_screening_prevalence"] = ppv_at_prevalence(test["sensitivity"], test["specificity"],
                                                            SCREENING_PREVALENCE)
    record = {
        "spec": asdict(spec), "cv": asdict(cv), "best_params": params, "threshold": threshold,
        "train_size": {"healthy": int((y_train == 0).sum()), "cancer": int((y_train == 1).sum())},
        "test_size": {"healthy": int((y_test == 0).sum()), "cancer": int((y_test == 1).sum())},
        "nested_cv": cv_summary, "test": test, "provenance": provenance(data_dir),
        "runtime_seconds": round(time.time() - started, 1),
    }
    if models_dir:
        save_model(Path(models_dir), spec, best, calibrator, threshold, record)
    return record


def save_model(models_dir, spec, pipeline, calibrator, threshold, record):
    """Persist the model with everything needed to use it safely: features, threshold, calibration, provenance."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "calibrator": calibrator, "threshold": threshold},
                models_dir / f"{spec.name}.joblib")
    metadata = {"feature_columns": FEATURE_COLUMNS, "positive_label": "cancer", **record}
    (models_dir / f"{spec.name}.json").write_text(json.dumps(metadata, indent=2))
