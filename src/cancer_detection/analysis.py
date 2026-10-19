"""Learning curves and permutation importance, both computed on training data only."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from cancer_detection.data import FEATURE_COLUMNS, load_task
from cancer_detection.features import LENGTHS


def learning_curve_section(spec, cv, data_dir, figures_dir, fractions=(0.25, 0.5, 0.75, 1.0)):
    """Does cross-validated ROC-AUC still rise as more (healthy) training samples are added?"""
    x, y, _, _ = load_task(spec.task, data_dir)
    sizes, _, val_scores = learning_curve(spec.pipeline(cv.seed), x, y, train_sizes=list(fractions), cv=cv.outer(),
                                          scoring="roc_auc", shuffle=True, random_state=cv.seed, n_jobs=-1)
    healthy_share = float((y == 0).mean())
    lines = [f"### Learning curve: {spec.name}", "",
             "Default hyperparameters; validation ROC-AUC over the outer CV folds.", "",
             "| Training samples | ≈ Healthy samples | Validation ROC-AUC |", "|---|---|---|"]
    lines += [f"| {n} | {round(n * healthy_share)} | {s.mean():.3f} ± {s.std():.3f} |"
              for n, s in zip(sizes, val_scores, strict=True)]
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.errorbar(sizes * healthy_share, val_scores.mean(axis=1), yerr=val_scores.std(axis=1), marker="o", capsize=3)
    ax.set(xlabel="Approximate healthy samples in training fold", ylabel="Validation ROC-AUC",
           title=f"Learning curve: {spec.name}")
    ax.grid(True)
    path = Path(figures_dir) / f"learning_curve_{spec.name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return lines + ["", f"![Learning curve]({Path('figures') / path.name})"]


def importance_section(spec, cv, data_dir, figures_dir, top=15):
    """Permute each raw fragment length on held-out folds and measure the ROC-AUC drop."""
    x, y, _, _ = load_task(spec.task, data_dir)
    drops = []
    for train_idx, val_idx in cv.inner().split(x, y):
        model = spec.pipeline(cv.seed).fit(x.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(model, x.iloc[val_idx], y.iloc[val_idx], scoring="roc_auc",
                                        n_repeats=5, random_state=cv.seed, n_jobs=-1)
        drops.append(result.importances_mean)
    mean_drop = np.mean(drops, axis=0)
    order = np.argsort(mean_drop)[::-1][:top]
    lines = [f"### Permutation importance: {spec.name}", "",
             "Mean drop in validation ROC-AUC when one fragment length is shuffled (default hyperparameters).", "",
             "| Fragment length | ROC-AUC drop |", "|---|---|"]
    lines += [f"| {FEATURE_COLUMNS[i].removeprefix('length_')} bp | {mean_drop[i]:.4f} |" for i in order]
    fig = Figure(figsize=(8, 3.5))
    ax = fig.subplots()
    ax.bar(LENGTHS, mean_drop, width=1.0)
    ax.axvspan(100, 150, alpha=0.15, color="tab:orange", label="short fragments (100-150 bp)")
    ax.set(xlabel="Fragment length (bp)", ylabel="ROC-AUC drop", title=f"Permutation importance: {spec.name}")
    ax.legend()
    path = Path(figures_dir) / f"importance_{spec.name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return lines + ["", f"![Permutation importance]({Path('figures') / path.name})"]
