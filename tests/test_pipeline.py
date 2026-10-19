import json

import numpy as np
import pytest

from cancer_detection.__main__ import main
from cancer_detection.data import load_task
from cancer_detection.experiment import CVConfig, Spec, run
from cancer_detection.models import build_pipeline

FAST_CV = CVConfig(n_splits=2, n_repeats=1, inner_splits=2, seed=0)


def test_resampler_runs_only_during_fit(data_dir):
    x_train, y_train, x_test, _ = load_task("hnc", data_dir)
    pipeline = build_pipeline("logreg", sampler="undersample").fit(x_train, y_train)
    assert "sample" in pipeline.named_steps
    assert len(pipeline.predict(x_test)) == len(x_test)  # nothing is resampled at prediction time


@pytest.mark.parametrize("bad", [dict(model="knn", class_weight="balanced"), dict(model="nope"),
                                 dict(model="rf", sampler="nope"), dict(model="rf", features="nope")])
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        build_pipeline(**bad)


def test_run_is_deterministic_and_complete(data_dir):
    spec = Spec(name="hne_rf_cw", task="hne", model="rf", class_weight="balanced")
    first, second = run(spec, FAST_CV, data_dir), run(spec, FAST_CV, data_dir)
    assert first["test"]["roc_auc"] == second["test"]["roc_auc"]
    assert first["test"]["roc_auc"] > 0.9  # synthetic classes are separable
    assert set(first) >= {"nested_cv", "test", "threshold", "provenance", "best_params"}


def test_cli_run_and_summary(data_dir, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("cv: {n_splits: 2, n_repeats: 1, inner_splits: 2, seed: 0}\n"
                      "variants: {gnb_log: {model: gnb, log_transform: true}}\n"
                      "groups: [{tasks: [hns], variants: [gnb_log]}]\n")
    runs = tmp_path / "runs"
    main(["--data-dir", str(data_dir), "run", str(config), "--runs-dir", str(runs)])
    record = json.loads((runs / "hns_gnb_log.json").read_text())
    assert np.isfinite(record["test"]["balanced_accuracy"])
    out = tmp_path / "results.md"
    main(["summary", "--runs-dir", str(runs), "--out", str(out)])
    assert "hns_gnb_log" in out.read_text()
