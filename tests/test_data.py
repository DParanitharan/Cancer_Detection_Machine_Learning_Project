import pytest

from cancer_detection.data import (
    DEFAULT_DATA_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    DataValidationError,
    fingerprint,
    load_split,
    load_task,
    validate,
)
from tests.conftest import synthetic_frame


def test_real_datasets_pass_validation():
    assert len(load_split("train", DEFAULT_DATA_DIR)) == 2193
    assert len(load_split("test", DEFAULT_DATA_DIR)) == 1034


def halve_features(frame):
    """Rows whose maximum is 0.5, i.e. no longer max-normalised."""
    halved = frame.copy()
    halved[FEATURE_COLUMNS] = halved[FEATURE_COLUMNS] * 0.5
    return halved


@pytest.mark.parametrize("corrupt", [
    lambda f: f.drop(columns=FEATURE_COLUMNS[0]),
    lambda f: f.assign(**{FEATURE_COLUMNS[0]: 0.0}),
    lambda f: f.assign(**{FEATURE_COLUMNS[0]: 1.5}),
    lambda f: f.assign(**{FEATURE_COLUMNS[0]: float("nan")}),
    lambda f: f.assign(**{LABEL_COLUMN: "unknown stage"}),
    halve_features,
])
def test_validate_rejects_bad_data(corrupt):
    with pytest.raises(DataValidationError):
        validate(corrupt(synthetic_frame()))


def test_validate_rejects_duplicates():
    frame = synthetic_frame()
    with pytest.raises(DataValidationError):
        validate(frame.iloc[[0, 0]])


def test_load_task_labels_cancer_as_one(data_dir):
    x_train, y_train, x_test, y_test = load_task("hne", data_dir)
    assert list(x_train.columns) == FEATURE_COLUMNS
    assert sorted(y_train.unique()) == [0, 1]
    assert (y_train == 0).sum() == 12 and (y_train == 1).sum() == 12


def test_all_cancer_task_trains_on_every_stage_but_tests_on_one(data_dir):
    _, y_train, _, y_test = load_task("hne_all", data_dir)
    assert (y_train == 1).sum() == 48
    assert (y_test == 1).sum() == 12


def test_fingerprint_changes_with_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x"), b.write_text("y")
    assert fingerprint(a) != fingerprint(b)
    assert len(fingerprint(a)) == 64
