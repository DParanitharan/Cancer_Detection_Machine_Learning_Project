"""Loading, validating and fingerprinting the fragment-length datasets."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_LENGTH, LAST_LENGTH = 51, 400
FEATURE_COLUMNS = [f"length_{n}" for n in range(FIRST_LENGTH, LAST_LENGTH + 1)]
LABEL_COLUMN = "class_label"
HEALTHY = "healthy"
CANCER_STAGES = ("early stage cancer", "screening stage cancer", "mid stage cancer", "late stage cancer")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SPLIT_FILES = {"train": "Train_Set.csv", "test": "Test_Set.csv"}


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


@dataclass(frozen=True)
class Task:
    """A binary healthy-vs-cancer problem; label 1 means cancer."""

    name: str
    train_positives: tuple
    test_positives: tuple
    description: str


TASKS = {
    "hnc": Task("hnc", CANCER_STAGES, CANCER_STAGES, "healthy vs cancer (all stages)"),
    "hne": Task("hne", ("early stage cancer",), ("early stage cancer",), "healthy vs early stage"),
    "hns": Task("hns", ("screening stage cancer",), ("screening stage cancer",), "healthy vs screening stage"),
    # Train on every cancer sample, evaluate on one stage (uses the 1,352 samples hne/hns discard).
    "hne_all": Task("hne_all", CANCER_STAGES, ("early stage cancer",), "train all cancers, test early stage"),
    "hns_all": Task("hns_all", CANCER_STAGES, ("screening stage cancer",), "train all cancers, test screening stage"),
}


def validate(frame: pd.DataFrame) -> None:
    """Check columns, value ranges and labels; raise DataValidationError on the first problem."""
    if list(frame.columns) != FEATURE_COLUMNS + [LABEL_COLUMN]:
        raise DataValidationError("expected columns length_51..length_400 followed by class_label")
    features = frame[FEATURE_COLUMNS]
    if not all(np.issubdtype(dtype, np.floating) for dtype in features.dtypes):
        raise DataValidationError("feature columns must be floating point")
    if features.isna().any().any():
        raise DataValidationError("feature columns contain missing values")
    if ((features <= 0) | (features > 1)).any().any():
        raise DataValidationError("max-normalised frequencies must lie in (0, 1]")
    if not np.allclose(features.max(axis=1), 1.0):
        raise DataValidationError("each row must be max-normalised (row maximum of 1)")
    unknown = set(frame[LABEL_COLUMN]) - {HEALTHY, *CANCER_STAGES}
    if unknown:
        raise DataValidationError(f"unknown class labels: {sorted(unknown)}")
    if frame.duplicated().any():
        raise DataValidationError("dataset contains duplicate rows")


def fingerprint(path: Path) -> str:
    """SHA-256 of a file, so every result can be tied to the exact data it used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    frame = pd.read_csv(Path(data_dir) / SPLIT_FILES[split])
    validate(frame)
    return frame


def to_binary(frame: pd.DataFrame, positives: tuple):
    """Keep healthy rows plus the given cancer stages; return features and 0/1 labels."""
    rows = frame[frame[LABEL_COLUMN].isin((HEALTHY, *positives))]
    labels = (rows[LABEL_COLUMN] != HEALTHY).astype(int)
    return rows[FEATURE_COLUMNS].reset_index(drop=True), labels.reset_index(drop=True)


def load_task(name: str, data_dir: Path = DEFAULT_DATA_DIR):
    """Return X_train, y_train, X_test, y_test for a task in TASKS."""
    task = TASKS[name]
    x_train, y_train = to_binary(load_split("train", data_dir), task.train_positives)
    x_test, y_test = to_binary(load_split("test", data_dir), task.test_positives)
    return x_train, y_train, x_test, y_test
