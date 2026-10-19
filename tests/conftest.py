import numpy as np
import pandas as pd
import pytest

from cancer_detection.data import CANCER_STAGES, FEATURE_COLUMNS, HEALTHY, LABEL_COLUMN


def synthetic_frame(n_per_class=12, seed=0):
    """Max-normalised rows; cancer rows get extra short (100-150 bp) fragments so the classes are separable."""
    rng = np.random.default_rng(seed)
    labels = [HEALTHY] * n_per_class + [stage for stage in CANCER_STAGES for _ in range(n_per_class)]
    lengths = np.arange(51, 401)
    base = np.exp(-((lengths - 167) ** 2) / (2 * 30**2))
    rows = []
    for label in labels:
        shift = 0.0 if label == HEALTHY else 0.4
        row = base + shift * ((lengths >= 100) & (lengths <= 150)) + rng.uniform(0.01, 0.05, size=lengths.size)
        rows.append(row / row.max())
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    frame[LABEL_COLUMN] = labels
    return frame


@pytest.fixture
def data_dir(tmp_path):
    synthetic_frame(seed=0).to_csv(tmp_path / "Train_Set.csv", index=False)
    synthetic_frame(seed=1).to_csv(tmp_path / "Test_Set.csv", index=False)
    return tmp_path
