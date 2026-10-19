import numpy as np

from cancer_detection.data import FEATURE_COLUMNS, HEALTHY, LABEL_COLUMN
from cancer_detection.features import FragmentomicsFeatures, LengthBinner, to_proportions
from tests.conftest import synthetic_frame


def test_binner_returns_35_bins_that_sum_to_one():
    x = synthetic_frame()[FEATURE_COLUMNS]
    binned = LengthBinner(width=10).fit_transform(x)
    assert binned.shape == (len(x), 35)
    assert np.allclose(binned.sum(axis=1), 1.0)
    assert LengthBinner(width=10).get_feature_names_out()[0] == "len_51_60"


def test_fragmentomics_detects_extra_short_fragments():
    frame = synthetic_frame()
    features = FragmentomicsFeatures().fit_transform(frame[FEATURE_COLUMNS])
    short_fraction = features[:, 0]
    healthy = (frame[LABEL_COLUMN] == HEALTHY).to_numpy()
    assert short_fraction[~healthy].mean() > short_fraction[healthy].mean()
    assert features.shape[1] == len(FragmentomicsFeatures.feature_names)


def test_proportions_are_a_distribution():
    assert np.allclose(to_proportions(np.array([[1.0, 1.0, 2.0]])), [[0.25, 0.25, 0.5]])
