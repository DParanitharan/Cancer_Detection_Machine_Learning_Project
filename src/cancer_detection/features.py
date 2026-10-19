"""Feature transformers that turn 350 fragment-length frequencies into compact representations."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.base import BaseEstimator, TransformerMixin

from cancer_detection.data import FIRST_LENGTH, LAST_LENGTH

LENGTHS = np.arange(FIRST_LENGTH, LAST_LENGTH + 1)


def to_proportions(x) -> np.ndarray:
    """Rescale each max-normalised row so it sums to 1, making it a length distribution."""
    x = np.asarray(x, dtype=float)
    return x / x.sum(axis=1, keepdims=True)


class LengthBinner(BaseEstimator, TransformerMixin):
    """Sum adjacent lengths into bins of `width` bp; neighbouring lengths are ~0.999 correlated."""

    def __init__(self, width=10):
        self.width = width

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        proportions = to_proportions(x)
        edges = range(0, proportions.shape[1], self.width)
        return np.column_stack([proportions[:, start:start + self.width].sum(axis=1) for start in edges])

    def get_feature_names_out(self, input_features=None):
        starts = LENGTHS[::self.width]
        return np.array([f"len_{s}_{min(s + self.width - 1, LAST_LENGTH)}" for s in starts])


class FragmentomicsFeatures(BaseEstimator, TransformerMixin):
    """Summary statistics used in cfDNA fragmentomics, computed from the length distribution."""

    feature_names = ("short_fraction", "long_fraction", "short_long_ratio", "mean_length",
                     "sd_length", "mode_length", "entropy", "periodicity_10bp")

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        p = to_proportions(x)
        short = p[:, (LENGTHS >= 100) & (LENGTHS <= 150)].sum(axis=1)
        long = p[:, (LENGTHS >= 151) & (LENGTHS <= 220)].sum(axis=1)
        mean = p @ LENGTHS
        sd = np.sqrt((p * (LENGTHS - mean[:, None]) ** 2).sum(axis=1))
        mode = LENGTHS[p.argmax(axis=1)]
        entropy = -(p * np.log(p)).sum(axis=1)
        return np.column_stack([short, long, short / long, mean, sd, mode, entropy, self._periodicity(p)])

    @staticmethod
    def _periodicity(p):
        """Amplitude of the ~10 bp oscillation below 150 bp, after removing the smooth trend."""
        region = (LENGTHS >= FIRST_LENGTH) & (LENGTHS <= 150)
        segment = p[:, region]
        residual = segment - uniform_filter1d(segment, size=11, axis=1, mode="nearest")
        wave = np.exp(-2j * np.pi * LENGTHS[region] / 10.0)
        return np.abs(residual @ wave) / region.sum()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)
