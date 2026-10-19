"""Build leakage-safe pipelines: every fitted step lives inside one imblearn Pipeline."""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cancer_detection.features import FragmentomicsFeatures, LengthBinner

FEATURE_SETS = {"all": lambda: None, "bins10": lambda: LengthBinner(width=10), "fragmentomics": FragmentomicsFeatures}
SELECTORS = ("none", "l1")
WEIGHTABLE_MODELS = {"logreg", "rf", "hgb"}

PARAM_GRIDS = {
    "dummy": {},
    "logreg": {"model__C": [0.01, 0.1, 1, 10]},
    "gnb": {},
    "knn": {"model__n_neighbors": [3, 5, 9, 15, 25]},
    "rf": {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]},
    "hgb": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]},
}


def make_classifier(model, class_weight, random_state):
    if class_weight and model not in WEIGHTABLE_MODELS:
        raise ValueError(f"class_weight is not supported for model '{model}'")
    factories = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: LogisticRegression(class_weight=class_weight, max_iter=5000),
        "gnb": GaussianNB,
        "knn": KNeighborsClassifier,
        "rf": lambda: RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=random_state),
        "hgb": lambda: HistGradientBoostingClassifier(class_weight=class_weight, random_state=random_state),
    }
    if model not in factories:
        raise ValueError(f"unknown model '{model}'")
    return factories[model]()


def make_sampler(sampler, random_state):
    samplers = {"none": None, "undersample": RandomUnderSampler(random_state=random_state),
                "smote": SMOTE(random_state=random_state)}
    if sampler not in samplers:
        raise ValueError(f"unknown sampler '{sampler}'")
    return samplers[sampler]


def build_pipeline(model, features="all", selector="none", sampler="none", class_weight=None,
                   log_transform=False, random_state=42):
    """features -> (log) -> scale -> (L1 selection) -> (resampling, training only) -> classifier."""
    if features not in FEATURE_SETS:
        raise ValueError(f"unknown feature set '{features}'")
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector '{selector}'")
    candidate_steps = [
        ("features", FEATURE_SETS[features]()),
        ("log", FunctionTransformer(np.log) if log_transform else None),
        ("scale", StandardScaler()),
        ("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", random_state=random_state))
         if selector == "l1" else None),
        ("sample", make_sampler(sampler, random_state)),
        ("model", make_classifier(model, class_weight, random_state)),
    ]
    return Pipeline([(name, step) for name, step in candidate_steps if step is not None])
