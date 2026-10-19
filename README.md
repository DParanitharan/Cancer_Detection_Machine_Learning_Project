# Cancer Detection from cfDNA Fragment Lengths

Can the length distribution of cell-free DNA (cfDNA) fragments in blood separate cancer patients from healthy people? This repository started as the IT1244 final project (Team 22, 2024). It was re-audited in 2026: the original notebooks were corrected, and a tested, leakage-safe pipeline (`src/cancer_detection`) now produces every reported number.

> **Not for clinical use.** The dataset has only 60 healthy training samples and 41 healthy test samples. See [Limitations](#limitations).

## Data

`data/raw/Train_Set.csv` and `data/raw/Test_Set.csv`: one row per person and 350 features, `length_51` … `length_400`. Each feature is the frequency of fragments of that length in base pairs, divided by the person's most common length (so every row's maximum is 1). The label column is `class_label`.

| Class | Train | Test |
|---|---|---|
| healthy | 60 | 41 |
| early stage cancer | 781 | 368 |
| screening stage cancer | 490 | 230 |
| mid stage cancer | 453 | 206 |
| late stage cancer | 409 | 189 |

Healthy people are the **rare** class. Always predicting "cancer" therefore scores very high accuracy and F1, so results are reported as balanced accuracy, ROC-AUC, sensitivity and specificity instead. `reports/data_audit.md` covers integrity checks, train/test shift and open provenance questions.

## Tasks

| Task | Train on | Test on |
|---|---|---|
| `hnc` | healthy vs all cancer stages | healthy vs all cancer stages |
| `hne` | healthy vs early stage | healthy vs early stage |
| `hns` | healthy vs screening stage | healthy vs screening stage |
| `hne_all` | healthy vs all cancer stages | healthy vs early stage |
| `hns_all` | healthy vs all cancer stages | healthy vs screening stage |

## Method

Every fitted step lives inside one `imblearn` pipeline, so cross-validation refits all of them on each training fold:

```text
features (raw 350 / 10 bp bins / fragmentomics) -> log (Naive Bayes) -> standardise -> L1 selection (optional)
    -> SMOTE or undersampling (training folds only) -> classifier
```

- **Nested cross-validation.** The outer loop (5 folds × 2 repeats) estimates performance. Inside each outer training fold, an inner 3-fold loop tunes hyperparameters by ROC-AUC and chooses the decision threshold that maximises balanced accuracy.
- **One test evaluation.** The final model is tuned on the full training set and scored once on the test set. Probabilities are Platt-calibrated using out-of-fold predictions. Test metrics come with 95% intervals (Wilson for sensitivity and specificity, bootstrap for ROC-AUC).
- **Models are compared by nested-CV score only.** Test scores are reported but never used for selection.
- **Baselines.** Every task includes a `DummyClassifier`, and results tables show what "always cancer" would score.
- **Models:**
  - logistic regression (L2, optional L1 feature selection)
  - Gaussian Naive Bayes on log features
  - KNN
  - Random Forest
  - histogram gradient boosting

  Class imbalance is handled by class weighting, undersampling or SMOTE, and these are compared directly.
- **Reproducibility.** Each run record in `results/runs/*.json` stores the spec, best hyperparameters, nested-CV and test metrics, the git commit, SHA-256 hashes of the data files, and library versions.

## Results

Full tables: [`reports/results.md`](reports/results.md). Learning curves and fragment-length importance: [`reports/analysis.md`](reports/analysis.md).

The model for each task is chosen by nested-CV balanced accuracy. Held-out test results include 95% intervals.

| Task | Model chosen by CV | CV balanced acc | Test balanced acc | Test ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Early stage vs healthy | logistic regression + SMOTE | 0.856 ± 0.038 | 0.877 | 0.935 [0.89, 0.97] | 0.92 [0.89, 0.95] | 0.83 [0.69, 0.91] |
| All cancer vs healthy | logistic regression, 10 bp bins | 0.754 ± 0.067 | 0.773 | 0.869 [0.83, 0.91] | 0.60 [0.56, 0.63] | 0.95 [0.84, 0.99] |
| Screening stage vs healthy | none beats chance | 0.444–0.495 | — | — | — | — |

- **Early-stage cancer is separable from healthy.** The most important fragment lengths sit at 120–130 bp, the short-fragment band associated with tumour-derived cfDNA. Validation ROC-AUC is still rising at 48 healthy training samples, so more healthy samples would likely help.
- **Screening-stage cancer is not.** Every model and imbalance strategy scores at chance in nested CV, and training on all cancer stages does not transfer (test balanced accuracy 0.52).
- **Simple models win.** Class-weighted or SMOTE logistic regression matches or beats random forests and gradient boosting on held-out data. The eight fragmentomics summary features lose information compared with the raw or binned lengths.
- **Screening would still be impractical.** At a 1% cancer prevalence, the best early-stage model's positive calls would be real cancers only about 5% of the time (range 3–10%).
- **The original notebooks gave a misleading picture.** Their saved results came from a resampled test set and from F1 on the majority class.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps, the package (editable) and dev tools
pre-commit install                    # strips notebook outputs and runs ruff on every commit
brew install graphviz                 # only needed for the tree plots in the original notebooks
```

## Usage

```bash
python -m cancer_detection run configs/experiments.yaml            # all experiments -> results/runs/
python -m cancer_detection run configs/experiments.yaml --only hne_rf_cw --save-models   # one, saving the model to models/
python -m cancer_detection summary                                  # -> reports/results.md
python -m cancer_detection audit                                    # -> reports/data_audit.md
python -m cancer_detection analysis configs/experiments.yaml --names hnc_rf_cw   # -> reports/analysis.md
pytest                                                              # tests
ruff check .                                                        # lint
```

To add an experiment, add a variant and a group entry in `configs/experiments.yaml`. A saved model (`models/<name>.joblib`) contains the pipeline, the calibrator and the decision threshold. Its metadata JSON records the expected feature columns and the run's provenance.

## Repository structure

```text
configs/                 experiment matrices (experiments.yaml, smoke.yaml for CI)
data/raw/                Train_Set.csv, Test_Set.csv
src/cancer_detection/    data loading and validation, features, pipelines, evaluation, CLI
tests/                   pytest suite (synthetic data plus a real-data schema check)
results/runs/            one JSON record per experiment
reports/                 results.md, analysis.md, data_audit.md, ERRATA.md, original report (PDF), figures/
notebooks/               original course notebooks (01-03 preprocessing, models/) and 04_pipeline_results.ipynb
```

## Original course notebooks

The notebooks in `notebooks/` are the original study, kept so the course work stays reproducible. They have been corrected:

- seeds are fixed,
- logistic regression and KNN are standardised,
- the broken p-value and PCA feature votes are removed,
- the correlation sweep is fixed,
- the tree-ensemble notebook's label flip and crashes are fixed,
- KNN no longer chooses k using the test set.

They still run each preprocessing notebook through `%run`, and some still cross-validate on already-resampled data. Use the package for any new work. Outputs are stripped on commit, so run them in Jupyter to see results.

The original report (`reports/IT1244 Project Report_Team 22.docx.pdf`) contains errors, including results computed on a resampled test set. Read it together with [`reports/ERRATA.md`](reports/ERRATA.md).

## Limitations

- **Too few healthy samples.** With 60 healthy training samples and 41 healthy test samples, specificity estimates have wide intervals (for example, 23/41 has a 95% interval of 0.41–0.70).
- **Inverted prevalence.** 96% of the test set has cancer; a screening population is closer to 1%. Each results table shows the positive predictive value at 1% prevalence, which is much lower than any number measured here.
- **Unknown provenance.** It is unknown whether healthy and cancer samples share sites, sequencing batches or age ranges. A model could learn those differences instead of cancer. See `reports/data_audit.md`.

## References

- [Machine learning-assisted evaluation of circulating DNA quantitative analysis for cancer screening](https://www.sciencedirect.com/science/article/pii/S2001037014000464)
- [Support Vector Machine Approach for Cancer Detection Using AFLP Screening](https://dl.acm.org/doi/10.1145/1056808.1056822)
- [A machine learning approach to optimizing cell-free DNA sequencing panels: with an application to prostate cancer](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-020-07013-6)

## Authors

IT1244 Team 22 (see the report for the full author list). No license file is included, so all rights are reserved by the authors.
