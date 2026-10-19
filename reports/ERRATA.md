# Errata: IT1244 Project Report (Team 22)

Corrections to `IT1244 Project Report_Team 22.docx.pdf`, found when the notebooks were re-audited in October 2026. The PDF itself is unchanged; read it together with this list.

## Results

1. **All reported metrics are invalid.** The original notebooks resampled the *test* set (SMOTE / undersampling) before scoring. That injected synthetic patients into the test set and discarded real ones, so every precision, recall, F1 and AUC in the report describes a population that does not exist. The test set is now scored unresampled; current results are produced by the `cancer_detection` package (see the README).
2. **F1 on the cancer class is not informative here.** Healthy people are the rare class (41 of 1034 test samples). Always predicting "cancer" scores F1 = 0.980 (cancer vs healthy), 0.947 (early vs healthy) and 0.918 (screening vs healthy), so high F1 values do not show skill. Balanced accuracy, ROC-AUC and specificity are the meaningful measures.
3. **"Minimise false negatives, and consequently maximise precision" is wrong.** Minimising false negatives maximises **recall** (sensitivity). The same paragraph also swaps the labels FN and FP.

## Methods described but not implemented

4. **K-means clustering** is described in Methods and Results, but no clustering code exists in the repository.
5. **XGBoost** is reported, but no XGBoost model exists. The boosting models are scikit-learn's `AdaBoostClassifier` and `GradientBoostingClassifier`.
6. **`max_depth = 8`** is never used. The tree models were tuned over other depth grids.
7. **Standardisation.** The report says features were standardised for all models. In the original code, the standardised features were used only for feature selection; logistic regression and KNN were trained on raw values. Both now standardise using statistics fitted on training data only. (Naive Bayes and tree models are unaffected by feature scale.)

## Feature selection

8. **The p-value method never worked.** `statsmodels.Logit` was fitted without an intercept (pseudo-R² of -4.47), and its features were named `x1..x350`, so none of its votes matched a `length_*` column. It has been removed.
9. **PCA does not rank original features.** The code treated component *k* as feature *k*, so it kept the first columns of the table by position. PCA no longer votes; its variance plots remain as exploration.
10. **RFE does not use p-values.** It removes the feature with the smallest coefficient magnitude at each step.
11. **The correlation threshold sweep was wrong.** It reset its lists inside the loop and reported 349 features at every threshold; the correct sweep keeps 2 features between 0.70 and 0.84.
12. **Feature counts.** The report gives 23 / 18 / 24 features (screening / early / cancer). The code actually kept 18 / 24 / 20; after the fixes above it keeps 17 / 23 / 17.

## Decision-tree section

13. The "decision tree" results come from Random Forest, AdaBoost and Gradient Boosting ensembles.
    - The "best tree" was a single AdaBoost weak learner chosen by the index of the best cross-validation fold, which is unrelated to the trees.
    - The evaluation of the random forest's first tree inverted every label.
    - The gradient-boosting "tree" was a new tree trained separately, not part of the model.
