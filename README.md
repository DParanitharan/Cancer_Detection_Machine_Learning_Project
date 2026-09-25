# IT1244 Final Project: Machine Learning for Cancer Detection Using DNA Fragment Analysis

## Environment Setup

```bash
# 1. System dependency for decision-tree visualisation
brew install graphviz

# 2. Create and populate a virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) register the kernel for Jupyter/VS Code
python -m ipykernel install --user --name cancer-detection --display-name "Python (cancer-detection)"
```

In VS Code / Jupyter, select the **Python (cancer-detection)** kernel (or the `.venv`
interpreter). The notebooks auto-detect the project root, so they run correctly
regardless of the working directory. `requirements.lock.txt` holds the exact
pinned versions that were verified to run end-to-end.

## Project Overview

This project develops and evaluates machine learning models for early cancer detection and classification using DNA fragment length profiles. Cancer is a disease where genetic mutations can alter the length and proportions of DNA fragments in a patient's genome. By analyzing the relationship between the proportions of known DNA fragment lengths, we aim to identify the existence and stage of cancer in patients.

The dataset consists of max normalized frequencies of DNA fragment lengths (350 features) and a response variable indicating health status or cancer stage. The project focuses on building classifiers to:

- Distinguish healthy individuals from those with cancer
- Differentiate early-stage cancer vs healthy
- Differentiate screening-stage cancer vs healthy

## Biological Motivation

Cancer often remains asymptomatic until advanced stages, making early detection critical for effective, non-invasive treatment. Genetic mutations in cancer can change the distribution of DNA fragment lengths. By profiling these fragment proportions, machine learning models can be trained to detect cancer presence and stage without requiring prior identification of specific genetic markers.

## Data & Preprocessing

- **Train_Set.csv**: 2193 samples
- **Test_Set.csv**: 1034 samples
- **Features**: 350 max normalized frequencies of DNA fragment lengths
- **Classes**: healthy, early-stage cancer, screening-stage cancer, mid-stage cancer, late-stage cancer

Preprocessing steps include:

- Checking for missing data (none found)
- Filtering response variables for each classification scenario
- Standardizing features
- Converting response variable to binary (0: healthy, 1: cancer)

## Feature Selection Techniques

To avoid overfitting and improve model generalization, four feature selection methods were used:

- **L1 Lasso Regularization**: Penalizes less important features, removing those with low impact.
- **Principal Component Analysis (PCA)**: Reduces dimensionality by retaining principal components explaining most variance.
- **Recursive Feature Elimination (RFE)**: Iteratively removes least important features based on logistic regression p-values.
- **Correlation Analysis**: Identifies and removes highly correlated features to reduce redundancy.

Features are scored across these methods, and only the most significant are retained for each classification scenario.

## Machine Learning Techniques

- **K-Means Clustering**: Unsupervised grouping of similar DNA profiles to explore data structure.
- **Logistic Regression**: Models the probability of cancer presence using a sigmoid function; outputs binary predictions.
- **Decision Trees & Ensemble Methods**: RandomForest, AdaBoost, and GradientBoosting classifiers for interpretable, tree-based classification. Max depth set to 8 to prevent overfitting.
- **K-Nearest Neighbors (KNN)**: Instance-based classification; hyperparameter tuning and cross-validation (GridSearchCV) used to optimize K.
- **Gaussian Naive Bayes (GNB)**: Probabilistic classification assuming normal distribution of features; log-transform applied to features to better approximate normality.

## Evaluation Metrics

- **Precision, Recall, F1 Score, AUC (Area Under Curve)**: Used to assess model performance, with emphasis on minimizing false negatives (FN) due to higher clinical cost.
- **Confusion Matrix, ROC-AUC, Precision-Recall Curves**: Visual and quantitative evaluation of classifier effectiveness.

## Results & Discussion

- Logistic Regression and GNB showed moderate precision and recall, with high recall for cancer detection but lower for other outcomes.
- Decision Tree ensembles (RandomForest, AdaBoost, GradientBoosting) achieved higher precision and F1 scores, with boosting methods helping to reduce overfitting.
- KNN performance was optimized using hyperparameter tuning and cross-validation.
- Feature selection and dimensionality reduction were critical to avoid overfitting and improve generalization to unseen test data.

## References

- [Machine learning‐assisted evaluation of circulating DNA quantitative analysis for cancer screening](https://www.sciencedirect.com/science/article/pii/S2001037014000464)
- [Support Vector Machine Approach for Cancer Detection Using Amplified Fragment Length Polymorphism (AFLP) Screening Method](https://dl.acm.org/doi/10.1145/1056808.1056822)
- [A machine learning approach to optimizing cell-free DNA sequencing panels: with an application to prostate cancer](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-020-07013-6)
- [Lasso Regression](https://www.ibm.com/topics/lasso-regression)
- [Principal Component Analysis](https://www.ibm.com/topics/principal-component-analysis)
- [Recursive Feature Elimination for Feature Selection](https://medium.com/@rithpansanga/logistic-regression-for-feature-selection-selecting-the-right-features-for-your-model-410ca093c5e0)

## How to Use

1. Clone the repository to your local machine.
2. Open the notebooks in Jupyter or VS Code.
3. Run the preprocessing steps to prepare the data.
4. Execute model training and evaluation cells for each scenario.
5. Review the results and visualizations to compare model performance.

## Author

Forked and maintained by dhxrth.

## License

See LICENSE file for details.

## Repository Structure

- **Jupyter Notebooks**: Step-by-step analysis, model building, and evaluation for different classification tasks:
  - Healthy vs Cancer
  - Healthy vs Early Stage Cancer
  - Healthy vs Screening Stage Cancer
  - Decision Tree and KNN models for all scenarios
  - Logistic Regression and Gaussian Naive Bayes models
- **code/**: Contains reusable Python scripts for data preprocessing and additional model implementations.
- **Train_Set.csv / Test_Set.csv**: Labeled datasets for training and testing the models.

## Techniques Used

- **Data Preprocessing**: Cleaning, feature selection, and transformation using pandas and custom scripts.
- **Class Balancing**: Addressing class imbalance with RandomUnderSampler from the imblearn package.
- **Modeling Algorithms**:
  - **Decision Tree Classifier**: For interpretable, tree-based classification.
  - **K-Nearest Neighbors (KNN)**: For instance-based learning and classification.
  - **Logistic Regression**: For probabilistic binary classification.
  - **Gaussian Naive Bayes**: For fast, probabilistic classification assuming feature independence.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix to assess model performance.
- **Visualization**: Matplotlib for plotting results and model summaries.

## How to Use

1. Clone the repository to your local machine.
2. Open the notebooks in Jupyter or VS Code.
3. Run the preprocessing steps to prepare the data.
4. Execute model training and evaluation cells for each scenario.
5. Review the results and visualizations to compare model performance.

## Author

Forked and maintained by dhxrth.

## License

See LICENSE file for details.
