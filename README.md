# AOA_BRF

Leakage-controlled **Balanced Random Forest (BRF)** pipeline optimized with **AOA / AOA++** for heart disease prediction.

---

## 1. Project Overview

This repository provides an end-to-end, **leakage-controlled** workflow for heart disease prediction using ensemble machine learning models (Random Forest, Balanced Random Forest) combined with:

- **AOA / AOA++ metaheuristic optimization** for hyperparameter search
- **SMOTEENN** for handling class imbalance
- **Permutation Importance (PI-TopK)** for feature selection
- **Probability calibration** using isotonic regression
- **SHAP** for model explainability
- **Statistical stability analysis** using fold-level ROC-AUC comparisons

> The analysis is based on two public datasets: the classic **UCI Heart Disease** dataset and a larger public **cardiovascular dataset**.

The goal is to build a **robust, clinically interpretable classifier** for heart disease detection with high discrimination (AUC) and high recall (sensitivity), while strictly avoiding information leakage from the test set.

The repository contains two reproducibility notebooks:

- **UCI Heart Disease notebook**
  - Main benchmark experiment
  - Baseline models
  - AOA / AOA++ optimized models
  - SHAP explainability
  - Fold-level SHAP stability analysis

- **Larger cardiovascular dataset notebook**
  - Additional robustness evaluation
  - Same leakage-controlled pipeline logic
  - Baseline, AOA, and AOA++ model comparison

The notebooks compare:

- **Baselines**
  - Logistic Regression
  - SVM
  - XGBoost
  - Random Forest
  - Grid Search Random Forest
  - Bayesian-optimized Random Forest

- **Imbalance-aware methods**
  - Random Forest + SMOTEENN
  - Balanced Random Forest (BRF)
  - BRF + SMOTEENN

- **AOA / AOA++ optimization**
  - Tree-based hyperparameter search
  - Fixed random seeds for reproducibility
  - Multi-metric fitness function emphasizing AUC and recall

- **Leakage-controlled evaluation**
  - Proper train/test split
  - Hyperparameter search, feature selection, calibration, and threshold selection only on training data
  - Out-of-fold (OOF) predictions for threshold tuning
  - Independent test-set evaluation only after model development is complete

The final UCI model is a **Balanced Random Forest with AOA++-optimized hyperparameters, PI-TopK features, and isotonic calibration**, evaluated using both OOF train predictions and a held-out test set.

The larger cardiovascular dataset is used as an additional robustness assessment to test whether the pipeline remains stable beyond the compact UCI benchmark.

---

## 2. Dataset

This project uses two public heart disease datasets.

### 2.1. UCI Heart Disease dataset

The **UCI Heart Disease** dataset is used as the primary benchmark.

Dataset link:

https://archive.ics.uci.edu/dataset/45/heart+disease

The notebook automatically detects the target column, such as:

```python
target_col = [c for c in df.columns if c.lower() in ["target", "output", "disease", "num"]][0]
```

If the target has more than two classes, it is binarized as:

```python
y = (y > 0).astype(int)
```

where:

```text
0 = absence of heart disease
1 = presence of heart disease
```

Identifier-like variables, such as `id`, are excluded from modeling and interpretation to avoid identifier leakage.

**Feature types**

- **Numeric columns** are automatically inferred from the dataframe dtypes.
- **Categorical columns** are inferred from known heart-disease fields (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, etc.) and/or `object` dtype.

---

### 2.2. Larger public cardiovascular dataset

The larger cardiovascular dataset is used as an additional robustness assessment.

Dataset link:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

In the revised experiments, the cleaned dataset contains:

```text
68,573 records
54,858 training samples
13,715 independent test samples
```

The dataset includes clinical and lifestyle-related predictors such as:

- age
- gender
- height
- weight
- systolic blood pressure
- diastolic blood pressure
- cholesterol
- glucose
- smoking status
- alcohol intake
- physical activity
- body mass index

The required file name for the larger cardiovascular notebook is:

```text
cardio_train.csv
```

Place `cardio_train.csv` in the same working directory as the notebook before running the code.

---

## 3. Methods & Pipeline

### 3.1. Preprocessing

All preprocessing is handled with a `ColumnTransformer`.

- **Numeric features**
  - `SimpleImputer(strategy="median")`
  - `RobustScaler`

- **Categorical features**
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

This preprocessing block is reused across all models via unified `Pipeline` objects.

All preprocessing is fitted only on the training data within each fold or train/test split to reduce information leakage.

---

### 3.2. Imbalance Handling

To handle class imbalance, the notebooks use:

- **SMOTEENN** (`imblearn.combine.SMOTEENN`)
  Hybrid over-sampling and cleaning

- **BalancedRandomForestClassifier** (`imblearn.ensemble.BalancedRandomForestClassifier`)
  Class-balanced random forest with internal balanced sampling

We compare three main families:

1. **RF + SMOTEENN**
2. **BRF**
3. **BRF + SMOTEENN**

SMOTEENN is applied only inside the training pipeline. Validation and test sets are never resampled.

---

### 3.3. AOA / AOA++ Optimization

The notebooks implement **Arithmetic Optimization Algorithm (AOA)** and **AOA++**-style searches for tree-based hyperparameters.

The optimized hyperparameters include:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

The main optimization setting used in the revised experiments is:

```text
max_iter = 15
pop_size = 24
```

For each candidate hyperparameter set:

1. A pipeline is built: **Preprocessing -> Feature Selection -> Optional SMOTEENN -> Calibrated Model**

2. Candidate performance is evaluated on an internal validation split derived from the training data.

3. A weighted multi-metric score is computed:

```text
Score = 0.4 * AUC + 0.4 * Recall + 0.1 * F1 + 0.1 * Accuracy
```

This objective emphasizes both discrimination and sensitivity, which are important in medical screening tasks.

Separate optimization loops are performed for:

- RF + SMOTEENN
- BRF
- BRF + SMOTEENN

This allows a fair comparison between different imbalance-handling strategies.

---

### 3.4. Feature Selection via Permutation Importance (PI-TopK)

For a more compact and interpretable model, the notebooks use **Permutation Importance**.

The general procedure is:

1. Split the data into training and test partitions.
2. Compute permutation importance only on the training partition.
3. Rank features by mean importance.
4. Keep the top-K features.
5. Use the selected feature subset in downstream pipelines.

The UCI experiment uses PI-TopK feature selection with:

```text
K = 15
```

Identifier-like columns, such as `id`, should be excluded before feature selection, modeling, and SHAP interpretation.

---

### 3.5. Calibration & Threshold Optimization

To obtain better probability estimates and a clinically meaningful decision rule, models are calibrated using:

```python
CalibratedClassifierCV(method="isotonic", cv=5)
```

For each model, the notebooks:

- Compute OOF probabilities on the training set.
- Search thresholds in `[0, 1]`.
- Select the threshold that maximizes F1-score on training OOF predictions.
- Apply the selected threshold to the independent test set.

Performance is reported at:

- The **default threshold** 0.50
- The **optimized threshold** selected from OOF predictions

The independent test set is not used for calibration fitting or threshold selection.

---

### 3.6. SHAP Explainability

For the final calibrated AOA++-BRF model on the UCI dataset, SHAP is used to explain model predictions.

The SHAP workflow includes:

- SHAP summary plot
- SHAP dependence plot
- Mean absolute SHAP ranking
- Fold-level SHAP feature-ranking stability

The SHAP interpretation focuses on clinically meaningful predictors rather than identifier-like variables.

---

## 4. Repository Structure

Current layout of this repository:

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── uci_heart_leakage_controlled_aoa_aoapp_brf_pipeline.ipynb
└── larger_cardiovascular_aoa_aoapp_brf_pipeline.ipynb
```

All files are stored directly on the main branch of the repository.

The repository may optionally generate additional local output files, such as figures, tables, SHAP plots, or HTML outputs, when the notebooks are executed. These outputs are not required for running the code.

---

## 5. Installation

It is recommended to use a virtual environment.

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

The `requirements.txt` contains packages used by both notebooks:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
scikit-optimize
shap
psutil
openpyxl
jupyter
notebook
```

---

## 6. How to Run

1. **Clone** the repository:

```bash
git clone https://github.com/amirrflh/AOA_BRF.git
cd AOA_BRF
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Launch Jupyter**:

```bash
jupyter notebook
# or:
jupyter lab
```

4. Open one of the notebooks:

```text
uci_heart_leakage_controlled_aoa_aoapp_brf_pipeline.ipynb
```

or:

```text
larger_cardiovascular_aoa_aoapp_brf_pipeline.ipynb
```

5. Run all cells from top to bottom.

For the larger cardiovascular notebook, make sure that the following file is placed in the same working directory before running:

```text
cardio_train.csv
```

AOA/AOA++ optimization blocks can be time-consuming because they evaluate multiple candidate models. Runtime depends on CPU resources and notebook configuration.

---

## 7. Results (Summary)

### 7.1. UCI Heart Disease dataset

On the independent UCI test set, the final **AOA++-optimized Balanced Random Forest** achieved:

- **AUC** = **0.9531**
- **Recall** = **0.9706**
- **Precision** = **0.8250**
- **F1-score** = **0.8919**
- **Accuracy** = **0.8696**
- **Optimized threshold** = **0.3317**

This high-recall, leakage-controlled configuration is desirable for clinical screening, where missing a true positive case is costly.

### 7.2. Larger cardiovascular dataset

On the larger public cardiovascular dataset, the final **AOA++ BRF+SMOTEENN** model achieved:

- **AUC** = **0.7967**
- **Recall** = **0.8015**
- **Precision** = **0.6833**
- **F1-score** = **0.7377**
- **Accuracy** = **0.7180**
- **Optimized threshold** = **0.1535**

The larger dataset provided a more conservative robustness assessment. Tuned Random Forest baselines remained highly competitive on this dataset.

For full metric tables, ROC/PR curves, confusion matrices, statistical comparisons, and SHAP visualizations, please refer to the notebook outputs.

---

## 8. Reproducibility Notes

To improve reproducibility, random seeds are fixed, including:

```python
import numpy as np, random, os

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
```

The workflow is designed to avoid information leakage:

- The test set is not used for feature selection.
- The test set is not used for hyperparameter optimization.
- The test set is not used for calibration fitting.
- The test set is not used for threshold selection.
- Preprocessing is fitted only within the training workflow.
- Resampling is applied only to training data inside the pipeline.
- Final test-set evaluation is performed only after model development is complete.

The notebooks may be uploaded with cleared outputs to avoid storing machine-specific warnings, paths, or temporary runtime artifacts.

---

## 9. Acknowledgments

- **Datasets**
  - UCI Machine Learning Repository – Heart Disease dataset
  - Kaggle Cardiovascular Disease dataset

- **Imbalanced learning**
  - `SMOTEENN` and `BalancedRandomForestClassifier` from `imbalanced-learn`

- **Explainability**
  - SHAP library (`shap`)

- **Machine learning**
  - `scikit-learn`
  - `xgboost`
  - `scikit-optimize`

The AOA / AOA++ implementation in these notebooks is inspired by the Arithmetic Optimization Algorithm literature and customized for leakage-controlled heart disease classification experiments.

---

## 10. License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full terms.
