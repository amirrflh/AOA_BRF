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

The notebook automatically detects the target column, such as:

```python
target_col = [c for c in df.columns if c.lower() in ["target", "output", "disease", "num"]][0]
