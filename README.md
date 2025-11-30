# AOA_BRF

Leak-safe **Balanced Random Forest (BRF)** pipeline optimized with **AOA / AOA++** for heart disease prediction.

---

## 1. Project Overview

This repository provides an end-to-end, **leak-safe** workflow for heart disease prediction using
ensemble machine learning models (Random Forest, Balanced Random Forest) combined with:

- **AOA / AOA++ metaheuristic optimization** for hyperparameter search  
- **SMOTEENN** for handling class imbalance  
- **Permutation Importance (PI-TopK)** for feature selection  
- **Probability calibration** (Isotonic)  
- **SHAP** for model explainability  

> 🔎 The analysis is based on the classic **UCI Heart Disease** dataset.

The goal is to build a **robust, clinically interpretable classifier** for heart disease detection
with high discrimination (AUC) and high recall (sensitivity), while strictly avoiding information leak
from the test set.

The notebook compares:

- **Baselines**
  - Logistic Regression (class-weighted)
  - SVM
  - XGBoost
  - Random Forest
- **Imbalance-aware methods**
  - RandomForest + SMOTEENN  
  - BalancedRandomForest (BRF)  
  - BRF + SMOTEENN
- **Multiple AOA / AOA++ configurations**
  - Different `max_iter` and `pop_size` to study optimization budget vs. performance
- **Leak-safe evaluation**
  - Proper train/test split  
  - Hyperparameter search, feature selection, and calibration **only on training data**  
  - Out-of-fold (OOF) predictions for threshold tuning and calibration  

The final “best” model is typically a **Balanced Random Forest with AOA++-optimized
hyperparameters, PI-TopK features, SMOTEENN (when used), and isotonic calibration**, evaluated on
both OOF train predictions and a held-out test set.

---

## 2. Dataset

We use the **Heart Disease** dataset from the UCI Machine Learning Repository.

A pre-cleaned CSV is loaded directly from a public GitHub mirror inside the notebook:

```python
url = "https://raw.githubusercontent.com/Harikishan63/Data-Learnings-/main/heart_disease_uci.csv"
df = pd.read_csv(url)
```

**Target handling**

- The code automatically detects the target column (e.g. `target`, `num`, `disease`, `output`).
- If the target has more than two classes, it is binarized as:

  ```python
  y = (y > 0).astype(int)
  ```

**Feature types**

- **Numeric columns** are automatically inferred from the dataframe dtypes.
- **Categorical columns** are inferred from known heart-disease fields
  (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, etc.) and/or `object` dtype.

---

## 3. Methods & Pipeline

### 3.1. Preprocessing

All preprocessing is handled with a `ColumnTransformer`:

- **Numeric features**
  - `SimpleImputer(strategy="median")`
  - `RobustScaler`
- **Categorical features**
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

This preprocessing block is reused across all models via unified `Pipeline` objects.

---

### 3.2. Imbalance Handling

To handle class imbalance, we use:

- **SMOTEENN** (`imblearn.combine.SMOTEENN`)  
  Hybrid over-sampling + cleaning
- **BalancedRandomForestClassifier** (`imblearn.ensemble.BalancedRandomForestClassifier`)  
  Class-balanced random forest with per-class bootstrapping

We compare three main families:

1. **RF + SMOTEENN**
2. **BRF** (BalancedRandomForest without sampler)
3. **BRF + SMOTEENN**

---

### 3.3. AOA / AOA++ Optimization

The notebook implements several **Arithmetic Optimization Algorithm (AOA)**-style searches
for hyperparameters.

We explore:

- **AOA (simple)** – tuning core tree parameters:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- **AOA++ (extended)** – tuning additional parameters:
  - `max_features` ∈ {`"sqrt"`, `"log2"`, `None`}
  - `bootstrap` ∈ {`True`, `False`}
  - `class_weight` ∈ {`None`, `"balanced"`} (for RF)

Multiple optimization budgets are tried, e.g.:

- `max_iter` ∈ {5, 10, 15, 20}
- `pop_size` ∈ {8, 16, 24, 32}

For each candidate hyperparameter set:

1. A pipeline is built:  
   **Preprocessing → (optional SMOTEENN) → Model**
2. Evaluation is done with:
   - `StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)`
   - Out-of-fold probabilities via `cross_val_predict(..., method="predict_proba")`
3. A custom score is computed:

   ```text
   score = 0.4 * AUC + 0.4 * Recall + 0.1 * F1 + 0.1 * Accuracy
   ```

We run **separate optimization loops** for:

- RF + SMOTEENN
- BRF (no sampler)
- BRF + SMOTEENN

This allows a fair comparison between different imbalance strategies.

---

### 3.4. Feature Selection via Permutation Importance (PI-TopK)

For a more compact and interpretable model, the notebook uses **Permutation Importance**:

1. Train a temporary **BRF pipeline** on the training data.
2. Use `sklearn.inspection.permutation_importance` with:
   - `scoring="roc_auc"`
   - multiple repeats (e.g. 20)
3. Rank features by mean importance and keep **Top-K** (e.g. K = 15).
4. A custom transformer `FeatureNameSelector`:
   - Fits the preprocessor to obtain `get_feature_names_out()`
   - Maps selected feature names to column indices
   - Slices the transformed matrix accordingly

This selector is inserted into the final pipelines:

> **Preprocessor → FeatureNameSelector → (optional SMOTEENN) → Calibrated Model**

---

### 3.5. Calibration & Threshold Optimization

To obtain well-calibrated probabilities and a clinically meaningful decision rule:

- Models are wrapped using:

  ```python
  CalibratedClassifierCV(base_estimator, method="isotonic", cv=5)
  ```

- For each model, we:
  - Compute **OOF probabilities** on the training set.
  - Use a `find_best_threshold` function to scan thresholds in `[0, 1]` and maximize:
    - F1-score (or, optionally, balanced accuracy).

We report performance at:

- The **default threshold** 0.50, and
- The **optimized threshold** found from OOF predictions.

---

### 3.6. SHAP Explainability

For the best calibrated BRF model with PI-TopK features:

- Extract the underlying `BalancedRandomForestClassifier` from the calibrated wrapper.
- Build a SHAP `TreeExplainer`:

  ```python
  shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
  ```

- Compute SHAP values on the (transformed) test set.
- For interpretability, log-odds outputs are mapped to approximate changes in predicted probability
  (ΔProbability).

The notebook generates, for example:

- `shap_summary_brf.png` – global summary plot  
- `shap_dependence_<feature>.png` – dependence plot for top features  
- `shap_force_brf_sample.html` – interactive force plot for a sample patient  

---

## 4. Repository Structure

Current layout of this repository:

```text
.
├── Leak-Safe BRF Pipeline Optimized with AOA and AOA++ for Heart.ipynb  # main notebook
├── requirements.txt
├── README.md
└── LICENSE
```

The notebook may optionally save additional files (e.g., SHAP figures and HTML outputs) into local
folders such as `figures/` or `shap_outputs/`. These can be created on your machine if desired but
are not required for running the code.

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

The `requirements.txt` contains (or may contain) packages such as:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
xgboost
shap

# Optional for notebooks:
ipykernel
jupyter
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

4. Open the main notebook:

   ```text
   Leak-Safe BRF Pipeline Optimized with AOA and AOA++ for Heart.ipynb
   ```

5. Run all cells from top to bottom.

   ⚠️ Blocks with large `max_iter` and `pop_size` in AOA/AOA++ can be time-consuming.
   For a quick demo, you can reduce these parameters or skip some optimization blocks.

---

## 7. Results (Summary)

On the held-out test set, the final **AOA++-optimized Balanced Random Forest** with PI-TopK
features and leak-safe training achieves:

- **AUC** ≈ **0.928** – high discrimination between patients with/without heart disease  
- **Recall (Sensitivity)** ≈ **0.961** – most positive cases are correctly detected  

This high-recall, leak-safe configuration is particularly desirable for clinical decision support,
where missing a true positive case is costly.

For full metric tables (F1, precision, accuracy at different thresholds) and ROC/PR curves,
please refer to the notebook outputs.

---

## 8. Reproducibility Notes

To improve reproducibility:

- Random seeds are set, e.g.:

  ```python
  import numpy as np, random, os

  np.random.seed(42)
  random.seed(42)
  os.environ["PYTHONHASHSEED"] = "42"
  ```

- Train/Test split:

  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, stratify=y, random_state=42
  )
  ```

- All **AOA / AOA++ runs**, permutation-importance feature selection, and calibration are performed
  **only on training data**.

- The test set is touched **only once**, after the final configuration is fixed, to obtain unbiased
  performance estimates.

---

## 9. Acknowledgments

- **Dataset**: UCI Machine Learning Repository – Heart Disease Data Set  
- **Imbalanced learning**:
  - `SMOTEENN` and `BalancedRandomForest` from `imbalanced-learn`
- **Explainability**:
  - SHAP library (`shap`)

The AOA / AOA++ implementation in this notebook is inspired by the Arithmetic Optimization
Algorithm literature and customized for this heart disease classification problem.

---

## 10. License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full terms.

---

**Note (FA)**  
بعضی از توضیحات و کامنت‌های داخل نوت‌بوک به زبان **فارسی** نوشته شده‌اند تا روند کار،
ایده‌های پشت AOA/AOA++ و تفسیر نتایج بالینی برای خوانندگان فارسی‌زبان شفاف‌تر باشد.
