# AOA_BRF
Leak-safe BRF pipeline with AOA/AOA++ for heart disease prediction
# Heart Disease Prediction with AOA-Optimized Ensembles

This repository contains an end-to-end notebook for **heart disease prediction** using
ensemble machine learning models (Random Forest, Balanced Random Forest) combined with:

- **AOA / AOA++ metaheuristic optimization** for hyperparameter search
- **SMOTEENN** for handling class imbalance
- **Permutation Importance** for feature selection
- **Probability calibration** (Isotonic)
- **SHAP** for model explainability

> 🔎 The analysis is based on the classic **UCI Heart Disease** dataset.

---

## 1. Project Overview

The goal of this project is to build a **robust, clinically interpretable classifier** for
heart disease detection.

The notebook explores:

- Standard baselines:
  - Logistic Regression (class-weighted)
  - SVM
  - XGBoost
  - Random Forest
- Class-imbalance–aware methods:
  - `RandomForest + SMOTEENN`
  - `BalancedRandomForest`
  - `BalancedRandomForest + SMOTEENN`
- Several **AOA / AOA++ configurations** (different `max_iter`, `pop_size`) to study the effect
  of optimization budget on performance.
- **Leak-safe evaluation**:
  - Proper train/test split
  - Hyperparameter search and feature selection done only on the training data
  - Out-of-fold (OOF) predictions for threshold tuning and calibration

The final "best" model in the notebook is typically a **Balanced Random Forest with
Permutation-Importance Top-K features + calibration**, evaluated both on train (OOF) and
held-out test set.

---

## 2. Dataset

We use the **Heart Disease** dataset from the UCI Machine Learning Repository.

A pre-cleaned CSV is loaded from a public GitHub mirror inside the notebook:

    url = "https://raw.githubusercontent.com/Harikishan63/Data-Learnings-/main/heart_disease_uci.csv"
    df = pd.read_csv(url)

Target handling:

- The code automatically detects the target column (e.g. `target`, `num`, `disease`, `output`).
- If the target has more than 2 classes, it is binarized as:

  `y = (y > 0).astype(int)`

Feature types:

- **Numeric columns** are automatically inferred (non-categorical / non-object).
- **Categorical columns** are inferred from known heart-disease fields (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, etc.) or dtype = `object`.

---

## 3. Methods & Pipeline

### 3.1. Preprocessing

All preprocessing is done using a `ColumnTransformer`:

- **Numeric features**
  - `SimpleImputer(strategy="median")`
  - `RobustScaler`
- **Categorical features**
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

This preprocessing block is reused across all models through unified pipelines.

---

### 3.2. Imbalance Handling

To handle class imbalance, we use:

- **SMOTEENN** (`imblearn.combine.SMOTEENN`)
  - Hybrid over-sampling + cleaning technique
- **BalancedRandomForestClassifier** (`imblearn.ensemble`)
  - Rebalanced random forest with class-wise bootstrapping

We compare three main families:

1. **RF + SMOTEENN**
2. **BRF** (BalancedRandomForest without sampler)
3. **BRF + SMOTEENN**

---

### 3.3. AOA / AOA++ Optimization

The notebook implements several versions of **Arithmetic Optimization Algorithm (AOA)**–style search
for hyperparameters.

We explore both:

- **AOA (simple)**: tuning core tree parameters:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- **AOA++ (extended)**: tuning additional parameters:
  - `max_features` ∈ {`"sqrt"`, `"log2"`, `None`}
  - `bootstrap` ∈ {True, False}
  - `class_weight` ∈ {None, `"balanced"`} (for RF)

Different blocks correspond to different budgets:

- `max_iter` ∈ {5, 10, 15, 20}
- `pop_size` ∈ {8, 16, 24, 32}

For each candidate hyperparameter set:

- A pipeline is built:
  - Preprocessing → (optional SMOTEENN) → Model
- Evaluated with:
  - `StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)`
  - Out-of-fold predicted probabilities (`cross_val_predict(..., method="predict_proba")`)
- A custom score is computed:

  score = 0.4 * AUC + 0.4 * Recall + 0.1 * F1 + 0.1 * Accuracy

We run **separate optimization loops** for:

- RF + SMOTEENN
- BRF (no sampler)
- BRF + SMOTEENN

This allows a fair comparison between different imbalance strategies.

---

### 3.4. Feature Selection via Permutation Importance (PI-TopK)

For a more compact and interpretable model, the notebook uses **Permutation Importance**:

1. A temporary **BalancedRandomForest** pipeline is trained (on *train* data).
2. `sklearn.inspection.permutation_importance` is used with:
   - `scoring="roc_auc"`
   - multiple repeats (e.g. 20)
3. Features are ranked by mean importance, and **Top-K** (e.g. K = 15) are selected.
4. A custom transformer `FeatureNameSelector`:
   - Fits the preprocessor to get `get_feature_names_out()`
   - Maps selected feature names to their column indices
   - Slices the transformed matrix accordingly

This selector is inserted into the final pipelines:

`FeatureNameSelector → (optional SMOTEENN) → Calibrated Model`

---

### 3.5. Calibration & Threshold Optimization

To produce well-calibrated probabilities and a clinically meaningful decision boundary:

- Models are wrapped in:
  - `CalibratedClassifierCV(base_estimator, method="isotonic", cv=5)`
- For each model:
  - We compute **out-of-fold probabilities** on training data.
  - A dedicated `find_best_threshold` function scans thresholds in `[0, 1]` to maximize:
    - F1 score (or optionally balanced accuracy).
- We report performance at:
  - **Default threshold 0.50**
  - **Optimized threshold** chosen based on OOF predictions.

---

### 3.6. SHAP Explainability

For the best-performing calibrated BRF model with PI-TopK:

- We extract the underlying `BalancedRandomForestClassifier` from the calibrated wrapper.
- Build a SHAP `TreeExplainer` with:
  - `feature_perturbation="tree_path_dependent"`
  - `model_output="raw"`
- Compute SHAP values on the processed test set.
- For better interpretability, SHAP value in log-odds is approximated as changes in probability (ΔProbability).

The notebook generates:

- **SHAP summary plot** (dot plot) for the top 15 features
- **SHAP dependence plot** for the most important feature
- An interactive **force plot** saved as an HTML file

Typical outputs (saved by the notebook):

- `shap_summary_brf.png`
- `shap_dependence_<feature>.png`
- `shap_force_brf_sample.html`

---

## 4. Repository Structure (Suggested)

You can organize the repository as follows:

    .
    ├── notebooks/
    │   └── heart_disease_aoa.ipynb       # main notebook (e.g. your final edition)
    ├── figures/
    │   ├── shap_summary_brf.png
    │   ├── shap_dependence_*.png
    │   └── ...
    ├── shap_outputs/
    │   └── shap_force_brf_sample.html
    ├── requirements.txt
    └── README.md

Feel free to adjust paths and names according to your setup.

---

## 5. Installation

It is recommended to use a virtual environment.

    python -m venv .venv
    # On Linux/Mac:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate

    pip install -r requirements.txt

Example `requirements.txt`:

    numpy
    pandas
    matplotlib
    seaborn
    scipy
    scikit-learn
    imbalanced-learn
    xgboost
    shap

Optional (for running notebooks):

    ipykernel
    jupyter

---

## 6. How to Run

1. **Clone** the repository:

       git clone https://github.com/<your-username>/<repo-name>.git
       cd <repo-name>

2. **Install dependencies**:

       pip install -r requirements.txt

3. **Launch Jupyter**:

       jupyter notebook
   or:

       jupyter lab

4. Open the main notebook (for example):  
   `notebooks/heart_disease_aoa.ipynb`

5. Run all cells from top to bottom.  
   - Note: AOA / AOA++ blocks with large `max_iter` and `pop_size` can be time-consuming.
     For a quick demo, you can reduce these parameters or skip some blocks.

---

## 7. Results (To Be Filled)

The notebook prints detailed metrics for:

- **Train (OOF)** at:
  - threshold = 0.50
  - optimized threshold
- **Test set** at:
  - threshold = 0.50
  - optimized threshold

For each of the main model families:

- RF + SMOTEENN
- BRF
- BRF + SMOTEENN

You can summarize your final results here, for example:

    Model: BRF (PI-Top15 + Calibration)
    Test (opt threshold):
    - AUC       : ...
    - F1        : ...
    - Recall    : ...
    - Precision : ...
    - Accuracy  : ...
    Threshold   : ...

You can also add ROC and Precision-Recall curves exported from the notebook as images.

---

## 8. Reproducibility Notes

To ensure reproducible experiments:

- Random seeds are set, e.g.:

      import numpy as np, random, os

      np.random.seed(42)
      random.seed(42)
      os.environ["PYTHONHASHSEED"] = "42"

- Train/Test split:

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, stratify=y, random_state=42
      )

- All AOA / AOA++ runs, feature selection, and calibration are performed on **training data only** in the leak-safe blocks.
- Evaluation on the test set is strictly separated and only done after the final model/pipeline is fixed.

---

## 9. Acknowledgments

- **Dataset**: UCI Machine Learning Repository – Heart Disease Data Set.
- **Imbalanced learning**:
  - `SMOTEENN` and `BalancedRandomForest` from `imbalanced-learn`
- **Explainability**:
  - SHAP library: `shap`

The implementation of AOA / AOA++ in this notebook is inspired by the Arithmetic Optimization
Algorithm and customized for this particular heart disease classification problem.

---

## 10. License

Specify your preferred license here, for example:

- **MIT License** – simple and permissive, suitable for research code.
- Or any other license you prefer (Apache-2.0, GPL-3.0, etc.).

---

**Note (FA):**  
Some comments and explanations inside the notebook are written in **Persian (Farsi)** to make
the workflow, ideas behind AOA/AOA++, and the interpretation of clinical results clearer for
Farsi-speaking readers.
