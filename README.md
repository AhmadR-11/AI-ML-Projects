# AI & Machine Learning Projects

A curated collection of end-to-end machine learning projects covering both **regression** and **classification** workflows. Each project demonstrates a complete pipeline — from exploratory data analysis and feature engineering through model training, evaluation, and deployment.

---

## Projects

| # | Project | Type | Stack | Interface |
|---|---------|------|-------|-----------|
| 1 | [House Price Prediction](./House%20Price%20Prediction/) | Regression | XGBoost · LightGBM · Ridge · Random Forest · Ensemble | Streamlit Web App + CLI |
| 2 | [Iris Flower Classification](./Iris-Flower-Classification/) | Classification | Logistic Regression · KNN · SVM · Decision Tree · Random Forest | Jupyter Notebook |

---

## 1 · House Price Prediction

An automated regression pipeline that accepts any CSV dataset, performs intelligent preprocessing, trains multiple gradient-boosted models, and ranks them on a live leaderboard. The best model is persisted and served through an interactive prediction interface.

### Highlights

- **Ensemble Stacking** — Combines XGBoost, LightGBM, and Ridge Regression via a weighted `VotingRegressor` (45/45/10 split) to reduce individual model variance.
- **Automatic Skew Correction** — Detects right-skewed targets (skewness > 0.75) and applies `log1p` transformation; reverses it at prediction time with `expm1`.
- **Robust Preprocessing** — Median/mode imputation for missing values, Z-score outlier capping at ±3σ, per-feature log normalization, one-hot encoding with high-cardinality filtering.
- **Feature Engineering** (CLI pipeline) — Derives domain features such as `TotalHouseAge`, `TotalSquareFootage`, and `TotalPorchArea` when applicable columns are present.
- **Dual Execution Modes** — Streamlit web UI for interactive use; headless CLI pipeline with `GridSearchCV` hyperparameter tuning.

### Directory Structure

```
House Price Prediction/
├── app.py                  # Streamlit web application (train + predict)
├── main.py                 # CLI pipeline orchestrator
├── requirements.txt        # Python dependencies
├── README.md               # Project-level documentation
├── data/
│   └── raw/                # Source CSV datasets
├── models/                 # Serialized models, scalers, and metadata (.pkl)
├── notebooks/
│   └── 01_EDA.ipynb        # Exploratory data analysis notebook
├── src/
│   ├── data_loader.py      # Dataset loading (falls back to California Housing)
│   ├── preprocessing.py    # Imputation, outlier capping, encoding, scaling
│   ├── train_model.py      # Multi-model training with GridSearchCV
│   ├── evaluate.py         # Benchmarking scorecard and winner analysis
│   └── predict.py          # Sample predictions using the saved model
└── visualizations/         # Generated charts (actual vs. predicted, residuals)
```

### Quick Start

```bash
# 1. Navigate to the project
cd "House Price Prediction"

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. (macOS only) Install OpenMP for XGBoost/LightGBM
brew install libomp

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Option A — Streamlit UI** *(recommended)*

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

1. Upload any CSV and select the target column.
2. The engine trains Ridge, Random Forest, XGBoost, LightGBM, and an Ensemble — then displays a ranked leaderboard.
3. Switch to the **Dynamic Test Predictions** tab to generate predictions using the winning model.

**Option B — CLI Pipeline**

```bash
python main.py
```

Runs the full pipeline (data loading → preprocessing → GridSearchCV training → evaluation) and prints results to the terminal.

### Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / Modeling | scikit-learn, XGBoost, LightGBM |
| Data | pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Serialization | joblib |

---

## 2 · Iris Flower Classification

A notebook-based classification project using the classic Iris dataset. It walks through the full data-science workflow — EDA, visualization, feature selection, model training, and evaluation — comparing five classifiers side by side.

### Highlights

- **Exploratory Data Analysis** — Pair plots, box plots by species, and a correlation heatmap to identify the most discriminative features (petal length and petal width).
- **Five Classifiers Compared** — Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, and Random Forest, all evaluated on accuracy, confusion matrices, and classification reports.
- **Stratified Splitting** — 80/20 train-test split with stratification to preserve class balance.
- **Feature Scaling** — StandardScaler applied to normalize feature distributions before training.

### Directory Structure

```
Iris-Flower-Classification/
├── Iris_Classification.ipynb   # Complete notebook (EDA → training → evaluation)
└── .venv/                      # Local virtual environment
```

### Quick Start

```bash
cd Iris-Flower-Classification

# Activate the existing virtual environment
source .venv/bin/activate

# Launch the notebook
jupyter notebook Iris_Classification.ipynb
```

### Key Results

| Model | Approx. Accuracy | Notes |
|-------|-------------------|-------|
| Logistic Regression | 93–97 % | Strong baseline, good generalization |
| K-Nearest Neighbors | 93–97 % | Simple and effective, sensitive to *k* |
| SVM | 93–100 % | Excellent on small, well-separated data |
| Decision Tree | 93–97 % | Interpretable but prone to overfitting |
| Random Forest | 93–97 % | Robust ensemble, handles feature noise well |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / Modeling | scikit-learn (LogisticRegression, KNN, SVC, DecisionTree, RandomForest) |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Repository Structure

```
AI-ML-Projects/
├── House Price Prediction/       # Regression project (Streamlit + CLI)
├── Iris-Flower-Classification/   # Classification project (Jupyter Notebook)
└── README.md                     # ← You are here
```

---

## License

This repository is intended for educational and portfolio purposes.