# AI & Machine Learning Projects

A curated collection of end-to-end machine learning projects covering **regression**, **classification**, and **exploratory data analysis** workflows. Each project demonstrates a complete pipeline — from data cleaning and visualization through model training, evaluation, and deployment.

---

## Projects

| # | Project | Type | Stack | Interface |
|---|---------|------|-------|-----------|
| 1 | [House Price Prediction](./House%20Price%20Prediction/) | Regression | XGBoost · LightGBM · Ridge · Random Forest · Ensemble | Streamlit Web App + CLI |
| 2 | [Iris Flower Classification](./Iris-Flower-Classification/) | Classification | Logistic Regression · KNN · SVM · Decision Tree · Random Forest | Jupyter Notebook |
| 3 | [Unemployment Analysis](./Unemployment-Analysis/) | EDA & Visualization | pandas · Matplotlib · Seaborn | Python Script (CLI) |
| 4 | [Car Price Prediction](./Car-Price-Prediction/) | EDA & Feature Engineering | pandas · Matplotlib · Seaborn · NumPy | Python Script (CLI) |

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

## 3 · Unemployment Analysis

An exploratory data analysis project that investigates unemployment trends across Indian states and regions. The script ingests government-sourced CSV data, performs automated cleaning, and generates a suite of publication-ready visualizations — including a pre-COVID vs. post-COVID impact comparison.

### Highlights

- **Automated Column Detection** — Dynamically identifies date, region, and unemployment-rate columns by name matching, making the script adaptable to similar CSV layouts.
- **Data Cleaning Pipeline** — Strips whitespace from headers, parses dates with day-first formatting, drops duplicates, removes rows with invalid dates, and derives `Month` / `Year` features.
- **Region-Level EDA** — Computes and prints average unemployment rates grouped by region, month, and year.
- **Time-Series Visualization** — Line chart tracking unemployment over time for the top 3 most-affected regions.
- **Top-10 Bar Chart** — Horizontal bar chart ranking the 10 regions with the highest average unemployment rates.
- **Correlation Heatmap** — Visualizes relationships between unemployment rate, estimated employed count, and labour participation rate.
- **COVID-19 Impact Comparison** — Splits data at March 2020 and plots a pre-COVID vs. post-COVID bar chart with percentage-change annotation.

### Directory Structure

```
Unemployment-Analysis/
├── analysis.py                             # Full analysis script (clean → EDA → plots)
├── data/
│   ├── Unemployment in India.csv           # Primary dataset (region, date, rate, employed, area)
│   └── Unemployment_Rate_upto_11_2020.csv  # Extended dataset with coordinates
├── outputs/                                # Generated charts (PNG, 150 DPI)
│   ├── timeseries_chart.png
│   ├── top10_bar_chart.png
│   ├── correlation_heatmap.png
│   └── covid_comparison.png
└── .venv/                                  # Local virtual environment
```

### Quick Start

```bash
cd Unemployment-Analysis

# Activate the existing virtual environment
source .venv/bin/activate

# Install dependencies (if not already present)
pip install pandas matplotlib seaborn

# Run the analysis
python analysis.py
```

All charts are saved to the `outputs/` directory.

### Generated Visualizations

| Chart | Description |
|-------|-------------|
| `timeseries_chart.png` | Unemployment rate over time for the top 3 regions |
| `top10_bar_chart.png` | Top 10 regions ranked by average unemployment rate |
| `correlation_heatmap.png` | Correlation between unemployment, employment, and labour participation |
| `covid_comparison.png` | Pre-COVID vs. post-COVID average unemployment rate comparison |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas |
| Visualization | Matplotlib, Seaborn |

---

## 4 · Car Price Prediction

An exploratory data analysis and feature engineering project built around an automotive pricing dataset (205 vehicles, 26 attributes). The script walks through a structured five-phase workflow — data loading, cleaning, feature engineering, visualization, and encoding — producing publication-ready charts and a fully encoded DataFrame ready for downstream modeling.

### Highlights

- **Structured Pipeline** — Five clearly separated phases: data loading → cleaning → feature engineering → EDA visualizations → encoding & correlation.
- **IQR Outlier Removal** — Detects and removes price outliers using the interquartile range method (1.5× IQR bounds).
- **Feature Engineering** — Derives `Car_Age` from the current year and extracts the `Brand` name from the `CarName` column with title-case normalization.
- **Categorical Standardization** — Trims whitespace and applies consistent title-casing across 9 categorical columns (fuel type, aspiration, body style, drive wheel, etc.).
- **Six Visualizations** — Price distribution histogram with KDE, fuel-type box plot, price-vs-age scatter plot (colored by fuel type), top-10 brands bar chart, transmission box plot, and a full-feature correlation heatmap.
- **One-Hot Encoding** — Expands categorical columns (`Fuel_Type`, `Seller_Type`, `Transmission`, `Brand`) via `pd.get_dummies` with `drop_first=True`, preparing the data for regression models.
- **Top Feature Correlation** — Ranks the 10 numeric features most correlated with selling price after encoding.

### Directory Structure

```
Car-Price-Prediction/
├── analysis.py                  # Full analysis script (5 phases)
├── data/
│   └── CarPrice_Assignment.csv  # 205 vehicles × 26 attributes
├── outputs/                     # Generated charts (PNG)
│   ├── 01_price_distribution.png
│   ├── 02_price_vs_fuel.png
│   ├── 04_top10_brands.png
│   └── 06_correlation_heatmap.png
└── .venv/                       # Local virtual environment
```

### Quick Start

```bash
cd Car-Price-Prediction

# Activate the existing virtual environment
source .venv/bin/activate

# Install dependencies (if not already present)
pip install pandas numpy matplotlib seaborn

# Run the analysis
python analysis.py
```

All charts are saved to the `outputs/` directory.

### Generated Visualizations

| Chart | Description |
|-------|-------------|
| `01_price_distribution.png` | Selling price distribution with KDE overlay and skewness metric |
| `02_price_vs_fuel.png` | Box plot comparing selling price across fuel types |
| `04_top10_brands.png` | Top 10 brands ranked by average selling price |
| `06_correlation_heatmap.png` | Full-feature correlation heatmap (post-encoding) |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Repository Structure

```
AI-ML-Projects/
├── House Price Prediction/       # Regression project (Streamlit + CLI)
├── Iris-Flower-Classification/   # Classification project (Jupyter Notebook)
├── Unemployment-Analysis/        # EDA & visualization project (Python Script)
├── Car-Price-Prediction/         # EDA & feature engineering project (Python Script)
└── README.md                     # ← You are here
```

---

## License

This repository is intended for educational and portfolio purposes.