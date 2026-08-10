# AI & Machine Learning Projects

A curated collection of end-to-end machine learning projects covering **regression**, **classification**, **NLP**, and **exploratory data analysis** workflows. Each project demonstrates a complete pipeline — from data cleaning and visualization through model training, evaluation, and deployment.

---

## Projects

| # | Project | Type | Stack | Interface |
|---|---------|------|-------|-----------|
| 1 | [House Price Prediction](./House%20Price%20Prediction/) | Regression | XGBoost · LightGBM · Ridge · Random Forest · Ensemble | Streamlit Web App + CLI |
| 2 | [Iris Flower Classification](./Iris-Flower-Classification/) | Classification | Logistic Regression · KNN · SVM · Decision Tree · Random Forest | Jupyter Notebook |
| 3 | [Unemployment Analysis](./Unemployment-Analysis/) | EDA & Visualization | pandas · Matplotlib · Seaborn | Python Script (CLI) |
| 4 | [Car Price Prediction](./Car-Price-Prediction/) | Regression & EDA | Linear Regression · Random Forest · Gradient Boosting | Dash Web App + CLI |
| 5 | [Email Spam Detection](./Email-Spam-Detection/) | NLP Classification | Naive Bayes · Logistic Regression · SVM · TF-IDF | Streamlit Web App |

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

An interactive Dash web dashboard for predicting automobile prices. The project pairs a reusable data-cleaning and model-training backend (`analysis.py`) with a polished, dark-themed Plotly dashboard (`app.py`) featuring five tabs — EDA, Correlations, Model Performance, Feature Importance, and a live Price Predictor.

### Highlights

- **Interactive Dash Dashboard** — Five-tab dark-themed web app built with Dash, Dash Bootstrap Components, and Plotly, featuring summary cards (total cars, average price, best model, R² score) and responsive layouts.
- **Three Regression Models** — Trains Linear Regression, Random Forest, and Gradient Boosting side by side; benchmarks all three on MAE, RMSE, and R² with annotated bar charts and actual-vs-predicted scatter plots (with ideal-fit line).
- **Feature Importance Comparison** — Ranks the top 15 features for both Random Forest and Gradient Boosting, plus a grouped bar chart comparing both models.
- **Rich EDA Visualizations** — Price distribution with marginal box plot, fuel-type box plot, price-vs-age scatter with OLS trendline, top-10 brands bar chart, and transmission violin plot — all interactive via Plotly.
- **Correlation Analysis** — Full-feature heatmap, KMs-driven scatter with trendline, and average price by year line chart.
- **Live Price Predictor** — Users select brand, fuel type, transmission, seller type, car age, and KMs driven; all three models produce predictions displayed as alerts, with a gauge chart highlighting the best estimate.
- **Robust Cleaning Backend** — Deduplication, null imputation (mode for categorical, median for numeric), IQR-based outlier removal, brand extraction from `CarName`, `Car_Age` derivation, and one-hot encoding with `drop_first=True`.
- **Custom CSS Theming** — Deep navy gradient backgrounds, accent color (`#00b4d8`), hover-lift cards, and styled sliders via `assets/style.css`.

### Directory Structure

```
Car-Price-Prediction/
├── app.py                       # Dash web dashboard (5 tabs, Plotly charts, live predictor)
├── analysis.py                  # Data cleaning, encoding, and model training module
├── assets/
│   └── style.css                # Custom dark-theme CSS for the dashboard
├── data/
│   └── CarPrice_Assignment.csv  # 205 vehicles × 26 attributes
├── outputs/                     # Pre-generated static charts (PNG)
└── .venv/                       # Local virtual environment
```

### Quick Start

```bash
cd Car-Price-Prediction

# Activate the existing virtual environment
source .venv/bin/activate

# Install dependencies (if not already present)
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn

# Launch the dashboard
python app.py
# Opens at http://127.0.0.1:8050
```

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 **EDA** | Price distribution, fuel-type box plot, price-vs-age scatter, top-10 brands, transmission violin plot |
| 🔥 **Correlations** | Full-feature heatmap, KMs-driven scatter with OLS trendline, average price by year |
| 🤖 **Model Performance** | Metrics table + MAE / RMSE / R² bar charts + actual-vs-predicted scatter for each model |
| 🎯 **Feature Importance** | Top-15 features for Random Forest & Gradient Boosting + side-by-side comparison chart |
| 🔮 **Price Predictor** | Input form (brand, fuel, transmission, seller, age, KMs) → predictions from all models + gauge chart |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / Modeling | scikit-learn (LinearRegression, RandomForest, GradientBoosting) |
| Data | pandas, NumPy |
| Web Interface | Dash, Dash Bootstrap Components |
| Visualization | Plotly |

---

## 5 · Email Spam Detection

A binary NLP classifier that distinguishes spam from legitimate (ham) emails using TF-IDF feature extraction and three machine learning models. The project includes both a detailed Jupyter notebook walkthrough and a polished, dark-themed Streamlit dashboard with five pages — Overview, EDA, WordClouds, Model Performance, and a Live Spam Detector.

### Highlights

- **NLP Text Preprocessing** — Lowercasing, URL/email removal, punctuation stripping, digit removal, stopword filtering, and Porter stemming via NLTK. Gracefully falls back to a regex-based tokenizer when NLTK data is unavailable.
- **TF-IDF Vectorization** — Extracts up to 5,000 features with unigram + bigram `ngram_range=(1, 2)`, `min_df=2`, and sublinear TF scaling for robust term weighting.
- **Three Classifiers** — Multinomial Naive Bayes (α = 0.1), Logistic Regression, and SVM (linear kernel) trained on an 80/20 stratified split; all evaluated on Accuracy, Precision, Recall, and F1 Score.
- **Interactive Streamlit Dashboard** — Five-page app with sidebar navigation, dataset stats, best-model badge, and custom GitHub-inspired dark CSS theme (`assets/style.css`).
- **EDA Visualizations** — Class distribution bar + pie chart, character/word count box plots and histograms by label, plus descriptive statistics grouped by class.
- **WordCloud Analysis** — Side-by-side word clouds (spam in reds, ham in greens) and top-20 most frequent word bar charts for each class.
- **Confusion Matrices & Metrics** — Per-model confusion matrix heatmaps, four side-by-side bar charts (Accuracy, Precision, Recall, F1), and score cards with best-model highlighting.
- **Live Spam Detector** — Text input with model selector, probability breakdown, confidence bar, preprocessed-text preview, and batch prediction across all three models.
- **Jupyter Notebook** — Step-by-step walkthrough: data loading → preprocessing → word clouds → TF-IDF → model training → evaluation → prediction.

### Directory Structure

```
Email-Spam-Detection/
├── app.py                    # Streamlit dashboard (5 pages, live predictor)
├── Spam_Detection.ipynb      # Full notebook walkthrough (33 cells)
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes .venv, nltk_data, outputs, __pycache__
├── assets/
│   └── style.css             # Custom dark-theme CSS (GitHub-inspired)
├── data/
│   ├── spam.csv              # Primary dataset (~5,500 messages, ham/spam labels)
│   └── Spam_SMS.csv          # Alternate dataset
├── nltk_data/                # Local NLTK corpora (stopwords, punkt tokenizer)
├── outputs/                  # Pre-generated static charts (PNG)
└── .venv/                    # Local virtual environment
```

### Quick Start

```bash
cd Email-Spam-Detection

# Activate the existing virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

Alternatively, open `Spam_Detection.ipynb` in Jupyter for a step-by-step notebook walkthrough.

### Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Dataset stats, model results summary table, sample spam/ham messages |
| 📊 **EDA & Visualizations** | Class distribution, character/word count analysis, descriptive statistics |
| ☁️ **WordClouds** | Spam vs. ham word clouds + top-20 most frequent words bar charts |
| 🤖 **Model Performance** | Score cards, 4-metric comparison bars, confusion matrix heatmaps |
| 🔮 **Live Spam Detector** | Real-time prediction with model selector, confidence bar, batch testing |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / Modeling | scikit-learn (MultinomialNB, LogisticRegression, SVM), TF-IDF |
| NLP | NLTK (stopwords, PorterStemmer, word_tokenize) |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web Interface | Streamlit |

---

## Repository Structure

```
AI-ML-Projects/
├── House Price Prediction/       # Regression project (Streamlit + CLI)
├── Iris-Flower-Classification/   # Classification project (Jupyter Notebook)
├── Unemployment-Analysis/        # EDA & visualization project (Python Script)
├── Car-Price-Prediction/         # Regression & EDA project (Dash Web App)
├── Email-Spam-Detection/         # NLP classification project (Streamlit Web App)
└── README.md                     # ← You are here
```

---

## License

This repository is intended for educational and portfolio purposes.