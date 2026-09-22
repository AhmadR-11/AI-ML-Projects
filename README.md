# AI & Machine Learning Projects

A curated collection of end-to-end machine learning projects covering **regression**, **classification**, **NLP**, **deep learning**, and **exploratory data analysis** workflows. Each project demonstrates a complete pipeline — from data cleaning and visualization through model training, evaluation, and deployment.

---

## Projects

| # | Project | Type | Stack | Interface |
|---|---------|------|-------|-----------|
| 1 | [House Price Prediction](./House%20Price%20Prediction/) | Regression | XGBoost · LightGBM · Ridge · Random Forest · Ensemble | Streamlit Web App + CLI |
| 2 | [Iris Flower Classification](./Iris-Flower-Classification/) | Classification | Logistic Regression · KNN · SVM · Decision Tree · Random Forest | Jupyter Notebook |
| 3 | [Unemployment Analysis](./Unemployment-Analysis/) | EDA & Visualization | pandas · Matplotlib · Seaborn | Python Script (CLI) |
| 4 | [Car Price Prediction](./Car-Price-Prediction/) | Regression & EDA | Linear Regression · Random Forest · Gradient Boosting | Dash Web App + CLI |
| 5 | [Email Spam Detection](./Email-Spam-Detection/) | NLP Classification | Naive Bayes · Logistic Regression · SVM · TF-IDF | Streamlit Web App |
| 6 | [MNIST Digit Recognition](./MNIST/) | Deep Learning | TensorFlow · Keras · CNN (Baseline + Advanced) | Jupyter Notebook |

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

## 6 · MNIST Digit Recognition

An end-to-end deep learning pipeline for handwritten digit classification on the Kaggle MNIST dataset (42,000 training + 28,000 test images). The project builds, trains, and compares two CNN architectures — a Baseline CNN (421K parameters) and an Advanced Deep CNN (872K parameters with Batch Normalization, multi-stage Dropout, and real-time data augmentation) — achieving **99.62% validation accuracy**. Includes comprehensive EDA, error analysis, Conv2D feature map visualization, and a Kaggle-ready submission file.

### Highlights

- **Two CNN Architectures** — Baseline CNN (8 layers, 421,642 params) for rapid prototyping and an Advanced Deep CNN (15 layers, 872,426 params) with double-convolution blocks, Batch Normalization, and progressive Dropout (0.25 → 0.4 → 0.5).
- **99.62% Validation Accuracy** — Advanced CNN achieves 99.62% accuracy (val_loss: 0.0157) on a 4,200-sample stratified validation set, with EarlyStopping restoring the best weights.
- **Real-Time Data Augmentation** — Conservative digit-safe augmentations via `ImageDataGenerator` (±10° rotation, ±10% shifts and zoom) — explicitly avoids flips to preserve digit semantics (e.g., 6 vs. 9).
- **Training Callbacks** — EarlyStopping (`patience=5`), ModelCheckpoint (saves best `.keras` weights), and ReduceLROnPlateau for adaptive learning rate scheduling.
- **Comprehensive EDA** — Class distribution bar chart, 5×10 sample digit grid, average pixel intensity images per class, and pixel statistics summary.
- **Preprocessing Pipeline** — Pixel normalization (0–255 → 0.0–1.0), reshape to 4D tensors (28×28×1), one-hot encoding, 90/10 stratified train/validation split, and processed arrays saved as `.npy` files.
- **Error Analysis** — Top-20 misclassified samples visualization, top confused digit pairs analysis (7→2, 9→4, 4→9), and side-by-side confusion matrix heatmaps for both models.
- **Feature Map Visualization** — Extracts and displays 16 feature maps from the first Conv2D layer to illustrate learned edge and texture detectors.
- **Kaggle Submission** — Generates a 28,000-row `submission.csv` (`ImageId`, `Label`) from the best-performing model for direct Kaggle upload.
- **Automated Scaffolding** — `setup_project.py` generates the full directory tree with `.gitkeep` files for Git tracking.

### Model Comparison

| Metric | Baseline CNN | Advanced CNN |
|--------|-------------|-------------|
| **Parameters** | 421,642 | 872,426 |
| **Training Epochs** | 22 | 23 |
| **Best Val Accuracy** | 99.14% | **99.62%** |
| **Best Val Loss** | 0.0375 | **0.0157** |
| **Training Time** | 142 s | 648 s |
| **Model Size** | 4.86 MB | 10.06 MB |

### Directory Structure

```
MNIST/
├── setup_project.py              # Automated project structure generator
├── requirements.txt              # Python dependencies (TensorFlow, scikit-learn, Kaggle)
├── .gitignore                    # Excludes data, models, credentials, IDE configs
├── notebooks/
│   └── 01_eda.ipynb              # Complete notebook (59 cells: EDA → preprocessing →
│                                 #   CNN architectures → training → evaluation → submission)
├── data/
│   ├── raw/                      # Raw Kaggle CSVs (train.csv, test.csv)
│   └── processed/                # Normalized .npy arrays (X_train, X_val, X_test, y_train, y_val)
├── models/                       # Saved model weights (.h5, .keras)
│   ├── baseline_model.h5         # Baseline CNN weights
│   ├── advanced_model.h5         # Advanced CNN weights
│   └── best_model.h5             # Best-performing model (Advanced CNN)
├── outputs/
│   ├── submission.csv            # Kaggle submission (28,000 predictions)
│   ├── plots/                    # Generated visualizations (PNG)
│   │   ├── digit_sample_grid.png
│   │   ├── digit_class_distribution.png
│   │   ├── average_digit_images.png
│   │   ├── training_curves.png
│   │   ├── confusion_matrices_comparison.png
│   │   ├── misclassified_top20.png
│   │   ├── top_confused_digit_pairs.png
│   │   ├── feature_maps_conv1.png
│   │   └── test_sample_predictions.png
│   └── reports/                  # Metrics and logs
│       ├── final_report.md
│       ├── model_comparison_summary.csv
│       ├── baseline_training_log.csv
│       └── advanced_training_log.csv
├── src/                          # Reusable Python modules
└── .mnist_env/                   # Local virtual environment
```

### Quick Start

```bash
cd MNIST

# Activate the existing virtual environment
source .mnist_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Regenerate the project structure
python setup_project.py

# Launch the notebook
jupyter notebook notebooks/01_eda.ipynb
```

### Notebook Pipeline (59 cells)

| Phase | Cells | Description |
|-------|-------|-------------|
| 📊 **EDA** | 1–12 | Load Kaggle CSVs, class distribution, 5×10 sample grid, average digit images, pixel stats |
| 🧹 **Preprocessing** | 13–27 | Normalize, reshape (28×28×1), one-hot encode, 90/10 stratified split, data augmentation, save `.npy` |
| 🧠 **Model Architecture** | 28–37 | Build Baseline CNN (8 layers) + Advanced Deep CNN (15 layers), compile with Adam + categorical cross-entropy |
| 🏋️ **Training** | 38–43 | Train both models with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, plot training curves |
| 📈 **Evaluation** | 44–47 | Classification reports, confusion matrices, Kaggle submission generation |
| 🔬 **Error Analysis** | 48–58 | Misclassified samples, top confused pairs, Conv2D feature maps, final report generation |

### Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | TensorFlow 2.x, Keras (Sequential API) |
| ML / Utilities | scikit-learn (train_test_split, classification_report, confusion_matrix) |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Data Source | Kaggle API |

---

## Repository Structure

```
AI-ML-Projects/
├── House Price Prediction/       # Regression project (Streamlit + CLI)
├── Iris-Flower-Classification/   # Classification project (Jupyter Notebook)
├── Unemployment-Analysis/        # EDA & visualization project (Python Script)
├── Car-Price-Prediction/         # Regression & EDA project (Dash Web App)
├── Email-Spam-Detection/         # NLP classification project (Streamlit Web App)
├── MNIST/                        # Deep learning project (TensorFlow CNN)
└── README.md                     # ← You are here
```

---

## License

This repository is intended for educational and portfolio purposes.