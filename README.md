# 🧠 AI & Machine Learning Projects Portfolio

Welcome to my central repository for Artificial Intelligence and Machine Learning architecture! This repository serves as a growing portfolio of my end-to-end ML engineering—from data mathematics and algorithmic preprocessing to deploying full-stack Model Ensembles and UI integrations.

---

## 📂 Projects Overview

Below is an index of the advanced ML platforms constructed in this repository:

### 1. [House Price Prediction (AutoML Edition) 🏡](./House%20Price%20Prediction/)
A heavily optimized, robust Kaggle-Level Machine Learning pipeline built to predict continuous targets (like Real Estate prices). It features an interactive **Streamlit Grandmaster Algorithm** that dynamically trains, benchmarks, and tests advanced gradient-boosted algorithms iteratively on any generic dataset instantly.

---

## 🚀 Deep Dive: Project 1 - Kaggle-Tier AutoML Platform

While initially designed strictly for classical House Price datasets, this project was refactored into a **Generic Top-20% Kaggle Tier AutoML Engine**. The platform empowers users to upload *any* CSV dataset, select a computational Target, and watch the system physically deploy elite Machine Learning algorithms (XGBoost, LightGBM, and Ensembled Voting Stacks) fully autonomously.

### 🌟 Key Mathematical Features
* **Gradient Boosting & Stacking**: Autonomously fuses `XGBoost`, `LightGBM`, and `Ridge` Regression structures together using an optimized `VotingRegressor` algorithm, mechanically canceling out individual algorithm weaknesses.
* **Skew Transformation Matrices**: Evaluates massive mathematical variance globally. If the target column is right-skewed >0.75, it instantly converts it utilizing Inverse `np.log1p` calculations to protect standard deviation continuity.
* **Automatic Inference Imputation**: Completely resolves empty row data anomalies using structural Medians & categorical Modes dynamically without bleeding standard-deviation variance.
* **Dynamic Pipeline Roadmap**: The web interface visualizes the literal algorithmic timeline exactly as it operates inside your MacOS CPU, printing operations in real-time.

---

### 🏗️ Complete Directory Structure
```text
House Price Prediction/
│
├── data/raw/              # Unmodified CSV datasets correctly loaded (train.csv, etc)
├── models/                # Serialized architecture parameters and AI brain weights (.pkl)
├── notebooks/             # Scratchpad Jupyter Notebooks for manual Exploratory Data Analysis (EDA)
│
├── src/                   # Procedural Python Terminal Pipeline (Classical Execution)
│   ├── data_loader.py     # Pulls and validates datasets locally
│   ├── preprocessing.py   # Cleans missing values, drops outliers, outputs matrix arrays
│   ├── train_model.py     # Automates `GridSearchCV` hyperparameter tuning across 4 discrete algorithms
│   └── evaluate.py        # Generates metrics (MAE, RMSE, R2 Score) and visual accuracy charts
│
├── visualizations/        # Standard Output directory for PNG regression charts from evaluate.py
├── app.py                 # 🌟 Elite Streamlit Kaggle AutoML Web UI (The Main Event)
├── main.py                # Console Pipeline Orchestrator Sequence
├── requirements.txt       # Critical architecture dependencies (xgboost, lightgbm, etc.)
└── README.md              # Project-level documentation
```

### 🛠️ Installation & Architecture Build

To run this platform locally via MacOS/Linux, execute the following strict terminal sequences:

**1. Create & Activate a Virtual Environment:**
```bash
cd "House Price Prediction"
python3 -m venv venv
source venv/bin/activate
```

**2. Install Core C++ Dependencies (Crucial for Mac):**
Because XGBoost/LightGBM use C++ to execute extreme multithreading arrays, Mac OS requires you manually install OpenMP:
```bash
brew install libomp
```

**3. Install Python Dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 💻 System Execution

You have two strict methods to trigger the Machine Learning loops:

#### 🏆 Method A: The Streamlit Web Engine (Recommended)
This is the ultimate, modernized UI deployment of the prediction stack. 
1. Boot the server continuously locally:
   ```bash
   streamlit run app.py
   ```
2. Navigate dynamically to `http://localhost:8501` securely in your browser.
3. Use the **🚀 Train Engine** tab to drag-and-drop a CSV, calculate predictions, and monitor algorithmic optimizations.
4. Shift exactly to the **🔮 Dynamic Testing** tab to reconstruct identical Neural matrices and predict exact custom equations smoothly!

#### ⚙️ Method B: Headless Terminal Scripts (Backend Dev Mode)
You can directly command the algorithm from the source script matrix:
1. **Initialize Complete Pipeline:** 
   ```bash
   python main.py
   ```
   *This automatically routes data, runs hyperparameter GridSearch, and prints the leaderboard accuracy of your current dataset into the Terminal seamlessly!*

---
> *More advanced architectures will be continually added to this root directory as I expand my AI & ML deployments permanently!*