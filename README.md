# 🧠 AI & Machine Learning Projects Portfolio

Welcome to my central repository for all Artificial Intelligence and Machine Learning explorations! This repository will serve as a growing portfolio of end-to-end ML projects—from data collection and preprocessing to model training, evaluation, and user interface deployment.

---

## 📂 Projects Overview

Below is an index of all the projects currently contained in this repository:

### 1. [House Price Prediction](./House%20Price%20Prediction/)
A complete, robust machine learning pipeline built to predict continuous targets (like real estate prices). It features a full Python scikit-learn backend alongside a generic, interactive **AutoML Streamlit Web Application** that can dynamically train and test on any uploaded CSV dataset seamlessly!

---

## 🚀 Deep Dive: Project 1 - House Price Prediction / AutoML Platform

While initially designed specifically to predict house prices using the Kaggle housing dataset, this project evolved into a generic Regression **AutoML UI**. The Streamlit web application allows users to upload *any* CSV dataset, select a target column, train a customized Linear Regression model dynamically, and test the model immediately with mathematically generated UI feature inputs.

### 🏗️ Directory Structure
```text
House Price Prediction/
│
├── data/                  # Houses all datasets
│   ├── raw/               # Unmodified datasets directly from sources (e.g. train.csv)
│   └── processed/         # Data manipulated and cleaned by python scripts
│
├── models/                # Holds trained serialized parameters (.pkl)
│   ├── linear_regression_model.pkl
│   ├── scaler.pkl
│   └── universal_* artifacts for the Streamlit UI
│
├── notebooks/             # Scratchpad Jupyter Notebooks for EDA
│
├── src/                   # The Core Machine Learning Scripts
│   ├── data_loader.py     # Pulls and validates the dataset
│   ├── preprocessing.py   # Cleans missing values, drops unneeded columns, splits subsets
│   ├── train_model.py     # Automates model initialization and fit
│   ├── evaluate.py        # Generates metrics (MAE, RMSE, R2) and Matplotlib Visualizations
│   └── predict.py         # Performs live terminal-based prediction on 5 random rows
│
├── visualizations/        # Output directory for PNG regression charts from evaluate.py
├── app.py                 # 🌟 Streamlit AutoML Web Application
├── requirements.txt       # Project python dependencies
└── README.md              # Project level documentation
```

### 🛠️ Environment Setup

To run this project locally, execute the following commands in your terminal from the `House Price Prediction/` directory:

**1. Create & Activate a Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install Required Packages:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 💻 How to Run the Code

You have two options for running the project: via **Terminal Scripts** or via the **Web UI**.

#### Option A: Terminal Scripts (Developer Mode)
Ensure you have dropped a raw dataset (like `train.csv` from Kaggle) into the `data/raw/` directory first!
1. **Train Model:** `python src/train_model.py` (Trains the scikit-learn Linear Regression model).
2. **Evaluate Model:** `python src/evaluate.py` (Outputs model accuracy, MAE, RMSE, and draws scatter & residual visual plots).
3. **Live Terminal Test:** `python src/predict.py` (Picks 5 houses dynamically from your dataset and visually predicts their prices vs reality in text).

#### Option B: Streamlit Web UI (End-User Mode)
The easiest and most interactive way to experience the project.
1. Start the server:
   ```bash
   streamlit run app.py
   ```
2. Navigate to `http://localhost:8501` in your browser.
3. Use the **Train Model** sidebar to drop a CSV file and construct a custom model.
4. Use the **Test Model** sidebar to dynamically input numbers and get live price inferences!

---
> *More projects will be continually added to this root directory as I expand my AI & ML portfolio!*