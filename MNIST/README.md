# 🔢 MNIST Digit Recognition — Production Streamlit Application

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Machine Learning web application for handwritten digit recognition built with **Python**, **TensorFlow/Keras**, and **Streamlit**. 

The application enables real-time digit classification using Deep Convolutional Neural Networks (CNN), interactive canvas drawing, image uploads, non-blocking background model training with live thread progress tracking, and comprehensive model evaluation analytics.

---

## 🌟 Key Features

### 1. 🏠 Home Page & Dataset Statistics
- **Model Status Detection:** Automatically scans disk for existing pre-trained model weights (`models/best_model.h5`).
- **Dataset Insights:** Displays total samples (42,000), grayscale image dimensions ($28 \times 28 \times 1$), class balance check, and interactive digit frequency distribution bar charts.
- **One-Click Model Loading:** Loads pre-trained model checkpoints instantly into memory.

### 2. 🧠 Live Model Training Dashboard
- **Multi-Architecture Selection:** Choose between an **Advanced CNN** (~872k parameters with BatchNormalization and Data Augmentation) and a **Simple CNN** (~421k parameters).
- **Non-Blocking Background Execution:** Training runs in a dedicated Python background thread (`threading.Thread`), keeping the UI responsive.
- **Real-Time Progress Metrics:** Live epoch progress bar, loss/accuracy curves, and metric updates using custom Keras callbacks (`StreamlitProgressCallback`).
- **User-Controlled Interrupt Safety:** Includes a persistent **Stop Training** button (`StopTrainingCallback`) that halts training cleanly while preserving the single best checkpoint (`ModelCheckpoint(save_best_only=True)`).

### 3. 📊 Model Evaluation Dashboard
- **Cached Analytical Computations:** Uses `@st.cache_data` for model classification reports to maximize UI rendering speed.
- **Interactive Confusion Matrix:** Generates $10 \times 10$ Seaborn heatmaps with downloadable high-resolution PNG exports.
- **Automated Confusion Pair Analysis:** Automatically identifies and highlights top confused digit pairs (e.g., *Digit 4 misclassified as Digit 9*).
- **Per-Class Metrics Table:** Formatted Pandas DataFrames displaying precision, recall, F1-score, and support with conditional color highlighting.
- **Validation Sample Grid:** Renders 20 random validation predictions with color-coded titles (Green for correct, Red for misclassified).

### 4. 🔍 Interactive Digit Classifier
- **HTML5 Drawing Canvas:** Interactive drawing interface using `streamlit-drawable-canvas` ($280 \times 280$ canvas with thick stroke width).
- **Single & Batch Image Uploads:** Process user-uploaded images (`PNG`, `JPG`, `JPEG`) with metadata inspection.
- **MNIST-Standard Preprocessing:** Custom `center_and_pad_digit()` pipeline performing bounding-box cropping, aspect-ratio-preserving scaling ($20 \times 20$), and center padding ($28 \times 28$).
- **Rich Result Visualizations:** Displays large predicted digit hero cards, confidence percentages, top 3 prediction candidates, 10-class probability distribution bar charts, and CSV batch exports.

---

## 📐 Deep Learning Model Architectures

The project supports two Convolutional Neural Network (CNN) architectures implemented in Keras:

### Advanced CNN (Default & Recommended)
```
Input (28x28x1)
  │
  ├── Conv2D (32 filters, 3x3, ReLU, same padding)
  ├── BatchNormalization
  ├── Conv2D (32 filters, 3x3, ReLU, same padding)
  ├── MaxPooling2D (2x2)
  ├── Dropout (0.25)
  │
  ├── Conv2D (64 filters, 3x3, ReLU, same padding)
  ├── BatchNormalization
  ├── Conv2D (64 filters, 3x3, ReLU, same padding)
  ├── MaxPooling2D (2x2)
  ├── Dropout (0.25)
  │
  ├── Flatten
  ├── Dense (256 units, ReLU)
  ├── BatchNormalization
  ├── Dropout (0.40)
  └── Dense (10 units, Softmax)
```
- **Total Parameters:** ~872,042
- **Data Augmentation:** Random rotations ($\pm 10^\circ$), shifts ($\pm 10\%$), and zooms ($\pm 10\%$).
- **Performance:** $\sim 99.2\%+$ Validation Accuracy.

### Simple CNN (Lightweight)
```
Input (28x28x1) → Conv2D(32) → MaxPool(2x2) → Conv2D(64) → MaxPool(2x2) → Flatten → Dense(128) → Dropout(0.3) → Softmax(10)
```
- **Total Parameters:** ~421,642
- **Performance:** $\sim 98.5\%$ Validation Accuracy.

---

## ⚙️ Data & Image Preprocessing Pipeline

To achieve high prediction accuracy on custom user drawings and external image uploads, inputs undergo an automated 5-step preprocessing pipeline (`center_and_pad_digit` in `utils.py`):

```
Raw Input (Canvas / File) 
   │
   ▼
1. Color Mode & Channel Extraction (Grayscale RGB conversion)
   │
   ▼
2. Contrast Check & Inversion (Ensure white digit on black background)
   │
   ▼
3. Bounding-Box Cropping (Detect non-zero digit pixels)
   │
   ▼
4. Aspect-Ratio Preserving Resizing (Fit crop inside 20x20 box via Lanczos resampling)
   │
   ▼
5. Center Padding (Paste 20x20 digit in center of 28x28 black canvas -> Normalize [0.0, 1.0])
```

---

## 📂 Project Folder Structure

```text
MNIST/
├── app.py                     # Main Streamlit Web Application (UI, Routing, Session State)
├── utils.py                   # Utility Module (Data Loading, Models, Callbacks, Preprocessing, Visualizations)
├── requirements.txt           # Project Dependencies
├── models/
│   └── best_model.h5          # Trained Keras Model Checkpoint (Single Best Epoch)
├── data/
│   ├── train.csv              # Kaggle MNIST Training Data (42,000 samples)
│   └── test.csv               # Kaggle MNIST Test Data (28,000 samples)
├── outputs/
│   ├── plots/                 # Saved Evaluation Figures
│   └── reports/               # Training Logs & Evaluation CSV Summaries
└── notebooks/
    └── 01_eda.ipynb           # Exploratory Data Analysis & Prototype Training Notebook
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ or Python 3.12
- `pip` package manager

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/your-username/mnist-streamlit-app.git
cd mnist-streamlit-app
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .mnist_env
source .mnist_env/bin/activate

# Windows (Command Prompt)
python -m venv .mnist_env
.mnist_env\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit app locally:

```bash
streamlit run app.py
```

Once executed, open your web browser at:
`http://localhost:8501`

---

## 🧪 Verifying Module Dependencies & Functions

You can test all 13 core utility functions, model builders, and data loaders directly without starting the UI:

```bash
python utils.py
```

---

## 📜 Technology Stack

- **Frontend / Framework:** Streamlit (`1.40+`), `streamlit-drawable-canvas`
- **Machine Learning & Deep Learning:** TensorFlow (`2.16+`), Keras (`3.0+`)
- **Data Manipulation & Analytics:** NumPy, Pandas, Scikit-Learn
- **Computer Vision & Image Processing:** Pillow (PIL), OpenCV
- **Visualization:** Matplotlib, Seaborn

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
