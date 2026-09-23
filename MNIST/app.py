"""
=============================================================================
MNIST Digit Recognition - Main Streamlit Application (app.py)
=============================================================================
This is the primary entry point for the MNIST Digit Recognition Web App.
It provides a multi-page interactive UI for:
1. Home / Overview & Dataset Statistics
2. Model Training & Real-Time Thread Progress
3. Model Evaluation & Metrics Visualizations
4. Interactive Canvas & File Upload Digit Recognition
"""

import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import threading
import time
import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.metrics import classification_report, confusion_matrix
import utils  # Helper utility module containing all ML business logic



# =============================================================================
# PART A — GLOBAL SETUP & PAGE CONFIGURATION
# =============================================================================
# Must be the VERY FIRST Streamlit command called in the script
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PART D — CUSTOM STYLING (CSS)
# =============================================================================
st.markdown("""
<style>
    /* Main container fonts and background touches */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header title banner */
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: #f9fafb;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #9ca3af;
        font-size: 1.05rem;
    }

    /* Custom Glassmorphism Cards */
    .custom-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        height: 100%;
    }
    .custom-card h3 {
        color: #f8fafc;
        margin-top: 0;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .custom-card p {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Success and Warning Status Banner Boxes */
    .status-box-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #ecfdf5;
    }
    .status-box-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #fffbeb;
    }
    .status-box-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Sidebar Badge */
    .sidebar-badge-ready {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        border-radius: 20px;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin-top: 1rem;
    }
    .sidebar-badge-empty {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
        border-radius: 20px;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
def init_session_state():
    """Initializes Streamlit session state keys with robust default values."""
    defaults = {
        'current_page': 'home',
        'model': None,
        'model_loaded': False,
        'training_active': False,
        'stop_flag_ref': [False],
        'progress_dict': {},
        'training_thread': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# PART B — SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar():
    """Renders the styled sidebar navigation and live model status badge."""
    with st.sidebar:
        st.markdown("<h1 style='margin-bottom: 0px;'>🔢 MNIST Recognizer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-top: 0px;'>Handwritten Digit Recognition</p>", unsafe_allow_html=True)
        st.divider()

        st.markdown("### 📌 Navigation")
        
        # Define Pages
        nav_items = [
            ("🏠 Home", "home"),
            ("🧠 Train Model", "train"),
            ("📊 Evaluate Model", "evaluate"),
            ("🔍 Test Your Digit", "test")
        ]

        # Render full-width navigation buttons
        for label, page_key in nav_items:
            is_active = (st.session_state['current_page'] == page_key)
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                if st.session_state['current_page'] != page_key:
                    st.session_state['current_page'] = page_key
                    st.rerun()

        st.divider()

        # Model Status Badge at Sidebar Bottom
        st.markdown("### ⚡ Model Status")
        if st.session_state['model_loaded'] and st.session_state['model'] is not None:
            st.markdown("<div class='sidebar-badge-ready'>🟢 Model Ready</div>", unsafe_allow_html=True)
        else:
            model_info = utils.check_model_exists()
            if model_info['exists']:
                st.markdown("<div class='sidebar-badge-empty' style='background-color:#451a03; color:#fde68a; border-color:#d97706;'>🟡 Found on Disk (Unloaded)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='sidebar-badge-empty'>🔴 No Model Saved</div>", unsafe_allow_html=True)

        st.caption("v1.0.0 • TensorFlow & Keras")


# =============================================================================
# PART C — HOME PAGE (show_home_page)
# =============================================================================
def show_home_page():
    """Renders the Home Page dashboard with stats, model check, and dataset overview."""
    
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>Welcome to MNIST Digit Recognizer</h1>
        <p>Production-grade Deep Convolutional Neural Network (CNN) application for handwritten digit classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TOP SECTION — 3 COLUMNS
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    # Column 1: Project Title & Description
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🚀 Project Overview</h3>
            <p>
                This application uses state-of-the-art Deep Convolutional Neural Networks (CNN)
                built with <b>TensorFlow & Keras</b> to classify handwritten digits (0–9) in real-time.
            </p>
            <p>
                Features include interactive drawing canvas recognition, live background training,
                and comprehensive evaluation metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Column 2: Dataset Quick Stats
    stats = utils.get_data_statistics()
    with col2:
        total_s = stats.get('total_samples', 0)
        img_s = stats.get('image_shape', (28, 28, 1))
        is_bal = stats.get('is_balanced', False)
        
        st.markdown(f"""
        <div class="custom-card">
            <h3>📊 Dataset Quick Stats</h3>
            <p><b>Total Samples:</b> {total_s:,}</p>
            <p><b>Resolution:</b> {img_s[0]} × {img_s[1]} Grayscale</p>
            <p><b>Classes:</b> {utils.NUM_CLASSES} Digits (0 to 9)</p>
            <p><b>Distribution:</b> {"✅ Balanced" if is_bal else "ℹ️ Nearly Balanced"}</p>
        </div>
        """, unsafe_allow_html=True)

    # Column 3: Quick Start Guide
    with col3:
        st.markdown("""
        <div class="custom-card">
            <h3>💡 Quick Start Guide</h3>
            <p><b>1. Load/Train:</b> Load the existing pre-trained CNN or train a new one live.</p>
            <p><b>2. Evaluate:</b> Review confusion matrices, per-class F1-scores, and accuracy plots.</p>
            <p><b>3. Test:</b> Draw digits on an interactive canvas or upload your own images!</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODEL STATUS SECTION
    # -------------------------------------------------------------------------
    st.subheader("⚡ Model Availability & Actions")
    status_info = utils.check_model_exists()

    if status_info['exists']:
        # CASE A: Model Exists on Disk
        st.markdown(f"""
        <div class="status-box-success">
            <div class="status-box-title">✅ Trained Model Found on Disk!</div>
            <div><b>📁 File Path:</b> <code>{status_info['path']}</code></div>
            <div><b>📦 Model Size:</b> {status_info['size_mb']:.2f} MB</div>
            <div><b>🕒 Last Modified:</b> {status_info['modified_time']}</div>
        </div>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶ Use Existing Model", type="primary", use_container_width=True):
                with st.spinner("Loading pre-trained CNN model into memory..."):
                    model, err = utils.load_trained_model(status_info['path'])
                    if model is not None:
                        st.session_state['model'] = model
                        st.session_state['model_loaded'] = True
                        st.success("Model loaded successfully!")
                        time.sleep(0.5)
                        st.session_state['current_page'] = 'evaluate'
                        st.rerun()
                    else:
                        st.error(f"Failed to load model: {err}")

        with btn_col2:
            if st.button("🔄 Retrain Model", type="secondary", use_container_width=True):
                st.session_state['current_page'] = 'train'
                st.rerun()

    else:
        # CASE B: Model Does Not Exist
        st.markdown(f"""
        <div class="status-box-warning">
            <div class="status-box-title">⚠️ No Trained Model Found</div>
            <div>No model file exists at <code>{utils.MODEL_PATH}</code>.</div>
            <div>You need to train a model first before testing digit recognition.</div>
        </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            if st.button("🚀 Go Train the Model", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'train'
                st.rerun()

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION — DATASET OVERVIEW & EDA
    # -------------------------------------------------------------------------
    st.subheader("📈 Dataset Overview & Class Distribution")
    
    if stats.get('class_distribution'):
        dist_data = stats['class_distribution']
        df_dist = pd.DataFrame({
            'Digit Class': [f"Digit {k}" for k in dist_data.keys()],
            'Sample Count': list(dist_data.values())
        })
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{stats.get('total_samples', 0):,}</div>
                <div class="metric-label">Total Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{utils.IMG_SIZE} × {utils.IMG_SIZE}</div>
                <div class="metric-label">Image Resolution (Pixels)</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{utils.NUM_CLASSES}</div>
                <div class="metric-label">Output Classes (0–9)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.bar_chart(df_dist.set_index('Digit Class'))
    else:
        st.info("Dataset statistics are unavailable. Ensure `data/train.csv` exists.")

    with st.expander("ℹ️ About the MNIST Dataset"):
        st.write("""
        The **MNIST dataset** (Modified National Institute of Standards and Technology database) is a benchmark
        dataset in computer vision and machine learning. It consists of 70,000 28×28 pixel grayscale images
        of handwritten digits (60,000 training images and 10,000 test images), collected from high school students
        and Census Bureau employees.
        """)


# =============================================================================
# PLACEHOLDER PAGE FUNCTIONS FOR FUTURE PHASES
# =============================================================================
def show_train_page():
    """
    Renders the Model Training Dashboard with real-time background thread training,
    live metric progress tracking, interactive loss/accuracy curves, stop button,
    and post-training status reports.
    """
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Model Training Dashboard</h1>
        <p>Configure, launch, and monitor deep CNN training with real-time feedback and Early Stopping.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART A — PRE-TRAINING CHECK & REFRESH HANDLING
    # -------------------------------------------------------------------------
    is_active = st.session_state.get('training_active', False)

    # -------------------------------------------------------------------------
    # PART B — CONFIGURATION UI (When not actively training)
    # -------------------------------------------------------------------------
    if not is_active:
        # Check if a model file already exists on disk
        model_info = utils.check_model_exists()
        if model_info['exists']:
            st.markdown(f"""
            <div class="status-box-warning">
                <div class="status-box-title">⚠️ Existing Model Detected</div>
                <div>An existing trained model was found at <code>{model_info['path']}</code> ({model_info['size_mb']:.2f} MB, modified {model_info['modified_time']}).</div>
                <div>Starting training will create a new checkpoint and save the best weights to <code>{utils.MODEL_PATH}</code>.</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("⚙️ Training Configuration")

        c1, c2 = st.columns([2, 1])
        with c1:
            model_type = st.selectbox(
                "Select CNN Architecture",
                options=["advanced", "simple"],
                index=0,
                format_func=lambda x: "Advanced CNN (Deep Architecture with BatchNormalization & Data Augmentation)" if x == "advanced" else "Simple CNN (Lightweight 2-Layer Architecture)"
            )

        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("ℹ️ Hyperparameters: Batch Size = 64, Max Epochs = 50, Optimizer = Adam (lr=0.001)")

        # Architecture Details Card
        if model_type == "advanced":
            st.markdown("""
            <div class="custom-card">
                <h3>🚀 Advanced CNN Architecture</h3>
                <p><b>Layers:</b> Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25) → Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25) → Dense(256) → BatchNorm → Dropout(0.4) → Softmax(10)</p>
                <p><b>Parameters:</b> ~872,042 trainable weights</p>
                <p><b>Features:</b> Data Augmentation (Rotation, Zoom, Shifts), Batch Normalization for faster convergence & stability.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-card">
                <h3>⚡ Simple CNN Architecture</h3>
                <p><b>Layers:</b> Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Softmax(10)</p>
                <p><b>Parameters:</b> ~421,642 trainable weights</p>
                <p><b>Features:</b> Lightweight structure, fast training speed, suitable for low-resource environments.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Start Training Button
        col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
        with col_start2:
            if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
                # 1. Reset progress dict
                st.session_state['progress_dict'] = {
                    'current_epoch': 0,
                    'total_epochs': 50,
                    'train_accuracy': 0.0,
                    'val_accuracy': 0.0,
                    'train_loss': 0.0,
                    'val_loss': 0.0,
                    'history_acc': [],
                    'history_val_acc': [],
                    'history_loss': [],
                    'history_val_loss': [],
                    'status': 'Loading data & initializing model...',
                    'completed': False,
                    'stopped': False,
                    'error': None
                }
                # 2. Reset stop flag
                st.session_state['stop_flag_ref'] = [False]
                # 3. Set active training state
                st.session_state['training_active'] = True
                
                # 4. Launch daemon thread
                train_thread = threading.Thread(
                    target=utils.train_model,
                    args=(model_type, st.session_state['progress_dict'], st.session_state['stop_flag_ref']),
                    daemon=True
                )
                train_thread.start()
                st.session_state['training_thread'] = train_thread
                
                # 5. Rerun to enter live progress view immediately
                st.rerun()

        # Check if previous training run just completed or stopped (when not active)
        p_dict = st.session_state.get('progress_dict', {})
        if p_dict.get('completed'):
            val_acc = p_dict.get('val_accuracy', 0.0)
            st.markdown(f"""
            <div class="status-box-success">
                <div class="status-box-title">🎉 Model Training Completed Successfully!</div>
                <div>Best model weights restored and saved to <code>{utils.MODEL_PATH}</code>.</div>
                <div><b>Final Validation Accuracy:</b> {val_acc:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-load the newly trained model into session state
            if not st.session_state.get('model_loaded'):
                loaded_model, err = utils.load_trained_model()
                if loaded_model is not None:
                    st.session_state['model'] = loaded_model
                    st.session_state['model_loaded'] = True
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button("📊 Go to Model Evaluation", type="primary", use_container_width=True):
                    st.session_state['current_page'] = 'evaluate'
                    st.rerun()
            with b2:
                if st.button("🔍 Test Your Digit", type="secondary", use_container_width=True):
                    st.session_state['current_page'] = 'test'
                    st.rerun()

        elif p_dict.get('stopped'):
            curr_ep = p_dict.get('current_epoch', 0)
            st.markdown(f"""
            <div class="status-box-warning">
                <div class="status-box-title">⏹ Training Stopped by User</div>
                <div>Best model up to epoch {curr_ep} has been preserved at <code>{utils.MODEL_PATH}</code>.</div>
                <div>Your previous model (if any) remains unchanged.</div>
            </div>
            """, unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("📊 Go to Model Evaluation", type="primary", use_container_width=True):
                    st.session_state['current_page'] = 'evaluate'
                    st.rerun()
            with b2:
                if st.button("🔍 Test Your Digit", type="secondary", use_container_width=True):
                    st.session_state['current_page'] = 'test'
                    st.rerun()

        elif p_dict.get('error'):
            st.error(f"❌ Training Error Occurred: {p_dict['error']}")
            if st.button("🔄 Try Again", type="secondary"):
                st.session_state['progress_dict'] = {}
                st.rerun()

    # -------------------------------------------------------------------------
    # PART C — LIVE PROGRESS VIEW (When training_active == True)
    # -------------------------------------------------------------------------
    else:
        st.subheader("⚡ Live Model Training Progress")
        
        p_dict = st.session_state.get('progress_dict', {})
        curr_ep = p_dict.get('current_epoch', 0)
        tot_ep = p_dict.get('total_epochs', 50)
        status_msg = p_dict.get('status', 'Training in progress...')

        # Stop Button Header Bar — Visible at top at all times
        stop_col1, stop_col2 = st.columns([3, 1])
        with stop_col1:
            st.markdown(f"#### Status: `{status_msg}`")
        with stop_col2:
            if st.button("⏹ Stop Training", type="primary", use_container_width=True):
                st.session_state['stop_flag_ref'][0] = True
                p_dict['status'] = "Stopping training after current epoch..."
                st.warning("Halting signal sent! Finishing current epoch...")

        # Progress Bar
        progress_val = min(1.0, max(0.0, float(curr_ep) / float(tot_ep))) if tot_ep > 0 else 0.0
        st.progress(progress_val)

        # 4 Metric Boxes
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{curr_ep} / {tot_ep}</div>
                <div class="metric-label">Epoch</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{p_dict.get('train_accuracy', 0.0):.2f}%</div>
                <div class="metric-label">Train Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{p_dict.get('val_accuracy', 0.0):.2f}%</div>
                <div class="metric-label">Val Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{p_dict.get('val_loss', 0.0):.4f}</div>
                <div class="metric-label">Val Loss</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Live Accuracy & Loss Plots
        if p_dict.get('history_acc'):
            fig_hist = utils.plot_training_history(p_dict)
            st.pyplot(fig_hist, use_container_width=True)

        # Auto-Refresh Logic (time.sleep(1) + st.rerun())
        thread = st.session_state.get('training_thread')
        is_completed = p_dict.get('completed', False)
        is_stopped = p_dict.get('stopped', False)
        has_error = bool(p_dict.get('error'))
        
        thread_done = (thread is not None and not thread.is_alive())

        if is_completed or is_stopped or has_error or (thread_done and curr_ep > 0):
            st.session_state['training_active'] = False
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()

    # -------------------------------------------------------------------------
    # PART E — TRAINING LOG EXPANDER (Visible during & after training)
    # -------------------------------------------------------------------------
    p_dict = st.session_state.get('progress_dict', {})
    if p_dict and p_dict.get('history_acc'):
        with st.expander("📋 Detailed Epoch Training Log", expanded=False):
            acc_list = p_dict.get('history_acc', [])
            val_acc_list = p_dict.get('history_val_acc', [])
            loss_list = p_dict.get('history_loss', [])
            val_loss_list = p_dict.get('history_val_loss', [])
            
            log_df = pd.DataFrame({
                'Epoch': list(range(1, len(acc_list) + 1)),
                'Train Accuracy (%)': acc_list,
                'Val Accuracy (%)': val_acc_list,
                'Train Loss': loss_list,
                'Val Loss': val_loss_list
            })
            st.dataframe(log_df, use_container_width=True)


# =============================================================================
# PART B — OVERVIEW METRICS CACHING
# =============================================================================
@st.cache_data
def get_cached_report_df(_model):
    """
    Computes and caches classification report DataFrame for the loaded Keras model.
    The underscore '_model' parameter prevents Streamlit from attempting to hash the Keras object.
    """
    return utils.get_classification_report_df(_model)


def show_evaluate_page():
    """
    Renders the Model Evaluation Dashboard containing accuracy metrics, 
    confusion matrix analysis, training curves, classification report, 
    sample predictions, and model architecture details.
    """
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Evaluation Dashboard</h1>
        <p>In-depth performance analytics, confusion matrix, and validation sample predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART A — MODEL GUARD CHECK
    # -------------------------------------------------------------------------
    if not st.session_state.get('model_loaded') or st.session_state.get('model') is None:
        st.markdown("""
        <div class="status-box-warning" style="text-align: center; padding: 2rem;">
            <div class="status-box-title" style="font-size: 1.5rem;">🔒 No Model Loaded</div>
            <p style="font-size: 1.05rem;">Please go to Home and either load an existing trained model or train a new one first.</p>
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            if st.button("🏠 Go to Home", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'home'
                st.rerun()
        return

    # If model is loaded: Green Banner
    model = st.session_state['model']
    model_info = utils.check_model_exists()
    model_path = model_info.get('path', utils.MODEL_PATH)
    
    st.markdown(f"""
    <div class="status-box-success">
        <div class="status-box-title">✅ Evaluating Loaded Model: <code>{model_path}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART B — OVERVIEW METRICS ROW
    # -------------------------------------------------------------------------
    report_df = get_cached_report_df(model)
    
    # Extract Macro & Overall Metrics
    overall_acc = 0.0
    macro_f1 = 0.0
    macro_prec = 0.0
    macro_rec = 0.0

    if not report_df.empty and 'class' in report_df.columns:
        acc_row = report_df[report_df['class'] == 'accuracy']
        macro_row = report_df[report_df['class'] == 'macro avg']
        
        if not acc_row.empty:
            overall_acc = float(acc_row['f1-score'].values[0]) * 100.0
        if not macro_row.empty:
            macro_f1 = float(macro_row['f1-score'].values[0]) * 100.0
            macro_prec = float(macro_row['precision'].values[0]) * 100.0
            macro_rec = float(macro_row['recall'].values[0]) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{overall_acc:.2f}%</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{macro_f1:.2f}%</div>
            <div class="metric-label">Macro F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{macro_prec:.2f}%</div>
            <div class="metric-label">Macro Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{macro_rec:.2f}%</div>
            <div class="metric-label">Macro Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART C — TABS LAYOUT
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Confusion Matrix",
        "📈 Training Curves",
        "📋 Classification Report",
        "🖼 Sample Predictions"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: CONFUSION MATRIX
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Validation Confusion Matrix")
        fig_cm = utils.plot_confusion_matrix(model)
        st.pyplot(fig_cm)

        # Download button for Confusion Matrix Image
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format="png", bbox_inches="tight", dpi=150)
        st.download_button(
            label="⬇ Download Confusion Matrix Image",
            data=buf_cm.getvalue(),
            file_name="confusion_matrix.png",
            mime="image/png",
            type="secondary"
        )
        import matplotlib.pyplot as plt
        plt.close(fig_cm)

        st.markdown("#### 🔍 Top Confused Digit Pairs")
        try:
            _, X_val, _, y_val, err = utils.load_and_preprocess_data()
            if X_val is not None:
                y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
                val_probs = model.predict(X_val, verbose=0)
                y_pred = np.argmax(val_probs, axis=1)
                
                cm = confusion_matrix(y_true, y_pred)
                cm_off = cm.copy()
                np.fill_diagonal(cm_off, 0)
                
                top3_flat = np.argsort(cm_off.ravel())[::-1][:3]
                found_pairs = False
                for idx in top3_flat:
                    r, c = np.unravel_index(idx, cm_off.shape)
                    cnt = cm_off[r, c]
                    if cnt > 0:
                        found_pairs = True
                        st.info(f"ℹ️ **Digit {r}** was confused with **Digit {c}** — **{cnt}** times")
                if not found_pairs:
                    st.success("🎉 Perfect validation classification! No digit confusion detected.")
        except Exception as e:
            st.caption(f"Could not compute top confused pairs: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 2: TRAINING CURVES
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Training & Validation Curves")
        p_dict = st.session_state.get('progress_dict', {})
        
        if p_dict and p_dict.get('history_acc'):
            fig_hist = utils.plot_training_history(p_dict)
            st.pyplot(fig_hist)
            import matplotlib.pyplot as plt
            plt.close(fig_hist)
        else:
            # Check for saved training log CSV files as a fallback
            log_candidates = [
                "outputs/reports/advanced_training_log.csv",
                "outputs/reports/baseline_training_log.csv"
            ]
            loaded_log = None
            for cand in log_candidates:
                resolved_cand = utils.get_resolved_path(cand)
                if os.path.exists(resolved_cand):
                    loaded_log = resolved_cand
                    break
                    
            if loaded_log and os.path.exists(loaded_log):
                try:
                    df_log = pd.read_csv(loaded_log)
                    acc_vals = [float(v * 100.0 if v <= 1.0 else v) for v in df_log['accuracy']]
                    val_acc_vals = [float(v * 100.0 if v <= 1.0 else v) for v in df_log['val_accuracy']]
                    loss_vals = [float(v) for v in df_log['loss']]
                    val_loss_vals = [float(v) for v in df_log['val_loss']]
                    
                    fallback_pdict = {
                        'history_acc': acc_vals,
                        'history_val_acc': val_acc_vals,
                        'history_loss': loss_vals,
                        'history_val_loss': val_loss_vals
                    }
                    fig_hist = utils.plot_training_history(fallback_pdict)
                    st.pyplot(fig_hist)
                    import matplotlib.pyplot as plt
                    plt.close(fig_hist)
                    st.caption(f"ℹ️ Loaded pre-trained model history from `{os.path.basename(loaded_log)}`")
                except Exception as e:
                    st.info("ℹ️ Training curves are available when you train the model live in this session.")
            else:
                st.info("ℹ️ Training curves are only available when you train the model in this active session.")


    # -------------------------------------------------------------------------
    # TAB 3: CLASSIFICATION REPORT
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Detailed Per-Class Performance")
        if not report_df.empty:
            def style_f1(val):
                try:
                    vf = float(val)
                    if vf < 0.97:
                        return 'background-color: #451a03; color: #fde68a; font-weight: bold;'
                    elif vf >= 0.99:
                        return 'background-color: #064e3b; color: #a7f3d0; font-weight: bold;'
                except:
                    pass
                return ''

            try:
                map_func = getattr(report_df.style, 'map', getattr(report_df.style, 'applymap', None))
                if map_func:
                    styled_df = map_func(style_f1, subset=['f1-score'])
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(report_df, use_container_width=True)
            except Exception:
                st.dataframe(report_df, use_container_width=True)
                
            st.caption("Support = number of samples per digit class in validation dataset")

    # -------------------------------------------------------------------------
    # TAB 4: SAMPLE PREDICTIONS
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Sample Predictions on Validation Set (20 Samples)")
        fig_samples = utils.plot_sample_predictions(model, n_samples=20)
        st.pyplot(fig_samples)
        import matplotlib.pyplot as plt
        plt.close(fig_samples)

    # -------------------------------------------------------------------------
    # PART D — MODEL ARCHITECTURE EXPANDER
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔧 Model Architecture Details", expanded=False):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.code('\n'.join(stringlist), language="text")

        e1, e2, e3 = st.columns(3)
        total_p = model.count_params()
        try:
            import tensorflow as tf
            trainable_p = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        except Exception:
            trainable_p = total_p

        with e1:
            st.metric("Total Parameters", f"{total_p:,}")
        with e2:
            st.metric("Trainable Parameters", f"{trainable_p:,}")
        with e3:
            st.metric("Disk Size", f"{model_info.get('size_mb', 0.0):.2f} MB")


def show_prediction_results(prediction_dict: dict):
    """
    Renders prediction card, confidence bar, top 3 predictions, 
    10-digit probability distribution chart, and interpretation advice.
    """
    if not prediction_dict or 'predicted_digit' not in prediction_dict:
        return

    digit = prediction_dict['predicted_digit']
    conf = prediction_dict.get('confidence', 0.0)

    # Color Coded Badge & Box based on confidence
    if conf >= 90.0:
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "#10b981"
        digit_color = "#34d399"
    elif conf >= 70.0:
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "#f59e0b"
        digit_color = "#fbbf24"
    else:
        badge_bg = "rgba(239, 68, 68, 0.15)"
        badge_border = "#ef4444"
        digit_color = "#f87171"

    st.markdown(f"""
    <div style="background:{badge_bg}; border:2px solid {badge_border}; border-radius:14px; padding:1.5rem; text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:0.95rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Predicted Digit</div>
        <div style="font-size:4.5rem; font-weight:800; color:{digit_color}; margin:0.2rem 0;">{digit}</div>
        <div style="font-size:1.1rem; font-weight:600; color:#f8fafc;">Confidence: {conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(1.0, max(0.0, conf / 100.0)))

    # Top 3 Predictions
    st.markdown("#### 🏆 Top 3 Candidates")
    top3 = prediction_dict.get('top3', [])
    for rank, (cand_digit, cand_conf) in enumerate(top3, 1):
        c_d, c_b, c_p = st.columns([1.5, 3.5, 1.5])
        c_d.markdown(f"**Digit {cand_digit}**")
        c_b.progress(min(1.0, max(0.0, cand_conf / 100.0)))
        c_p.markdown(f"**{cand_conf:.1f}%**")

    st.divider()

    # Full Probability Distribution Chart
    st.markdown("#### 📊 Probability Distribution (Digits 0–9)")
    all_probs = prediction_dict.get('all_probabilities', [0.0] * 10)
    df_probs = pd.DataFrame({
        'Probability (%)': all_probs
    }, index=[f"Digit {i}" for i in range(10)])
    st.bar_chart(df_probs)

    # Confidence Interpretation Alert
    if conf >= 95.0:
        st.success("✅ **Very High Confidence** — The model is extremely certain of this classification.")
    elif conf >= 80.0:
        st.info("ℹ️ **Good Confidence** — High probability match, likely correct.")
    elif conf >= 60.0:
        st.warning("⚠️ **Moderate Confidence** — The stroke pattern may be slightly ambiguous.")
    else:
        st.error("❌ **Low Confidence** — Ambiguous digit shape or noisy input.")


def show_test_page():
    """
    Renders the Interactive Digit Classification Page featuring live canvas drawing,
    file upload preprocessing, batch testing, and tips.
    """
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Test Your Digit Classifier</h1>
        <p>Draw a digit on the interactive canvas or upload an image to test real-time CNN predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART A — MODEL GUARD CHECK
    # -------------------------------------------------------------------------
    if not st.session_state.get('model_loaded') or st.session_state.get('model') is None:
        st.markdown("""
        <div class="status-box-warning" style="text-align: center; padding: 2rem;">
            <div class="status-box-title" style="font-size: 1.5rem;">🔒 No Model Loaded</div>
            <p style="font-size: 1.05rem;">Please go to Home and either load an existing trained model or train a new one first.</p>
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            if st.button("🏠 Go to Home", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'home'
                st.rerun()
        return

    model = st.session_state['model']
    
    st.markdown("""
    <div class="status-box-success">
        <div class="status-box-title">✅ Model Ready — Draw on the canvas or upload a digit image to predict</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize canvas key if not set
    if 'canvas_key' not in st.session_state:
        st.session_state['canvas_key'] = "canvas_0"

    # -------------------------------------------------------------------------
    # PART B — TWO INPUT TABS
    # -------------------------------------------------------------------------
    tab_upload, tab_canvas = st.tabs(["📁 Upload Image", "✏️ Draw Digit"])

    # -------------------------------------------------------------------------
    # TAB 1: UPLOAD IMAGE
    # -------------------------------------------------------------------------
    with tab_upload:
        u_col1, u_col2 = st.columns([3, 2])
        
        with u_col1:
            st.subheader("Upload Digit Image")
            uploaded_file = st.file_uploader(
                "Choose a handwritten digit image (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of a single handwritten digit (0-9)"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    
                    st.markdown("#### Original Uploaded Image")
                    i_col1, i_col2 = st.columns(2)
                    with i_col1:
                        st.image(image, caption=f"Uploaded File ({uploaded_file.name})", width=200)
                    with i_col2:
                        st.caption(f"**Dimensions:** {image.width} × {image.height} px")
                        st.caption(f"**Color Mode:** {image.mode}")
                    
                    # Preprocess image
                    prep_tensor = utils.preprocess_uploaded_image(image)
                    prep_img_2d = prep_tensor.reshape(utils.IMG_SIZE, utils.IMG_SIZE)
                    
                    st.markdown("#### Model Input (28×28 Grayscale)")
                    st.image(prep_img_2d, caption="What the CNN model sees (28×28)", width=150)

                    # Predict button
                    if st.button("🔍 Predict Uploaded Digit", type="primary", key="btn_predict_upload", use_container_width=True):
                        with st.spinner("Classifying image..."):
                            pred_res = utils.predict_digit(model, prep_tensor)
                            st.session_state['last_pred_upload'] = pred_res
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Could not process uploaded image: {str(e)}")

        with u_col2:
            st.subheader("Prediction Results")
            if 'last_pred_upload' in st.session_state and st.session_state['last_pred_upload']:
                show_prediction_results(st.session_state['last_pred_upload'])
            else:
                st.info("👈 Upload an image on the left and click **Predict Uploaded Digit** to view results here.")

    # -------------------------------------------------------------------------
    # TAB 2: DRAW DIGIT
    # -------------------------------------------------------------------------
    with tab_canvas:
        c_col1, c_col2 = st.columns([3, 2])

        with c_col1:
            st.subheader("Interactive Canvas")
            st.caption("Draw a single digit (0–9) centered on the canvas using thick strokes.")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=st.session_state['canvas_key'],
                display_toolbar=True,
            )

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("🔍 Predict Drawn Digit", type="primary", use_container_width=True):
                    if canvas_result.image_data is None or np.max(canvas_result.image_data[:, :, 3]) == 0:
                        st.warning("⚠️ Please draw a digit on the canvas first!")
                    else:
                        with st.spinner("Classifying drawn stroke..."):
                            prep_tensor = utils.preprocess_canvas_image(canvas_result.image_data)
                            pred_res = utils.predict_digit(model, prep_tensor)
                            st.session_state['last_pred_canvas'] = pred_res
                            st.rerun()

            with btn_c2:
                if st.button("🗑 Clear Canvas", type="secondary", use_container_width=True):
                    st.session_state['canvas_key'] = f"canvas_{time.time()}"
                    if 'last_pred_canvas' in st.session_state:
                        del st.session_state['last_pred_canvas']
                    st.rerun()

        with c_col2:
            st.subheader("Prediction Results")
            if 'last_pred_canvas' in st.session_state and st.session_state['last_pred_canvas']:
                show_prediction_results(st.session_state['last_pred_canvas'])
            else:
                st.info("👈 Draw a digit on the canvas and click **Predict Drawn Digit** to view results here.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART D — TIPS SECTION
    # -------------------------------------------------------------------------
    with st.expander("💡 Tips for Best Classification Accuracy", expanded=False):
        t1, t2 = st.columns([2, 1])
        with t1:
            st.markdown("""
            - **Draw Large & Centered:** Fill roughly 70-80% of the canvas height.
            - **Use Thick Strokes:** Keep stroke width around 15-20px for clear feature extraction.
            - **Avoid Edges:** Keep drawn strokes away from the canvas boundaries.
            - **Image Uploads:** Works best with high-contrast images (either white digit on black background or black digit on white background).
            - **MNIST Style:** Standard simple handwriting without unusual cursive or decorative flourishes works best.
            """)
        with t2:
            st.markdown("##### MNIST Reference Grid")
            try:
                fig_samples = utils.plot_sample_predictions(model, n_samples=6)
                st.pyplot(fig_samples)
                import matplotlib.pyplot as plt
                plt.close(fig_samples)
            except Exception:
                st.caption("Reference grid unavailable.")

    # -------------------------------------------------------------------------
    # PART E — BATCH TESTING (BONUS FEATURE)
    # -------------------------------------------------------------------------
    with st.expander("🗂 Batch Test Multiple Digit Images", expanded=False):
        st.write("Upload a collection of handwritten digit images to run batch predictions and download CSV summary.")
        batch_files = st.file_uploader(
            "Upload multiple digit files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if batch_files:
            batch_data = []
            high_conf_cnt = 0
            
            with st.spinner(f"Running predictions on {len(batch_files)} images..."):
                for b_file in batch_files:
                    try:
                        b_img = Image.open(b_file)
                        prep_t = utils.preprocess_uploaded_image(b_img)
                        res = utils.predict_digit(model, prep_t)
                        conf = res['confidence']
                        if conf >= 90.0:
                            high_conf_cnt += 1
                        
                        batch_data.append({
                            'Filename': b_file.name,
                            'Predicted Digit': res['predicted_digit'],
                            'Confidence (%)': f"{conf:.2f}%",
                            'Top 2 Candidate': f"Digit {res['top3'][1][0]} ({res['top3'][1][1]:.1f}%)" if len(res['top3']) > 1 else "N/A"
                        })
                    except Exception as e:
                        batch_data.append({
                            'Filename': b_file.name,
                            'Predicted Digit': "Error",
                            'Confidence (%)': "0.0%",
                            'Top 2 Candidate': str(e)
                        })

            df_batch = pd.DataFrame(batch_data)
            st.dataframe(df_batch, use_container_width=True)
            
            st.success(f"📊 **{high_conf_cnt} out of {len(batch_files)}** predictions achieved ≥90% confidence!")
            
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Batch Results CSV",
                data=csv_data,
                file_name="batch_digit_predictions.csv",
                mime="text/csv",
                type="secondary"
            )



# =============================================================================
# PAGE ROUTER
# =============================================================================
def main():
    """Main routing function executing the selected page component."""
    render_sidebar()
    
    page_map = {
        'home': show_home_page,
        'train': show_train_page,
        'evaluate': show_evaluate_page,
        'test': show_test_page,
    }

    current_page = st.session_state.get('current_page', 'home')
    if current_page in page_map:
        page_map[current_page]()
    else:
        show_home_page()

if __name__ == '__main__':
    main()
