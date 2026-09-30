import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

from tensorflow.keras.models import load_model

# ==============================================================================
# 1. Streamlit Page Configuration (MUST be the first Streamlit call)
# ==============================================================================
st.set_page_config(
    page_title="Traffic Signs Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. Load Custom CSS (if available)
# ==============================================================================
if os.path.exists('assets/style.css'):
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ==============================================================================
# 3. Class Names Dictionary (GTSRB 43 Classes)
# ==============================================================================
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road',
    24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End speed + passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# ==============================================================================
# 4. Cached Model Loading Function & Check
# ==============================================================================
@st.cache_resource
def load_traffic_model():
    """
    Loads the pre-trained CNN model from disk (.keras format).
    Cached so it only loads once per session, not on every user interaction.
    """
    model_paths = [
        'models/best_traffic_sign_model.keras',
        'models/final_traffic_sign_model.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = load_model(path)
                return model, path
            except Exception as e:
                st.warning(f"Failed to load model from {path}: {e}")
                
    return None, None

with st.spinner('🔄 Loading trained CNN model...'):
    model, loaded_model_path = load_traffic_model()

if model is None:
    st.error("""
    ❌ **Model Not Found!**
    
    Please ensure a valid trained model file exists in the `models/` directory:
    - `models/best_traffic_sign_model.keras`
    
    Run the Jupyter Notebook `Traffic_Signs.ipynb` first to train and save the model.
    """)
    st.stop()

# ==============================================================================
# 5. Page Component Function Stubs
# ==============================================================================
def show_overview():
    # 1. Header
    st.title("🚦 Traffic Signs Recognition")
    st.markdown("*CNN-based classifier trained on the GTSRB dataset to identify 43 categories of traffic signs.*")
    st.divider()

    # 2. Top KPI metrics row — 4 columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Total Classes", "43")
    with col2:
        total_params = model.count_params()
        st.metric("🧠 Model Parameters", f"{total_params:,}")
    with col3:
        st.metric("📐 Input Size", "32x32 px")
    with col4:
        st.metric("🔧 Framework", "TensorFlow/Keras")

    st.divider()

    # 3. Two column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("📋 About This Project")
        st.markdown("""
        This dashboard demonstrates a **Convolutional Neural Network (CNN)** 
        trained from scratch to recognize traffic signs from images.
        
        **Model Pipeline:**
        1. 🖼️ Image preprocessing (resize to 32x32, normalize pixels)
        2. 🧠 3-block CNN architecture (32→64→128 filters)
        3. 🔄 Data augmentation (rotation, zoom, shift)
        4. 🎯 Softmax classification across 43 sign categories
        
        **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
        — ~50,000 real-world traffic sign images across 43 classes.
        
        **Navigate using the sidebar to:**
        - 📤 Upload your own image and get instant predictions
        - 🖼️ Test the model on random samples from the dataset
        - 📊 Review detailed training and evaluation metrics
        """)

    with right_col:
        st.subheader("🏗️ Model Architecture")
        layer_info = []
        for layer in model.layers:
            try:
                out_shape = str(layer.output_shape)
            except Exception:
                out_shape = "Dynamic"
            layer_info.append({
                'Layer': layer.name,
                'Type': layer.__class__.__name__,
                'Output Shape': out_shape
            })
        arch_df = pd.DataFrame(layer_info)
        st.dataframe(arch_df, use_container_width=True, height=350)

    st.divider()

    # 4. Sample images section
    st.subheader("👀 Sample Traffic Signs from Training Data")
    train_dir = 'data/Train'
    if os.path.exists(train_dir):
        cols = st.columns(6)
        sample_classes = [0, 1, 14, 17, 25, 33]  # variety of signs
        
        for idx, class_id in enumerate(sample_classes):
            class_folder = os.path.join(train_dir, str(class_id))
            if os.path.exists(class_folder):
                valid_images = [f for f in os.listdir(class_folder) if not f.startswith('.')]
                if valid_images:
                    img_path = os.path.join(class_folder, valid_images[0])
                    img = Image.open(img_path)
                    with cols[idx]:
                        st.image(img, use_column_width=True)
                        st.caption(class_names[class_id])
    else:
        st.info("💡 Sample images unavailable — dataset folder not found. This doesn't affect predictions.")

    # 5. Footer callout
    st.divider()
    st.info("""
    💡 **Get Started:** Head to the **Upload & Predict** page in the sidebar to test the model on your own traffic sign image!
    """)

def show_upload_predict():
    # 1. Header
    st.title("📤 Upload & Predict")
    st.markdown("Upload a traffic sign image and get an instant prediction from the trained CNN model.")
    st.divider()

    # 2. Two column layout — ratio (5, 5)
    col1, col2 = st.columns(2)
    
    # LEFT COLUMN — Upload
    use_sample = None
    with col1:
        st.subheader("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image",
            type=['jpg', 'jpeg', 'png', 'ppm'],
            help="Supported formats: JPG, PNG, PPM"
        )
        
        st.divider()
        st.markdown("**🧪 Or try a sample image:**")
        
        sample_dir = 'data/Test'
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if not f.startswith('.')][:5]
            sample_choice = st.selectbox(
                "Pick a sample test image:",
                options=["-- None --"] + sample_files
            )
            if sample_choice != "-- None --":
                use_sample = os.path.join(sample_dir, sample_choice)
        else:
            st.caption("Sample images unavailable — Test folder not found")

    # RIGHT COLUMN — Prediction Result
    with col2:
        st.subheader("🎯 Prediction Result")
        
        image_source = None
        if uploaded_file is not None:
            image_source = Image.open(uploaded_file)
        elif use_sample is not None:
            image_source = Image.open(use_sample)
        
        if image_source is not None:
            # Display the uploaded/selected image
            st.image(image_source, caption="Input Image", width=250)
            
            with st.spinner("🔍 Analyzing..."):
                # Preprocess exactly like training data
                img_array = np.array(image_source.convert('RGB'))
                img_resized = cv2.resize(img_array, (32, 32))
                img_normalized = img_resized.astype('float32') / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                predictions = model.predict(img_batch, verbose=0)[0]
                pred_class = int(np.argmax(predictions))
                confidence = float(np.max(predictions) * 100)
                
                # Top 3 predictions
                top3_idx = np.argsort(predictions)[-3:][::-1]
                top3 = [(class_names[i], float(predictions[i] * 100)) for i in top3_idx]
            
            # Display main prediction
            if confidence >= 80:
                st.success(f"### ✅ {class_names[pred_class]}")
            elif confidence >= 50:
                st.warning(f"### ⚠️ {class_names[pred_class]}")
            else:
                st.error(f"### ❓ {class_names[pred_class]}")
            
            st.markdown(f"**Confidence: {confidence:.1f}%**")
            st.progress(min(1.0, max(0.0, confidence / 100.0)))
            
            st.divider()
            st.markdown("**📊 Top 3 Predictions:**")
            
            for i, (name, conf) in enumerate(top3):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.markdown(f"{emoji} **{name}** — {conf:.1f}%")
                st.progress(min(1.0, max(0.0, conf / 100.0)))
        
        else:
            st.info("👈 Upload an image or select a sample to see the prediction here.")

    # 3. Confidence guide section
    st.divider()
    
    with st.expander("ℹ️ How to read confidence scores"):
        st.markdown("""
        - **80%+ (Green)** — Model is highly confident in this prediction
        - **50-80% (Yellow)** — Moderate confidence, could be ambiguous
        - **Below 50% (Red)** — Low confidence, image may be unclear, cropped incorrectly, or an unfamiliar sign type
        
        For best results, upload a clear, well-lit, centered image of a single traffic sign.
        """)

def show_batch_testing():
    # 1. Header
    st.title("🖼️ Batch Testing")
    st.markdown("Run the model on multiple random test images at once and see how it performs.")
    st.divider()

    # 2. Check if test data is available
    test_dir = 'data/Test'
    test_csv = 'data/Test.csv'
    
    if not (os.path.exists(test_dir) and os.path.exists(test_csv)):
        st.warning("""
        ⚠️ Batch testing requires the GTSRB Test folder and Test.csv file. 
        Please ensure 'data/Test/' and 'data/Test.csv' exist.
        """)
        return

    # 3. Controls row
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_samples = st.slider(
            "Number of random images to test:",
            min_value=4, max_value=16, value=8, step=4
        )
    
    with col2:
        st.write("")
        st.write("")
        run_batch = st.button("🎲 Run Batch Test", type="primary", use_container_width=True)

    # 4. On button click, run batch prediction
    if run_batch:
        with st.spinner(f"Testing {num_samples} random images..."):
            test_df = pd.read_csv(test_csv)
            sample_rows = test_df.sample(n=num_samples, random_state=None)
            
            results = []
            for idx, row in sample_rows.iterrows():
                img_rel_path = str(row['Path']).replace('\\', '/')
                img_path = os.path.join('data', img_rel_path)
                true_class = int(row['ClassId'])
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (32, 32))
                img_norm = img_resized.astype('float32') / 255.0
                img_batch = np.expand_dims(img_norm, axis=0)
                
                pred_probs = model.predict(img_batch, verbose=0)[0]
                pred_class = int(np.argmax(pred_probs))
                confidence = float(np.max(pred_probs) * 100)
                
                results.append({
                    'image': img_rgb,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'correct': (true_class == pred_class)
                })
            
            st.session_state['batch_results'] = results

    # 5. Display results if available in session state
    if 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        
        correct_count = sum(r['correct'] for r in results)
        total_count = len(results)
        accuracy_pct = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        st.divider()
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Correct", f"{correct_count}/{total_count}")
        m2.metric("📊 Batch Accuracy", f"{accuracy_pct:.1f}%")
        m3.metric("🎲 Sample Size", total_count)
        
        st.divider()
        
        # Display images in a grid — 4 per row
        st.subheader("📷 Prediction Results")
        
        num_cols = 4
        rows_needed = (len(results) + num_cols - 1) // num_cols
        
        for row_idx in range(rows_needed):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                result_idx = row_idx * num_cols + col_idx
                if result_idx < len(results):
                    r = results[result_idx]
                    with cols[col_idx]:
                        st.image(r['image'], use_column_width=True)
                        
                        true_name = class_names[r['true_class']]
                        pred_name = class_names[r['pred_class']]
                        
                        if r['correct']:
                            st.success(f"✅ {pred_name}")
                        else:
                            st.error(f"❌ {pred_name}")
                            st.caption(f"Actual: {true_name}")
                        
                        st.caption(f"Confidence: {r['confidence']:.1f}%")
    else:
        st.info("👆 Click **Run Batch Test** above to see predictions on random test images.")

def show_model_performance():
    # 1. Header
    st.title("📊 Model Performance")
    st.markdown("Training results and evaluation metrics from the model training notebook.")
    st.divider()

    # 2. Check if outputs directory exists
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        st.warning("⚠️ Outputs folder not found. Run the training notebook first to generate evaluation charts.")
        return

    # 3. Helper function to safely display an image
    def show_chart(filename, caption):
        path = os.path.join(outputs_dir, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.info(f"Chart not found: {filename}")

    # 4. Tabbed layout for organized viewing
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Training History", 
        "🔲 Confusion Matrix",
        "📋 Classification Report",
        "❌ Error Analysis"
    ])
    
    with tab1:
        st.subheader("Training & Validation Curves")
        show_chart('04_training_curves.png', 'Accuracy and Loss over training epochs')
        
        st.markdown("""
        **How to read this:**
        - If training and validation lines stay close together, the model generalizes well (low overfitting)
        - A growing gap between them indicates overfitting
        """)
    
    with tab2:
        st.subheader("Confusion Matrix — Test Set")
        show_chart('05_confusion_matrix.png', 'Predicted vs Actual class for all 43 categories')
        
        st.markdown("""
        **How to read this:**
        - Diagonal cells = correct predictions
        - Off-diagonal cells = misclassifications between classes
        - Darker diagonal = stronger overall performance
        """)
    
    with tab3:
        st.subheader("Per-Class F1 Scores")
        show_chart('06_f1_per_class.png', 'F1-Score for each of the 43 traffic sign classes')
        
        st.markdown("""
        **Color coding:**
        - 🟢 Green: F1-Score > 0.95 (excellent)
        - 🟠 Orange: F1-Score 0.90-0.95 (good)
        - 🔴 Red: F1-Score < 0.90 (needs attention)
        """)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Misclassified Examples")
            show_chart('07_misclassified.png', 'Sample images the model got wrong')
        with col2:
            st.subheader("Live Prediction Samples")
            show_chart('08_live_predictions.png', 'Random test predictions with confidence')

    # 5. Below tabs — Dataset distribution reference
    st.divider()
    st.subheader("📊 Dataset Class Distribution")
    show_chart('01_class_distribution.png', 'Number of training images per class')
    
    st.caption("""
    💡 Classes with fewer training images may show lower accuracy — 
    this is a known limitation addressed via data augmentation during training.
    """)

    # 6. Final footer
    st.divider()
    st.info("""
    📌 **Note:** These charts are generated from the training notebook (`Traffic_Signs.ipynb`). 
    Re-run the notebook and refresh this page to see updated results after retraining.
    """)

# ==============================================================================
# 6. Sidebar Component & Controls
# ==============================================================================
with st.sidebar:
    st.markdown("# 🚦 Traffic Signs")
    st.markdown("*CNN Recognition System*")
    st.divider()
    
    page = st.radio(
        "Navigate",
        options=[
            "🏠 Overview",
            "📤 Upload & Predict",
            "🖼️ Batch Testing",
            "📊 Model Performance"
        ],
        label_visibility="collapsed"
    )
    st.divider()
    
    st.markdown("### ℹ️ Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classes", "43")
    with col2:
        st.metric("Input Size", "32x32")
    
    st.divider()
    st.caption("Built with TensorFlow + Streamlit")
    st.caption("Dataset: GTSRB")

# ==============================================================================
# 7. Main Content Page Routing
# ==============================================================================
if page == "🏠 Overview":
    show_overview()
elif page == "📤 Upload & Predict":
    show_upload_predict()
elif page == "🖼️ Batch Testing":
    show_batch_testing()
elif page == "📊 Model Performance":
    show_model_performance()
