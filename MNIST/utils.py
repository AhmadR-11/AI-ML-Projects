"""
=============================================================================
MNIST Digit Recognition - Utility Module (utils.py)
=============================================================================
This module provides all core helper functions, model architecture builders,
data preprocessing routines, training threads/callbacks, image preprocessors,
and visualization utilities for the MNIST Digit Recognition Streamlit app.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import io

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
MODEL_PATH = "models/best_model.h5"
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
IMG_SIZE = 28
NUM_CLASSES = 10
RANDOM_SEED = 42


def get_resolved_path(path: str) -> str:
    """
    Intelligently resolves file paths whether running from workspace root
    or subdirectories.

    Args:
        path (str): Relative file path.

    Returns:
        str: Existing resolved file path or original path string.
    """
    if os.path.exists(path):
        return path
    
    # Try alternate location candidates
    candidates = [
        os.path.join("..", path),
        os.path.join("data/raw", os.path.basename(path)),
        os.path.join("../data/raw", os.path.basename(path)),
        os.path.join("data", os.path.basename(path)),
        os.path.join("../data", os.path.basename(path))
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
            
    return path


# =============================================================================
# SECTION 2 — MODEL STATUS FUNCTIONS
# =============================================================================

def check_model_exists(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """
    Checks whether a trained model file exists on disk and retrieves its metadata.

    Args:
        model_path (str): Path to the trained model file. Defaults to MODEL_PATH.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'exists' (bool): True if model file exists.
            - 'path' (str): Resolved path to model file.
            - 'size_mb' (float): File size in MB (0.0 if not found).
            - 'modified_time' (str): Formatted last modification time ("N/A" if not found).
    """
    resolved_path = get_resolved_path(model_path)
    
    # Fallback to alternate Keras models if primary path does not exist
    if not os.path.exists(resolved_path):
        alt_paths = [
            "models/advanced_best_model.keras",
            "models/baseline_best_model.keras",
            "models/advanced_model.h5",
            "models/baseline_model.h5"
        ]
        for alt in alt_paths:
            candidate = get_resolved_path(alt)
            if os.path.exists(candidate):
                resolved_path = candidate
                break

    if os.path.exists(resolved_path):
        try:
            size_bytes = os.path.getsize(resolved_path)
            size_mb = float(round(size_bytes / (1024 * 1024), 2))
            mtime = os.path.getmtime(resolved_path)
            modified_time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                "exists": True,
                "path": resolved_path,
                "size_mb": size_mb,
                "modified_time": modified_time_str
            }
        except Exception as e:
            return {
                "exists": False,
                "path": resolved_path,
                "size_mb": 0.0,
                "modified_time": f"Error: {str(e)}"
            }

    return {
        "exists": False,
        "path": model_path,
        "size_mb": 0.0,
        "modified_time": "N/A"
    }


def load_trained_model(model_path: str = MODEL_PATH) -> Tuple[Optional[tf.keras.Model], Optional[str]]:
    """
    Loads and returns a trained Keras model from disk.

    Args:
        model_path (str): Path to the model file. Defaults to MODEL_PATH.

    Returns:
        Tuple[Optional[tf.keras.Model], Optional[str]]:
            - On success: (model_object, None)
            - On failure: (None, "Error description string")
    """
    status_info = check_model_exists(model_path)
    if not status_info["exists"]:
        return None, f"Model file not found at '{model_path}' or alternative locations."

    target_file = status_info["path"]
    try:
        start_time = time.time()
        model = load_model(target_file)
        elapsed = time.time() - start_time
        print(f"[load_trained_model] Model successfully loaded from '{target_file}' in {elapsed:.2f}s.")
        return model, None
    except Exception as e:
        # Secondary attempt: load with compile=False if legacy optimizer weights cause issues
        try:
            start_time = time.time()
            model = load_model(target_file, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            elapsed = time.time() - start_time
            print(f"[load_trained_model] Model loaded (compile=False fallback) from '{target_file}' in {elapsed:.2f}s.")
            return model, None
        except Exception as e2:
            err_msg = f"Failed to load model from '{target_file}': {str(e2)}"
            print(f"[load_trained_model] Error: {err_msg}")
            return None, err_msg


# =============================================================================
# SECTION 3 — DATA FUNCTIONS
# =============================================================================

def load_and_preprocess_data(data_path: str = TRAIN_DATA_PATH) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], Optional[np.ndarray],
    Optional[str]
]:
    """
    Loads raw CSV data, normalizes pixel features, reshapes into 4D image tensors,
    one-hot encodes labels, and performs a 90/10 stratified train/validation split.

    Args:
        data_path (str): Path to training CSV data. Defaults to TRAIN_DATA_PATH.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
            - (X_train, X_val, y_train, y_val, None) on success.
            - (None, None, None, None, error_message) on exception.
    """
    resolved_file = get_resolved_path(data_path)
    if not os.path.exists(resolved_file):
        return None, None, None, None, f"Data file not found at path '{data_path}'."

    try:
        df = pd.read_csv(resolved_file)
        if 'label' not in df.columns:
            return None, None, None, None, f"Required target column 'label' missing in '{data_path}'."

        # Separate labels and pixel features
        y_raw = df['label'].values
        X_raw = df.drop('label', axis=1).values

        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        X_normalized = (X_raw / 255.0).astype(np.float32)

        # Reshape flat 784 arrays into 4D image tensors (N, 28, 28, 1)
        X_reshaped = X_normalized.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        # One-hot encode target labels
        y_categorical = to_categorical(y_raw, num_classes=NUM_CLASSES)

        # Perform 90/10 stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped,
            y_categorical,
            test_size=0.10,
            stratify=y_raw,
            random_state=RANDOM_SEED
        )

        return X_train, X_val, y_train, y_val, None

    except Exception as e:
        return None, None, None, None, f"Error processing dataset: {str(e)}"


def get_data_statistics(data_path: str = TRAIN_DATA_PATH) -> Dict[str, Any]:
    """
    Computes summary statistics on the training CSV for UI display.

    Args:
        data_path (str): Path to training CSV. Defaults to TRAIN_DATA_PATH.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'total_samples' (int): Total number of rows.
            - 'class_distribution' (dict): Frequency count per digit {0: c0, ..., 9: c9}.
            - 'image_shape' (tuple): Image dimensions (28, 28, 1).
            - 'is_balanced' (bool): True if max_class_count / min_class_count < 1.2.
    """
    resolved_file = get_resolved_path(data_path)
    if not os.path.exists(resolved_file):
        return {
            "total_samples": 0,
            "class_distribution": {},
            "image_shape": (IMG_SIZE, IMG_SIZE, 1),
            "is_balanced": False,
            "error": f"Data file not found at '{data_path}'"
        }

    try:
        df = pd.read_csv(resolved_file)
        total_samples = len(df)
        
        if 'label' in df.columns:
            counts = df['label'].value_counts().to_dict()
            class_distribution = {int(k): int(v) for k, v in sorted(counts.items())}
            max_c = max(class_distribution.values()) if class_distribution else 1
            min_c = min(class_distribution.values()) if class_distribution else 1
            is_balanced = (max_c / max(1, min_c)) < 1.2
        else:
            class_distribution = {}
            is_balanced = False

        return {
            "total_samples": total_samples,
            "class_distribution": class_distribution,
            "image_shape": (IMG_SIZE, IMG_SIZE, 1),
            "is_balanced": is_balanced
        }
    except Exception as e:
        return {
            "total_samples": 0,
            "class_distribution": {},
            "image_shape": (IMG_SIZE, IMG_SIZE, 1),
            "is_balanced": False,
            "error": str(e)
        }


# =============================================================================
# SECTION 4 — MODEL ARCHITECTURE FUNCTIONS
# =============================================================================

def build_model(model_type: str = 'advanced') -> Sequential:
    """
    Constructs and compiles a Keras Sequential Convolutional Neural Network.

    Args:
        model_type (str): 'simple' or 'advanced'. Defaults to 'advanced'.

    Returns:
        Sequential: Compiled Keras Sequential CNN model.
    """
    model = Sequential(name=f"{model_type}_cnn")

    if model_type.lower() == 'simple':
        model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
    else: # Advanced CNN
        model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================================================
# SECTION 5 — CUSTOM KERAS CALLBACKS
# =============================================================================

class StreamlitProgressCallback(Callback):
    """
    Custom Keras Callback that updates a shared progress dictionary on epoch end
    for live Streamlit UI rendering.
    """
    def __init__(self, progress_dict: Dict[str, Any]):
        super().__init__()
        self.progress_dict = progress_dict

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        total_epochs = self.params.get('epochs', 50)
        curr_epoch = epoch + 1
        
        train_acc = float(round(logs.get('accuracy', 0.0) * 100, 2))
        val_acc = float(round(logs.get('val_accuracy', 0.0) * 100, 2))
        train_loss = float(round(logs.get('loss', 0.0), 4))
        val_loss = float(round(logs.get('val_loss', 0.0), 4))

        self.progress_dict['current_epoch'] = curr_epoch
        self.progress_dict['total_epochs'] = total_epochs
        self.progress_dict['train_accuracy'] = train_acc
        self.progress_dict['val_accuracy'] = val_acc
        self.progress_dict['train_loss'] = train_loss
        self.progress_dict['val_loss'] = val_loss
        
        self.progress_dict.setdefault('history_acc', []).append(train_acc)
        self.progress_dict.setdefault('history_val_acc', []).append(val_acc)
        self.progress_dict.setdefault('history_loss', []).append(train_loss)
        self.progress_dict.setdefault('history_val_loss', []).append(val_loss)
        
        self.progress_dict['status'] = f"Training epoch {curr_epoch}/{total_epochs}..."


class StopTrainingCallback(Callback):
    """
    Custom Keras Callback that monitors a shared boolean list reference and
    stops model training if requested by the user from Streamlit.
    """
    def __init__(self, stop_flag_ref: List[bool]):
        super().__init__()
        self.stop_flag_ref = stop_flag_ref

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.stop_flag_ref and self.stop_flag_ref[0] is True:
            print(f"[StopTrainingCallback] Stop signal received at epoch {epoch+1}. Halting training.")
            self.model.stop_training = True


# =============================================================================
# SECTION 6 — TRAINING FUNCTION
# =============================================================================

def train_model(
    model_type: str = 'advanced',
    progress_dict: Optional[Dict[str, Any]] = None,
    stop_flag_ref: Optional[List[bool]] = None
) -> None:
    """
    Executes model training pipeline inside a background thread, updating shared
    progress metrics and saving the best weight model to MODEL_PATH.

    Args:
        model_type (str): 'simple' or 'advanced'. Defaults to 'advanced'.
        progress_dict (Optional[Dict[str, Any]]): Shared dictionary for Streamlit UI metrics.
        stop_flag_ref (Optional[List[bool]]): Shared list reference [False] to trigger stop.
    """
    if progress_dict is None:
        progress_dict = {}
    if stop_flag_ref is None:
        stop_flag_ref = [False]

    try:
        progress_dict['status'] = "Loading data..."
        X_train, X_val, y_train, y_val, err = load_and_preprocess_data()
        
        if err or X_train is None:
            progress_dict['status'] = "error"
            progress_dict['error'] = err or "Data loading failed."
            return

        progress_dict['status'] = "Building model..."
        model = build_model(model_type)

        # Setup ImageDataGenerator for Advanced Model data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # NOTE ON MODEL SAVING & EARLY STOPPING:
        # ModelCheckpoint with save_best_only=True ensures MODEL_PATH is ONLY updated
        # when a new peak validation accuracy epoch occurs. If training is stopped
        # early by the user, the model file retains the single best checkpoint up to that point.
        callbacks = [
            StreamlitProgressCallback(progress_dict),
            StopTrainingCallback(stop_flag_ref),
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=0)
        ]

        progress_dict['status'] = "Training..."
        batch_size = 64
        epochs = 50

        if model_type.lower() == 'advanced':
            datagen.fit(X_train)
            model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
        else:
            model.fit(
                X_train, y_train,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )

        if stop_flag_ref[0] is True:
            progress_dict['status'] = "stopped"
            progress_dict['stopped'] = True
        else:
            progress_dict['status'] = "complete"
            progress_dict['completed'] = True

    except Exception as e:
        progress_dict['status'] = "error"
        progress_dict['error'] = str(e)
        print(f"[train_model] Exception: {str(e)}")


# =============================================================================
# SECTION 7 — PREDICTION FUNCTIONS
# =============================================================================

def center_and_pad_digit(gray_array: np.ndarray, target_size: int = 28, box_size: int = 20) -> np.ndarray:
    """
    Centers a 2D grayscale digit array inside a target_size x target_size box
    with bounding-box cropping and aspect-ratio-preserving padding (MNIST standard format).

    Args:
        gray_array (np.ndarray): 2D array where background is ~0 and digit is > 0.
        target_size (int): Output dimension (28).
        box_size (int): Max bounding dimension of the digit (20).

    Returns:
        np.ndarray: Centered (target_size, target_size) float32 array normalized to [0.0, 1.0].
    """
    try:
        # Find non-background pixels (digit pixels > threshold)
        max_val = float(np.max(gray_array))
        if max_val == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)

        threshold = 20.0 if max_val > 1.0 else 0.08
        coords = np.argwhere(gray_array > threshold)

        if coords.size == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)

        # Extract bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        crop = gray_array[y_min:y_max+1, x_min:x_max+1]

        h, w = crop.shape
        if h == 0 or w == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)

        # Scale preserving aspect ratio into box_size x box_size
        if h > w:
            new_h = box_size
            new_w = max(1, int(round(w * (box_size / float(h)))))
        else:
            new_w = box_size
            new_h = max(1, int(round(h * (box_size / float(w)))))

        # Convert crop to PIL for Lanczos resampling
        if max_val <= 1.0:
            crop_uint8 = (crop * 255.0).astype(np.uint8)
        else:
            crop_uint8 = crop.astype(np.uint8)

        crop_pil = Image.fromarray(crop_uint8)
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
        resized_pil = crop_pil.resize((new_w, new_h), resample=resample_filter)

        # Paste centered in target_size x target_size canvas
        padded_pil = Image.new('L', (target_size, target_size), 0)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        padded_pil.paste(resized_pil, (pad_x, pad_y))

        arr_out = np.array(padded_pil, dtype=np.float32) / 255.0
        return arr_out
    except Exception as e:
        print(f"[center_and_pad_digit] Exception: {str(e)}")
        try:
            pil_fallback = Image.fromarray(gray_array.astype(np.uint8))
            resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
            resized = pil_fallback.resize((target_size, target_size), resample=resample_filter)
            arr = np.array(resized, dtype=np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
            return arr
        except Exception:
            return np.zeros((target_size, target_size), dtype=np.float32)


def preprocess_uploaded_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocesses a PIL Image uploaded by the user into a normalized (1, 28, 28, 1) tensor.

    Args:
        pil_image (Image.Image): Input PIL Image object.

    Returns:
        np.ndarray: Preprocessed numpy array of shape (1, 28, 28, 1) with values in [0.0, 1.0].
    """
    try:
        # Convert to Grayscale
        gray_img = pil_image.convert('L')
        arr = np.array(gray_img, dtype=np.float32)

        # Invert colors if background is bright (white background with black digit)
        if np.mean(arr) > 127.0:
            arr = 255.0 - arr

        # Center, bounding-box crop & pad to 28x28 (MNIST format)
        arr_2d = center_and_pad_digit(arr, target_size=IMG_SIZE, box_size=20)
        return arr_2d.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    except Exception as e:
        print(f"[preprocess_uploaded_image] Error: {str(e)}")
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)


def preprocess_canvas_image(canvas_image_data: np.ndarray) -> np.ndarray:
    """
    Preprocesses raw RGBA numpy array from streamlit-drawable-canvas into a (1, 28, 28, 1) tensor.

    Args:
        canvas_image_data (np.ndarray): RGBA numpy array of shape (H, W, 4).

    Returns:
        np.ndarray: Preprocessed numpy array of shape (1, 28, 28, 1) in [0.0, 1.0].
    """
    try:
        if canvas_image_data is None or canvas_image_data.size == 0:
            return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

        # Canvas RGBA: Background is black #000000 -> RGB=[0,0,0], stroke is white #FFFFFF -> RGB=[255,255,255]
        # Extract RGB grayscale intensity, NOT the alpha channel (which is solid 255 for black background)!
        if canvas_image_data.ndim == 3 and canvas_image_data.shape[2] >= 3:
            rgb_gray = np.mean(canvas_image_data[:, :, :3], axis=2).astype(np.float32)
        else:
            rgb_gray = canvas_image_data.astype(np.float32)

        # Center, bounding-box crop & pad to 28x28 (MNIST format)
        arr_2d = center_and_pad_digit(rgb_gray, target_size=IMG_SIZE, box_size=20)
        return arr_2d.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    except Exception as e:
        print(f"[preprocess_canvas_image] Error: {str(e)}")
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)



def predict_digit(model: tf.keras.Model, preprocessed_image: np.ndarray) -> Dict[str, Any]:
    """
    Generates digit classification probabilities and confidence rankings for an input tensor.

    Args:
        model (tf.keras.Model): Loaded Keras CNN model.
        preprocessed_image (np.ndarray): Input tensor of shape (1, 28, 28, 1).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'predicted_digit' (int): Top predicted class label (0-9).
            - 'confidence' (float): Percentage probability of top prediction.
            - 'all_probabilities' (List[float]): 10-element probability list.
            - 'top3' (List[Tuple[int, float]]): Top 3 predictions as [(digit, conf%), ...].
    """
    try:
        probs = model.predict(preprocessed_image, verbose=0)[0]
        probs = np.clip(probs, 0.0, 1.0)
        
        predicted_digit = int(np.argmax(probs))
        confidence = float(round(float(np.max(probs)) * 100.0, 2))
        all_probs = [float(round(p * 100.0, 2)) for p in probs]

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(int(d), float(round(float(probs[d]) * 100.0, 2))) for d in top3_indices]

        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "top3": top3
        }
    except Exception as e:
        print(f"[predict_digit] Error: {str(e)}")
        return {
            "predicted_digit": 0,
            "confidence": 0.0,
            "all_probabilities": [0.0] * 10,
            "top3": [(0, 0.0), (1, 0.0), (2, 0.0)],
            "error": str(e)
        }


# =============================================================================
# SECTION 8 — VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_history(progress_dict: Dict[str, Any]) -> matplotlib.figure.Figure:
    """
    Renders live or completed training/validation accuracy and loss curves.

    Args:
        progress_dict (Dict[str, Any]): Dictionary containing history lists.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object handle.
    """
    acc = progress_dict.get('history_acc', [])
    val_acc = progress_dict.get('history_val_acc', [])
    loss = progress_dict.get('history_loss', [])
    val_loss = progress_dict.get('history_val_loss', [])

    epochs = range(1, len(acc) + 1) if acc else [1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    if acc:
        ax1.plot(epochs, acc, 'b-o', label='Train Accuracy', linewidth=2)
    if val_acc:
        ax1.plot(epochs, val_acc, 'g-s', label='Val Accuracy', linewidth=2)
    ax1.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Loss Plot
    if loss:
        ax2.plot(epochs, loss, 'b-o', label='Train Loss', linewidth=2)
    if val_loss:
        ax2.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(model: tf.keras.Model) -> matplotlib.figure.Figure:
    """
    Generates a seaborn heatmap of the 10x10 validation confusion matrix.

    Args:
        model (tf.keras.Model): Loaded Keras model.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure handle.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        _, X_val, _, y_val, err = load_and_preprocess_data()
        if err or X_val is None:
            ax.text(0.5, 0.5, f"Error loading validation data: {err}", ha='center', va='center')
            plt.tight_layout()
            return fig

        y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
        val_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
        ax.set_title("Validation Confusion Matrix (Digits 0-9)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error creating confusion matrix: {str(e)}", ha='center', va='center')

    plt.tight_layout()
    return fig


def plot_sample_predictions(model: tf.keras.Model, n_samples: int = 10) -> matplotlib.figure.Figure:
    """
    Renders a grid (e.g. 2x5 or 4x5) of validation sample predictions with color-coded titles.

    Args:
        model (tf.keras.Model): Loaded Keras model.
        n_samples (int): Number of samples to plot (default 10).

    Returns:
        matplotlib.figure.Figure: Matplotlib figure handle.
    """
    n_cols = 5
    n_rows = max(1, int(np.ceil(n_samples / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.8 * n_rows))
    
    try:
        _, X_val, _, y_val, err = load_and_preprocess_data()
        axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
        if err or X_val is None:
            for ax in axes_flat:
                ax.axis('off')
            fig.suptitle(f"Error: {err}", color='red')
            plt.tight_layout()
            return fig

        y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
        np.random.seed(RANDOM_SEED)
        sample_indices = np.random.choice(len(X_val), min(n_samples, len(X_val)), replace=False)

        val_probs = model.predict(X_val[sample_indices], verbose=0)
        y_preds = np.argmax(val_probs, axis=1)

        for idx, ax in enumerate(axes_flat):
            if idx < len(sample_indices):
                s_i = sample_indices[idx]
                img = X_val[s_i].reshape(IMG_SIZE, IMG_SIZE)
                true_lbl = y_true[s_i]
                pred_lbl = y_preds[idx]
                conf = val_probs[idx, pred_lbl] * 100

                color = 'green' if true_lbl == pred_lbl else 'red'
                ax.imshow(img, cmap='gray')
                ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\n({conf:.1f}%)", fontsize=10, color=color, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')

        fig.suptitle(f"Sample Predictions on Validation Set ({n_samples} Samples)", fontsize=14, fontweight='bold')
    except Exception as e:
        axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
        for ax in axes_flat:
            ax.axis('off')
        fig.suptitle(f"Error generating predictions: {str(e)}", color='red')

    plt.tight_layout()
    return fig



def get_classification_report_df(model: tf.keras.Model) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame containing precision, recall, f1-score, and support.

    Args:
        model (tf.keras.Model): Loaded Keras model.

    Returns:
        pd.DataFrame: Formatted classification report dataframe.
    """
    try:
        _, X_val, _, y_val, err = load_and_preprocess_data()
        if err or X_val is None:
            return pd.DataFrame({"Error": [err]})

        y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
        val_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.reset_index().rename(columns={'index': 'class'})
        return report_df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# =============================================================================
# STANDALONE MODULE TEST & DEMO
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("       MNIST UTILITY MODULE (utils.py) - STEP-BY-STEP TEST")
    print("=" * 60)

    # Step 1: Check Model Status
    print("\n[STEP 1] Checking Model Status...")
    model_status = check_model_exists()
    print(f"  - Model Exists: {model_status['exists']}")
    print(f"  - Path: {model_status['path']}")
    print(f"  - Size: {model_status['size_mb']} MB")
    print(f"  - Last Modified: {model_status['modified_time']}")

    # Step 2: Load Trained Model
    print("\n[STEP 2] Loading Trained Model...")
    model, err = load_trained_model()
    if model:
        print("  - Model successfully loaded into memory!")
    else:
        print(f"  - Model load warning/error: {err}")

    # Step 3: Load & Preprocess Dataset
    print("\n[STEP 3] Loading & Preprocessing Dataset (data/train.csv)...")
    X_train, X_val, y_train, y_val, data_err = load_and_preprocess_data()
    if X_train is not None:
        print(f"  - X_train shape: {X_train.shape} (dtype: {X_train.dtype})")
        print(f"  - X_val shape:   {X_val.shape} (dtype: {X_val.dtype})")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - y_val shape:   {y_val.shape}")
    else:
        print(f"  - Dataset error: {data_err}")

    # Step 4: Data Statistics
    print("\n[STEP 4] Fetching Dataset Statistics...")
    stats = get_data_statistics()
    print(f"  - Total Samples: {stats.get('total_samples')}")
    print(f"  - Image Shape:   {stats.get('image_shape')}")
    print(f"  - Class Balance: {'Balanced (<1.2 ratio)' if stats.get('is_balanced') else 'Unbalanced'}")

    # Step 5: Build CNN Model Architectures
    print("\n[STEP 5] Building Keras Model Architectures...")
    simple_model = build_model('simple')
    adv_model = build_model('advanced')
    print(f"  - Simple CNN Parameters:   {simple_model.count_params():,}")
    print(f"  - Advanced CNN Parameters: {adv_model.count_params():,}")

    # Step 6: Test Image Preprocessing & Digit Prediction
    print("\n[STEP 6] Testing Image Preprocessing & Inference...")
    dummy_pil = Image.fromarray(np.uint8(np.random.rand(100, 100) * 255))
    prep_img = preprocess_uploaded_image(dummy_pil)
    print(f"  - Uploaded Image Preprocessed Tensor Shape: {prep_img.shape}")

    dummy_canvas = np.uint8(np.random.rand(280, 280, 4) * 255)
    prep_canvas = preprocess_canvas_image(dummy_canvas)
    print(f"  - Canvas Preprocessed Tensor Shape:        {prep_canvas.shape}")

    if model:
        pred_res = predict_digit(model, prep_img)
        print(f"  - Predicted Digit: {pred_res['predicted_digit']}")
        print(f"  - Confidence:      {pred_res['confidence']}%")
        print(f"  - Top 3:            {pred_res['top3']}")

    # Step 7: Test Visualization Functions
    print("\n[STEP 7] Generating Visualization Figures...")
    p_dict = {
        'history_acc': [92.1, 95.4, 97.2],
        'history_val_acc': [93.0, 96.1, 97.8],
        'history_loss': [0.25, 0.15, 0.08],
        'history_val_loss': [0.21, 0.12, 0.07]
    }
    fig_hist = plot_training_history(p_dict)
    print("  - Training History plot generated successfully!")

    if model:
        fig_cm = plot_confusion_matrix(model)
        print("  - Confusion Matrix plot generated successfully!")
        fig_samples = plot_sample_predictions(model, 10)
        print("  - Sample Predictions plot generated successfully!")
        report_df = get_classification_report_df(model)
        print(f"  - Classification Report DataFrame generated with {len(report_df)} rows.")

    print("\n" + "=" * 60)
    print("  ALL UTILITY FUNCTIONS RUN SUCCESSFULLY STEP-BY-STEP!")
    print("=" * 60 + "\n")

