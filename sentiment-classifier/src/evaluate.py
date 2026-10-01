"""Final evaluation module testing both TF-IDF and Embedding models on held-out test dataset."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BEST_EMBEDDING_MODEL_PATH,
    BEST_TFIDF_MODEL_PATH,
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_DIR,
    PYTORCH_EMBEDDING_MODEL_PATH,
    TEST_DATA_PATH,
    TEST_EMBEDDINGS_PATH,
    TFIDF_VECTORIZER_PATH,
)
from src.features_tfidf import transform_texts
from src.train_embeddings import PyTorchSentimentMLP, predict_pytorch_mlp
from src.utils import get_logger, save_figure, save_json_metrics

logger = get_logger("evaluate")


def load_tfidf_pipeline() -> Tuple[Any, Any]:
    """Loads fitted TF-IDF vectorizer and best trained model."""
    if not TFIDF_VECTORIZER_PATH.exists() or not BEST_TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError("TF-IDF vectorizer or model missing. Run training scripts first.")

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    artifact = joblib.load(BEST_TFIDF_MODEL_PATH)
    model = artifact["estimator"]
    return vectorizer, model


def load_embedding_pipeline() -> Tuple[str, Any]:
    """Loads trained dense embedding classifier artifact."""
    if not BEST_EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Embedding model missing at {BEST_EMBEDDING_MODEL_PATH}.")

    artifact = joblib.load(BEST_EMBEDDING_MODEL_PATH)
    m_type = artifact.get("type", "sklearn")

    if m_type == "pytorch":
        model = PyTorchSentimentMLP(input_dim=384, num_classes=3)
        model.load_state_dict(torch.load(PYTORCH_EMBEDDING_MODEL_PATH))
        model.eval()
        return "pytorch", model
    else:
        return "sklearn", artifact["estimator"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics including primary Macro-F1."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro"))
    rec_macro = float(recall_score(y_true, y_pred, average="macro"))

    per_class = f1_score(y_true, y_pred, average=None)
    class_f1_dict = {
        cls_name: round(float(score), 4) for cls_name, score in zip(CLASS_NAMES, per_class)
    }

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "per_class_f1": class_f1_dict,
    }


def plot_side_by_side_confusion_matrices(
    cm_tfidf: np.ndarray, cm_emb: np.ndarray, force: bool = False
) -> Path:
    """Generates side-by-side Confusion Matrix comparison plots."""
    output_path = FIGURES_DIR / "eval_confusion_matrices.png"
    if output_path.exists() and not force:
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm_tfidf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("TF-IDF Approach Confusion Matrix", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        cm_emb,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Embedding Approach Confusion Matrix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    save_figure(fig, output_path)
    return output_path


def run_slice_based_evaluation(
    test_df: pd.DataFrame, tfidf_preds: np.ndarray, emb_preds: np.ndarray
) -> pd.DataFrame:
    """Computes Macro-F1 score sliced by airline group."""
    airlines = sorted(test_df["airline"].dropna().unique())
    slice_data = []

    for airline in airlines:
        mask = (test_df["airline"] == airline).values
        sub_y = test_df["label"].values[mask]
        sub_tfidf = tfidf_preds[mask]
        sub_emb = emb_preds[mask]

        f1_tfidf = f1_score(sub_y, sub_tfidf, average="macro")
        f1_emb = f1_score(sub_y, sub_emb, average="macro")

        slice_data.append(
            {
                "Airline": airline,
                "Sample Count": int(mask.sum()),
                "TF-IDF Macro-F1": round(float(f1_tfidf), 4),
                "Embedding Macro-F1": round(float(f1_emb), 4),
            }
        )

    slice_df = pd.DataFrame(slice_data)
    slice_df.to_csv(METRICS_DIR / "test_airline_slice_evaluation.csv", index=False)
    logger.info(f"Saved slice-based evaluation to {METRICS_DIR / 'test_airline_slice_evaluation.csv'}")
    return slice_df


def evaluate_all(force: bool = False) -> None:
    """Runs complete test set evaluation for both TF-IDF and Embedding approaches."""
    logger.info("Starting Final Evaluation Phase on Held-Out Test Set...")
    if not TEST_DATA_PATH.exists() or not TEST_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("Test data or test embeddings missing. Run pipeline steps first.")

    test_df = pd.read_csv(TEST_DATA_PATH)
    y_test = test_df["label"].values
    test_emb = np.load(TEST_EMBEDDINGS_PATH)

    # 1. TF-IDF Inference
    tfidf_vectorizer, tfidf_model = load_tfidf_pipeline()
    X_test_tfidf = transform_texts(tfidf_vectorizer, test_df["clean_text"])
    tfidf_preds = tfidf_model.predict(X_test_tfidf)

    # 2. Embedding Inference
    emb_type, emb_model = load_embedding_pipeline()
    if emb_type == "pytorch":
        emb_preds = predict_pytorch_mlp(emb_model, test_emb)
    else:
        emb_preds = emb_model.predict(test_emb)

    # Compute Metrics
    tfidf_metrics = compute_metrics(y_test, tfidf_preds)
    emb_metrics = compute_metrics(y_test, emb_preds)

    summary_comparison = {
        "TF-IDF_Approach": tfidf_metrics,
        "Embedding_Approach": emb_metrics,
    }

    # Save Metrics
    save_json_metrics(summary_comparison, METRICS_DIR / "test_evaluation_comparison.json")

    summary_df = pd.DataFrame(
        [
            {"Approach": "TF-IDF", **tfidf_metrics},
            {"Approach": "Dense Embedding (all-MiniLM-L6-v2)", **emb_metrics},
        ]
    )
    summary_df.to_csv(METRICS_DIR / "test_evaluation_comparison.csv", index=False)

    # Plot Confusion Matrices
    cm_tfidf = confusion_matrix(y_test, tfidf_preds)
    cm_emb = confusion_matrix(y_test, emb_preds)
    plot_side_by_side_confusion_matrices(cm_tfidf, cm_emb, force=force)

    # Slice-based evaluation
    run_slice_based_evaluation(test_df, tfidf_preds, emb_preds)

    logger.info("=== FINAL TEST SET PERFORMANCE COMPARISON ===")
    logger.info(f"TF-IDF    Approach Macro-F1: {tfidf_metrics['macro_f1']} (Acc: {tfidf_metrics['accuracy']})")
    logger.info(f"Embedding Approach Macro-F1: {emb_metrics['macro_f1']} (Acc: {emb_metrics['accuracy']})")


def main() -> None:
    """CLI handler for running final evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF and Embedding approaches on test set.")
    parser.add_argument("--force", action="store_true", help="Force overwriting evaluation outputs.")
    args = parser.parse_args()

    evaluate_all(force=args.force)


if __name__ == "__main__":
    main()
