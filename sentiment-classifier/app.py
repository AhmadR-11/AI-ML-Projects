"""Streamlit Web Application for Twitter Sentiment Classification & Model Comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    REVERSE_CLASS_MAPPING,
)
from src.predict import predict_single_embedding, predict_single_tfidf

# Page Configuration
st.set_page_config(
    page_title="Twitter Sentiment Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Vanilla CSS Injection)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .sentiment-positive {
        color: #10B981;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #64748B;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #EF4444;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_dict=True,
)


def render_sidebar():
    st.sidebar.title("✈️ Navigation & Settings")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "Real-Time Sentiment Sandbox",
            "Model Performance Comparison",
            "Airline Slice Analysis",
            "Dataset & EDA Overview",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Details:**")
    st.sidebar.markdown("- **Primary Metric:** Macro-F1")
    st.sidebar.markdown("- **Models:** TF-IDF vs. Dense Embeddings")
    st.sidebar.markdown("- **Embedding Model:** `all-MiniLM-L6-v2`")
    return page


def render_sandbox():
    st.markdown("<div class='main-header'>Real-Time Sentiment Prediction Sandbox</div>", unsafe_allow_dict=True)
    st.markdown("<div class='sub-header'>Test tweet sentiment classification using TF-IDF or Dense Sentence Embeddings.</div>", unsafe_allow_dict=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "Enter Tweet Text:",
            value="@USAirways My flight was delayed by 4 hours and no one helped us at the gate!",
            height=120,
        )

    with col2:
        approach = st.selectbox(
            "Select Inference Approach:",
            options=["embedding", "tfidf"],
            format_func=lambda x: "Dense Embeddings (all-MiniLM-L6-v2)" if x == "embedding" else "TF-IDF + Scikit-Learn Classifier",
        )
        predict_btn = st.button("Predict Sentiment", type="primary", use_container_width=True)

    if predict_btn and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            if approach == "tfidf":
                res = predict_single_tfidf(user_input)
            else:
                res = predict_single_embedding(user_input)

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            sentiment = res["sentiment"].upper()
            sentiment_color = "#10B981" if sentiment == "POSITIVE" else ("#EF4444" if sentiment == "NEGATIVE" else "#64748B")

            st.markdown(f"### Predicted Sentiment: <span style='color:{sentiment_color};'>{sentiment}</span>", unsafe_allow_dict=True)
            st.markdown(f"**Cleaned Text:** `{res['clean_text']}`")
            st.markdown(f"**Target Label ID:** `{res['predicted_label']}`")

        with res_col2:
            st.markdown("### Class Confidence Breakdown")
            prob_df = pd.DataFrame(
                list(res["probabilities"].items()), columns=["Sentiment", "Confidence"]
            )
            st.bar_chart(prob_df.set_index("Sentiment"))


def render_model_comparison():
    st.markdown("<div class='main-header'>Model Evaluation & Performance Comparison</div>", unsafe_allow_dict=True)
    st.markdown("<div class='sub-header'>Side-by-side comparison of TF-IDF baseline vs. Dense Sentence Embedding approaches on the test split.</div>", unsafe_allow_dict=True)

    comp_file = METRICS_DIR / "test_evaluation_comparison.json"
    if not comp_file.exists():
        st.warning("Test evaluation metrics not found. Please run: `python -m src.evaluate`")
        return

    with open(comp_file, "r") as f:
        metrics = json.load(f)

    tfidf_m = metrics["TF-IDF_Approach"]
    emb_m = metrics["Embedding_Approach"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TF-IDF Macro-F1", f"{tfidf_m['macro_f1']:.4f}")
    col2.metric("Embedding Macro-F1", f"{emb_m['macro_f1']:.4f}", delta=f"{emb_m['macro_f1'] - tfidf_m['macro_f1']:.4f}")
    col3.metric("TF-IDF Accuracy", f"{tfidf_m['accuracy']:.4f}")
    col4.metric("Embedding Accuracy", f"{emb_m['accuracy']:.4f}", delta=f"{emb_m['accuracy'] - tfidf_m['accuracy']:.4f}")

    st.markdown("---")
    st.subheader("Per-Class F1 Score Comparison")

    classes = ["negative", "neutral", "positive"]
    class_df = pd.DataFrame(
        {
            "Class": classes,
            "TF-IDF F1": [tfidf_m["per_class_f1"][c] for c in classes],
            "Embedding F1": [emb_m["per_class_f1"][c] for c in classes],
        }
    )
    st.dataframe(class_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix Visualization")
    cm_path = FIGURES_DIR / "eval_confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Side-by-Side Confusion Matrices (TF-IDF vs Embeddings)", use_column_width=True)


def render_airline_slice():
    st.markdown("<div class='main-header'>Airline Subpopulation Slice Analysis</div>", unsafe_allow_dict=True)
    st.markdown("<div class='sub-header'>Evaluating classification performance across individual airline sub-groups.</div>", unsafe_allow_dict=True)

    slice_file = METRICS_DIR / "test_airline_slice_evaluation.csv"
    if not slice_file.exists():
        st.warning("Slice metrics not found. Please run: `python -m src.evaluate`")
        return

    df_slice = pd.read_csv(slice_file)
    st.dataframe(df_slice, use_container_width=True)

    st.markdown("---")
    st.subheader("Macro-F1 Performance by Airline")
    st.bar_chart(df_slice.set_index("Airline")[["TF-IDF Macro-F1", "Embedding Macro-F1"]])


def render_eda():
    st.markdown("<div class='main-header'>Dataset & Exploratory Data Analysis Overview</div>", unsafe_allow_dict=True)
    st.markdown("<div class='sub-header'>Kaggle Twitter US Airline Sentiment Dataset statistics and visualizations.</div>", unsafe_allow_dict=True)

    eda_file = METRICS_DIR / "eda_summary.json"
    if eda_file.exists():
        with open(eda_file, "r") as f:
            eda_data = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tweets", f"{eda_data['total_tweets']:,}")
        col2.metric("Negative Class Ratio", f"{eda_data['class_percentages'].get('negative', 0)}%")
        col3.metric("Airlines Represented", len(eda_data["airlines"]))

    st.markdown("---")
    st.subheader("EDA Visualizations")

    fig1 = FIGURES_DIR / "eda_class_distribution.png"
    fig2 = FIGURES_DIR / "eda_airline_sentiment.png"

    col_a, col_b = st.columns(2)
    with col_a:
        if fig1.exists():
            st.image(str(fig1), caption="Target Class Imbalance", use_column_width=True)
    with col_b:
        if fig2.exists():
            st.image(str(fig2), caption="Sentiment per Airline", use_column_width=True)


def main():
    selected_page = render_sidebar()

    if selected_page == "Real-Time Sentiment Sandbox":
        render_sandbox()
    elif selected_page == "Model Performance Comparison":
        render_model_comparison()
    elif selected_page == "Airline Slice Analysis":
        render_airline_slice()
    elif selected_page == "Dataset & EDA Overview":
        render_eda()


if __name__ == "__main__":
    main()
