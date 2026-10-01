"""Data loading, validation, and structural cleaning pipeline for Twitter sentiment dataset."""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import (
    CONFIDENCE,
    LABEL2ID,
    METRICS_DIR,
    MIN_CONFIDENCE,
    OUT_AIRLINE,
    OUT_CONFIDENCE,
    OUT_LABEL,
    OUT_LABEL_ID,
    OUT_NEG_REASON,
    OUT_TEXT_RAW,
    OUT_TWEET_ID,
    RAW_DATA_PATH,
    VALIDATED_DATA_PATH,
)
from src.utils import get_logger, save_json

logger = get_logger("data_loader")

REQUIRED_RAW_COLUMNS = [
    "tweet_id",
    "airline_sentiment",
    "airline_sentiment_confidence",
    "negativereason",
    "airline",
    "text",
]


def load_raw_data(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw Twitter dataset from CSV and verifies required column schema.

    Args:
        csv_path: Path to Tweets.csv file.

    Returns:
        pd.DataFrame containing raw loaded data.

    Raises:
        FileNotFoundError: If input CSV file is missing.
        ValueError: If parse error or required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset file missing at path: {csv_path}")

    logger.info(f"Loading raw dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file at {csv_path}: {e}") from e

    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Raw dataset missing required columns: {missing_cols}. Expected: {REQUIRED_RAW_COLUMNS}"
        )

    logger.info(f"Successfully loaded raw CSV with {len(df)} rows and {len(df.columns)} columns.")
    return df


def generate_validation_report(df: pd.DataFrame, save_report: bool = True) -> Dict[str, Any]:
    """Generates comprehensive data validation report and saves JSON metric artifact.

    Args:
        df: Input DataFrame (raw or processed).
        save_report: If True, saves JSON report to results/metrics/data_validation_report.json.

    Returns:
        Dictionary of validation statistics.
    """
    total_rows = int(len(df))
    total_cols = int(len(df.columns))

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}

    tweet_id_col = "tweet_id" if "tweet_id" in df.columns else OUT_TWEET_ID
    text_col = "text" if "text" in df.columns else OUT_TEXT_RAW
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else OUT_LABEL
    conf_col = "airline_sentiment_confidence" if "airline_sentiment_confidence" in df.columns else OUT_CONFIDENCE

    unique_tweet_ids = int(df[tweet_id_col].nunique()) if tweet_id_col in df.columns else 0
    duplicate_tweet_ids = total_rows - unique_tweet_ids
    duplicate_texts = int(df[text_col].duplicated().sum()) if text_col in df.columns else 0

    label_counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    label_pcts = (
        (df[label_col].value_counts(normalize=True) * 100).round(2).to_dict()
        if label_col in df.columns
        else {}
    )

    conf_series = df[conf_col].dropna() if conf_col in df.columns else pd.Series([], dtype=float)
    confidence_stats = {
        "min": round(float(conf_series.min()), 4) if not conf_series.empty else 0.0,
        "mean": round(float(conf_series.mean()), 4) if not conf_series.empty else 0.0,
        "max": round(float(conf_series.max()), 4) if not conf_series.empty else 0.0,
    }

    text_series = df[text_col].astype(str) if text_col in df.columns else pd.Series([], dtype=str)
    char_lens = text_series.apply(len)
    word_lens = text_series.apply(lambda s: len(s.split()))

    text_length_stats = {
        "char_len": {
            "min": int(char_lens.min()) if not char_lens.empty else 0,
            "mean": round(float(char_lens.mean()), 2) if not char_lens.empty else 0.0,
            "max": int(char_lens.max()) if not char_lens.empty else 0,
        },
        "word_len": {
            "min": int(word_lens.min()) if not word_lens.empty else 0,
            "mean": round(float(word_lens.mean()), 2) if not word_lens.empty else 0.0,
            "max": int(word_lens.max()) if not word_lens.empty else 0,
        },
    }

    report = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "unique_tweet_ids": unique_tweet_ids,
        "duplicate_tweet_ids": duplicate_tweet_ids,
        "duplicate_texts": duplicate_texts,
        "label_distribution": {
            "counts": label_counts,
            "percentages": label_pcts,
        },
        "confidence_stats": confidence_stats,
        "text_length_stats": text_length_stats,
    }

    if save_report:
        report_path = METRICS_DIR / "data_validation_report.json"
        save_json(report, report_path)
        logger.info(f"Validation report saved to {report_path}")

    return report


def clean_structural_issues(
    df: pd.DataFrame, min_confidence: float = MIN_CONFIDENCE
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Performs structural cleaning in exact mandated order and logs row removals.

    Steps:
        a) Drop missing/empty text or label.
        b) Drop duplicate tweet_id rows (keep first).
        c) Handle duplicate texts: drop exact text duplicates; if conflicting labels exist, drop ALL of them.
        d) Keep only valid labels (negative, neutral, positive).
        e) Filter airline_sentiment_confidence >= min_confidence.

    Args:
        df: Input raw DataFrame.
        min_confidence: Confidence threshold.

    Returns:
        Tuple of (cleaned_df, removal_audit_dict).
    """
    initial_rows = len(df)
    logger.info(f"Starting structural cleaning pipeline on {initial_rows} raw rows...")

    # Step a: Drop missing/empty text or label
    step_a_mask = (
        df["text"].notnull()
        & (df["text"].astype(str).str.strip().str.len() > 0)
        & df["airline_sentiment"].notnull()
        & (df["airline_sentiment"].astype(str).str.strip().str.len() > 0)
    )
    df_step_a = df[step_a_mask].copy()
    rows_removed_a = initial_rows - len(df_step_a)
    logger.info(f"Step (a) - Missing/empty text or label removed: {rows_removed_a} rows")

    # Step b: Drop duplicate tweet_id rows (keep first)
    before_b = len(df_step_a)
    df_step_b = df_step_a.drop_duplicates(subset=["tweet_id"], keep="first").copy()
    rows_removed_b = before_b - len(df_step_b)
    logger.info(f"Step (b) - Duplicate tweet_ids removed: {rows_removed_b} rows")

    # Step c: Handle duplicate texts (conflicting labels -> drop all; identical labels -> keep first)
    before_c = len(df_step_b)
    # Standardize label strings temporarily for text duplicate check
    df_step_b["temp_label"] = df_step_b["airline_sentiment"].astype(str).str.strip().str.lower()

    # Identify conflicting texts (same text, different labels)
    text_label_groups = df_step_b.groupby("text")["temp_label"].nunique()
    conflicting_texts = set(text_label_groups[text_label_groups > 1].index)

    # Drop ALL rows with conflicting texts
    n_conflicting_rows = df_step_b["text"].isin(conflicting_texts).sum()
    df_non_conflicting = df_step_b[~df_step_b["text"].isin(conflicting_texts)].copy()

    # Drop exact duplicate texts with identical labels (keep first)
    before_exact_dup = len(df_non_conflicting)
    df_step_c = df_non_conflicting.drop_duplicates(subset=["text"], keep="first").copy()
    n_exact_dup_dropped = before_exact_dup - len(df_step_c)

    rows_removed_c = n_conflicting_rows + n_exact_dup_dropped
    logger.info(
        f"Step (c) - Duplicate texts removed: {rows_removed_c} rows "
        f"({n_conflicting_rows} conflicting label rows, {n_exact_dup_dropped} exact duplicate rows)"
    )

    # Step d: Keep only valid labels (negative, neutral, positive)
    before_d = len(df_step_c)
    valid_labels = set(LABEL2ID.keys())
    df_step_c["temp_label"] = df_step_c["airline_sentiment"].astype(str).str.strip().str.lower()
    df_step_d = df_step_c[df_step_c["temp_label"].isin(valid_labels)].copy()
    rows_removed_d = before_d - len(df_step_d)
    logger.info(f"Step (d) - Invalid sentiment labels removed: {rows_removed_d} rows")

    # Step e: Filter by minimum confidence
    before_e = len(df_step_d)
    if min_confidence > 0.0:
        df_step_e = df_step_d[df_step_d["airline_sentiment_confidence"] >= min_confidence].copy()
    else:
        df_step_e = df_step_d.copy()
    rows_removed_e = before_e - len(df_step_e)
    logger.info(f"Step (e) - Low confidence (<{min_confidence}) removed: {rows_removed_e} rows")

    # Format Output Columns (Renamed with config constants)
    cleaned_df = pd.DataFrame()
    cleaned_df[OUT_TWEET_ID] = df_step_e["tweet_id"]
    cleaned_df[OUT_AIRLINE] = df_step_e["airline"]
    cleaned_df[OUT_TEXT_RAW] = df_step_e["text"]  # Untouched raw text content
    cleaned_df[OUT_LABEL] = df_step_e["temp_label"]
    cleaned_df[OUT_LABEL_ID] = df_step_e["temp_label"].map(LABEL2ID)
    cleaned_df[OUT_CONFIDENCE] = df_step_e["airline_sentiment_confidence"]
    cleaned_df[OUT_NEG_REASON] = df_step_e["negativereason"]

    # Reconcile row counts
    final_rows = len(cleaned_df)
    total_removed = rows_removed_a + rows_removed_b + rows_removed_c + rows_removed_d + rows_removed_e
    assert (
        initial_rows - total_removed == final_rows
    ), f"Row count mismatch! Initial ({initial_rows}) - Removed ({total_removed}) != Final ({final_rows})"

    removal_audit = {
        "initial_rows": initial_rows,
        "missing_empty_removed": rows_removed_a,
        "duplicate_tweet_ids_removed": rows_removed_b,
        "duplicate_texts_removed": rows_removed_c,
        "invalid_labels_removed": rows_removed_d,
        "low_confidence_filtered": rows_removed_e,
        "total_removed": total_removed,
        "final_validated_rows": final_rows,
    }

    logger.info(
        f"Reconciliation Success: {initial_rows} raw rows - {total_removed} removed = {final_rows} final validated rows."
    )
    return cleaned_df, removal_audit


def process_and_save_validated_data(
    force: bool = False, min_confidence: float = MIN_CONFIDENCE
) -> pd.DataFrame:
    """Orchestrates Phase 1 loading, validation, structural cleaning, and persistence.

    Args:
        force: Force re-processing even if validated CSV exists.
        min_confidence: Confidence threshold filter.

    Returns:
        Validated pd.DataFrame.
    """
    if VALIDATED_DATA_PATH.exists() and not force:
        logger.info(f"Validated dataset already exists at {VALIDATED_DATA_PATH}. Use --force to reprocess.")
        return pd.read_csv(VALIDATED_DATA_PATH)

    raw_df = load_raw_data(RAW_DATA_PATH)
    generate_validation_report(raw_df, save_report=True)

    cleaned_df, audit = clean_structural_issues(raw_df, min_confidence=min_confidence)

    VALIDATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(VALIDATED_DATA_PATH, index=False)
    logger.info(f"Saved validated dataset ({len(cleaned_df)} rows) to {VALIDATED_DATA_PATH}")

    # Summary Table Output
    logger.info("\n=== PHASE 1 DATASET SUMMARY TABLE ===")
    logger.info(f"Raw Dataset Rows:       {audit['initial_rows']}")
    logger.info(f"Total Rows Removed:     {audit['total_removed']}")
    logger.info(f"Validated Dataset Rows: {audit['final_validated_rows']}")
    logger.info("\n--- Final Class Distribution ---")
    dist = cleaned_df[OUT_LABEL].value_counts()
    pcts = (cleaned_df[OUT_LABEL].value_counts(normalize=True) * 100).round(2)
    summary_df = pd.DataFrame({"Count": dist, "Percentage (%)": pcts})
    logger.info("\n" + str(summary_df))

    return cleaned_df


def load_validated_data(csv_path: Path = VALIDATED_DATA_PATH) -> pd.DataFrame:
    """Importable helper function for downstream phases to load validated data.

    Args:
        csv_path: Path to tweets_validated.csv.

    Returns:
        Validated pd.DataFrame.

    Raises:
        FileNotFoundError: If validated CSV does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Validated dataset missing at {csv_path}. Run 'python -m src.data_loader' first."
        )
    return pd.read_csv(csv_path)


def main() -> None:
    """CLI handler for Phase 1 data loader execution."""
    parser = argparse.ArgumentParser(description="Load, validate, structurally clean, and save Twitter dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-processing raw dataset.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Optional minimum airline sentiment confidence filter.",
    )
    args = parser.parse_args()

    process_and_save_validated_data(force=args.force, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
