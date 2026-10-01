"""Central configuration module for sentiment classifier project paths, constants, and parameters."""

from pathlib import Path
from typing import Dict, List

# Project Root Directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Path Constants
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "Tweets.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VALIDATED_DATA_PATH: Path = PROCESSED_DIR / "tweets_validated.csv"
EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
LOGS_DIR: Path = PROJECT_ROOT / "logs"


# Ensure all essential project directories exist
def ensure_dirs() -> None:
    """Creates all required output and artifact directories if missing."""
    for directory in [
        DATA_DIR,
        PROCESSED_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# Run ensure_dirs at module load time
ensure_dirs()

# Experiment & Pipeline Constants
SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# Target Label Mappings
LABELS: List[str] = ["negative", "neutral", "positive"]
LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

# Model Architecture Constants
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# Dataset Column Name Constants
TEXT_RAW: str = "text"
TEXT_LIGHT: str = "clean_text"
TEXT_FULL: str = "clean_text_full"
LABEL: str = "airline_sentiment"
LABEL_ID: str = "label"
CONFIDENCE: str = "airline_sentiment_confidence"
AIRLINE: str = "airline"
TWEET_ID: str = "tweet_id"
NEG_REASON: str = "negativereason"

# Output Schema Column Names
OUT_TWEET_ID: str = "tweet_id"
OUT_AIRLINE: str = "airline"
OUT_TEXT_RAW: str = "text_raw"
OUT_LABEL: str = "label"
OUT_LABEL_ID: str = "label_id"
OUT_CONFIDENCE: str = "confidence"
OUT_NEG_REASON: str = "negativereason"

# Dataset Filtering Constants
MIN_CONFIDENCE: float = 0.0

# Evaluation Metric Constant
PRIMARY_METRIC: str = "f1_macro"

# Backward Compatibility Aliases
RANDOM_SEED: int = SEED
CLASS_MAPPING: Dict[str, int] = LABEL2ID
REVERSE_CLASS_MAPPING: Dict[int, str] = ID2LABEL
CLASS_NAMES: List[str] = LABELS
PROCESSED_DATA_DIR: Path = PROCESSED_DIR
TRAIN_DATA_PATH: Path = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DIR / "test.csv"
TRAIN_EMBEDDINGS_PATH: Path = EMBEDDINGS_DIR / "train_embeddings.npy"
TEST_EMBEDDINGS_PATH: Path = EMBEDDINGS_DIR / "test_embeddings.npy"
TFIDF_VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
BEST_TFIDF_MODEL_PATH: Path = MODELS_DIR / "best_tfidf_model.joblib"
BEST_EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "best_embedding_model.joblib"
PYTORCH_EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "best_embedding_mlp.pt"
FIGURE_DPI: int = 150
