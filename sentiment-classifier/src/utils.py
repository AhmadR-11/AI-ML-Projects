"""Utility functions for logging, random seed control, metric saving/loading, timing, and figure saving."""

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import FIGURE_DPI, FIGURES_DIR, LOGS_DIR, SEED


def get_logger(name: str = "sentiment_classifier") -> logging.Logger:
    """Configures and returns a shared logger writing to standard console and logs/project.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream Handler (Console)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (logs/project.log)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "project.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int = SEED) -> None:
    """Sets global random seed across Python random, hash seed, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Alias for backward compatibility
set_seed = set_global_seed


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Saves dictionary metrics/data to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    get_logger().info(f"Saved JSON data to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Loads dictionary data from a JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Alias for backward compatibility
save_json_metrics = save_json


@contextmanager
def timer(name: str = "Block") -> Generator[None, None, None]:
    """Context manager measuring and logging execution time of a code block."""
    start_time = time.perf_counter()
    logger = get_logger("timer")
    logger.info(f"Started '{name}'...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"Finished '{name}' in {elapsed:.4f} seconds.")


def save_figure(fig: plt.Figure, filepath: Path, dpi: int = FIGURE_DPI) -> None:
    """Saves a matplotlib figure into FIGURES_DIR or target path at specified resolution."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    get_logger().info(f"Saved figure to {filepath}")
