"""Utility helpers for MovieLens paper-style stage-1 bandit experiments."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may be unavailable in some environments
    torch = None


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file using UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)