"""Reproducible seeding for PyTorch, NumPy and the Python RNG."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed all stochastic components with ``seed`` (v1.1 convention: 100)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import: keeps this module importable without torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed: nothing more to seed
        pass
