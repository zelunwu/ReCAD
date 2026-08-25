"""Small, side-effect-free IO helpers."""

from __future__ import annotations

import json
from pathlib import Path


def touch_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if missing; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj, path: str | Path) -> None:
    """Write a JSON-serializable object with stable formatting."""
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_json(path: str | Path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
