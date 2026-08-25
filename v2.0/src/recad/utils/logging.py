"""Logging configuration for ReCAD v2.0.

A single, standardised logger named ``recad`` is used across the package.
Reproducible run logs should be handled by the CLI (tee to file), while this
module only controls formatting.
"""

from __future__ import annotations

import logging
import sys

_RECAD_LOGGER_NAME = "recad"
_CONFIGURED = False


def get_logger(name: str = _RECAD_LOGGER_NAME) -> logging.Logger:
    """Return the (lazily configured) package logger."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger
