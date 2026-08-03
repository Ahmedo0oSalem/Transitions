"""Logging helpers for TRANSITIONS."""

from __future__ import annotations

import logging


def get_logger(name: str = "TRANSITIONS") -> logging.Logger:
    """Return a logger configured for package use."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
