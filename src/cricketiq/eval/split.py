"""Temporal train/val/test split for the win-probability model.

By SEASON — and since season is constant within a match, that's a split by MATCH:
every ball of a match lands in exactly one split. NEVER shuffle rows (that scatters
a match across splits = leakage). See docs/leakage-audit.md.
"""
from __future__ import annotations

import polars as pl

from cricketiq.core.config import TRAIN_END_YEAR, VAL_YEAR, TEST_START_YEAR


def split_by_season(state: pl.DataFrame):
    """Return (train, val, test) frames: <=2023 / ==2024 / >=2025."""
    train = state.filter(pl.col("season") <= TRAIN_END_YEAR)
    val = state.filter(pl.col("season") == VAL_YEAR)
    test = state.filter(pl.col("season") >= TEST_START_YEAR)
    return train, val, test