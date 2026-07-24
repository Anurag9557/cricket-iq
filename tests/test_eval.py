"""Eval harness tests: metric sanity checks (known answers) + split integrity."""
import numpy as np
import polars as pl

from cricketiq.core.config import PROCESSED_DIR
from cricketiq.eval.metrics import evaluate, expected_calibration_error
from cricketiq.eval.split import split_by_season


# ---- metrics: inputs whose answers we know by hand ----

def test_all_half_gives_brier_quarter():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = evaluate(y, np.full(len(y), 0.5))
    assert abs(m["brier"] - 0.25) < 1e-9          # (0.5 - y)^2 == 0.25 always
    assert abs(m["log_loss"] - np.log(2)) < 1e-6  # -ln(0.5) == ln 2

def test_perfect_predictions():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = evaluate(y, y.astype(float))
    assert m["brier"] < 1e-6
    assert m["auc"] == 1.0

def test_ece_zero_when_calibrated():
    y = np.array([0, 1] * 50)                       # predict 0.5, exactly 50% win
    assert expected_calibration_error(y, np.full(100, 0.5)) < 1e-9

def test_ece_detects_miscalibration():
    y = np.ones(100)                                # predict 0.5 but everyone wins
    assert abs(expected_calibration_error(y, np.full(100, 0.5)) - 0.5) < 1e-9


# ---- split: integrity ----

def test_split_no_match_overlap():
    tr, va, te = split_by_season(pl.read_parquet(PROCESSED_DIR / "state.parquet"))
    a, b, c = (set(x["match_id"].unique()) for x in (tr, va, te))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)

def test_split_covers_all_rows_by_year():
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet")
    tr, va, te = split_by_season(state)
    assert tr["season"].max() <= 2023
    assert va["season"].min() == 2024 and va["season"].max() == 2024
    assert te["season"].min() >= 2025
    assert tr.height + va.height + te.height == state.height