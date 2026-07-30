"""
Serve-layer explainability — why the model gave this win probability.

Uses LightGBM's built-in TreeSHAP (`booster_.predict(X, pred_contrib=True)`): exact
per-feature contributions, no extra `shap` dependency. Contributions are in log-odds
space, so we don't dress them up as "+X%"; we rank features by |contribution| and
report the direction each one pushes the chase's win probability.
"""
from __future__ import annotations

import joblib
import numpy as np
import polars as pl

from cricketiq.core import config
from cricketiq.models.gbm import FEATURES

# Human-readable labels for the "Why?" panel.
LABELS = {
    "over": "overs bowled",
    "innings_runs": "runs scored",
    "wickets_in_hand": "wickets in hand",
    "balls_remaining": "balls remaining",
    "runs_needed": "runs needed",
    "target": "target",
    "current_rr": "current run rate",
    "required_rr": "required run rate",
    "rr_diff": "rate gap (current − required)",
    "runs_last30": "runs, last 30 balls",
    "wkts_last30": "wickets, last 30 balls",
}

_model = None
_state = None


def _load_model():
    global _model
    if _model is None:
        path = config.PROCESSED_DIR / "b1.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run models.gbm to save the model.")
        _model = joblib.load(path)


def _load_state():
    global _state
    if _state is None:
        _state = pl.read_parquet(config.PROCESSED_DIR / "state.parquet")


def explain_features(values: dict, top_k: int = 5) -> dict:
    """Explain an arbitrary state given all 11 model features as {feature: number}."""
    _load_model()
    X = np.array([[float(values[f]) for f in FEATURES]], dtype=float)
    prob = float(_model.predict_proba(X)[0, 1])
    # TreeSHAP: shape (1, n_features + 1); the trailing column is the base value.
    contrib = _model.booster_.predict(X, pred_contrib=True)[0]

    drivers = [
        {
            "feature": f,
            "label": LABELS[f],
            "value": round(float(values[f]), 2),
            "contribution": round(float(c), 4),   # log-odds; sign = direction
            "direction": "up" if c > 0 else "down",
        }
        for f, c in zip(FEATURES, contrib[:-1])
    ]
    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return {"win_prob": round(prob, 4), "drivers": drivers[:top_k]}


def explain_ball(match_id: str, ball_seq: int, top_k: int = 5) -> dict | None:
    """Explain a real delivery by looking up its feature row in state.parquet."""
    _load_state()
    row = _state.filter(
        (pl.col("match_id") == match_id) & (pl.col("ball_seq") == ball_seq)
    ).select(FEATURES)
    if row.height == 0:
        return None
    result = explain_features(row.to_dicts()[0], top_k)
    return {"match_id": match_id, "ball_seq": ball_seq, **result}