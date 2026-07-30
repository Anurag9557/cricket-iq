"""Calibration comparison (raw vs isotonic vs Platt) of B1 + reliability diagram.

    python -m cricketiq.models.calibrate

Each calibrator is FIT on val (2024), applied to test (2025-26). We KEEP whichever
wins on held-out ECE/Brier — an already-calibrated model may not benefit at all.
"""
from __future__ import annotations

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cricketiq.core.config import PROCESSED_DIR, ROOT
from cricketiq.eval.metrics import evaluate
from cricketiq.eval.split import split_by_season
from cricketiq.models.gbm import load_xy


def reliability_points(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    ids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            xs.append(y_prob[mask].mean())
            ys.append(y_true[mask].mean())
    return xs, ys


def main() -> None:
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet")
    train, val, test = split_by_season(state)
    X_tr, y_tr = load_xy(train)
    X_va, y_va = load_xy(val)
    X_te, y_te = load_xy(test)

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    p_va = model.predict_proba(X_va)[:, 1]
    p_raw = model.predict_proba(X_te)[:, 1]

    p_iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va).predict(p_raw)
    platt = LogisticRegression().fit(p_va.reshape(-1, 1), y_va)
    p_platt = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    preds = {"raw B1": p_raw, "isotonic": p_iso, "Platt": p_platt}
    print("             Brier    log-loss   ECE")
    for name, p in preds.items():
        m = evaluate(y_te, p)
        print(f"  {name:9s}  {m['brier']:.4f}   {m['log_loss']:.4f}   {m['ece']:.4f}")

    colors = {"raw B1": "#eb6834", "isotonic": "#2a78d6", "Platt": "#008300"}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#898781", label="perfect calibration")
    for name, p in preds.items():
        x, y = reliability_points(y_te, p)
        ax.plot(x, y, "o-", color=colors[name], label=f"{name} (ECE {evaluate(y_te, p)['ece']:.3f})")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Reliability - B1 win probability (test 2025-26)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, color="#e1e0d9", alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = ROOT / "docs" / "reliability.png"
    fig.savefig(out, dpi=130)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()