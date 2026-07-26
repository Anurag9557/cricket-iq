"""Ablations: per-phase Brier + feature-group contribution.

    python -m cricketiq.eval.ablations

(1) B1 Brier by phase (powerplay/middle/death) - where is the model confident?
(2) Feature-group ablation: full vs -momentum vs resources-only.
"""
from __future__ import annotations

import lightgbm as lgb
import polars as pl

from cricketiq.core.config import PROCESSED_DIR
from cricketiq.eval.metrics import evaluate
from cricketiq.eval.split import split_by_season

RESOURCES = ["wickets_in_hand", "balls_remaining", "required_rr"]
MOMENTUM = ["runs_last30", "wkts_last30"]
FULL = [
    "over", "innings_runs", "wickets_in_hand", "balls_remaining", "runs_needed",
    "target", "current_rr", "required_rr", "rr_diff", "runs_last30", "wkts_last30",
]


def train_predict(train, val, test, feats):
    Xtr, ytr = train.select(feats).to_numpy(), train["chase_won"].to_numpy()
    Xva, yva = val.select(feats).to_numpy(), val["chase_won"].to_numpy()
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.predict_proba(test.select(feats).to_numpy())[:, 1]


def main() -> None:
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet")
    train, val, test = split_by_season(state)
    y_te = test["chase_won"].to_numpy()

    p_full = train_predict(train, val, test, FULL)

    print("=== B1 (full) by phase - TEST 2025-26 ===")
    print(f"  {'phase':10s} {'n':>7s}   {'Brier':>6s}  {'log-loss':>8s}  {'AUC':>6s}")
    for ph in ["powerplay", "middle", "death"]:
        mask = (test["phase"] == ph).to_numpy()
        m = evaluate(y_te[mask], p_full[mask])
        print(f"  {ph:10s} {m['n']:>7d}   {m['brier']:.4f}    {m['log_loss']:.4f}   {m['auc']:.4f}")
    mo = evaluate(y_te, p_full)
    print(f"  {'overall':10s} {mo['n']:>7d}   {mo['brier']:.4f}    {mo['log_loss']:.4f}   {mo['auc']:.4f}")

    print("\n=== feature-group ablation (TEST) ===")
    p_nomo = train_predict(train, val, test, [f for f in FULL if f not in MOMENTUM])
    p_res = train_predict(train, val, test, RESOURCES)
    for name, p in [("full (11)", p_full), ("- momentum (9)", p_nomo), ("resources only (3)", p_res)]:
        m = evaluate(y_te, p)
        print(f"  {name:20s} Brier {m['brier']:.4f}   AUC {m['auc']:.4f}")


if __name__ == "__main__":
    main()