"""B1 - LightGBM on the full core-feature set.

    python -m cricketiq.models.gbm

Trains on <=2023, early-stops on 2024 (val), evaluates on 2025-26 (test).
Drops features that are exact functions of others (wickets_lost, legal_bowled,
phase) since trees recover them from what's kept.
"""
from __future__ import annotations

import lightgbm as lgb
import polars as pl

from cricketiq.core.config import PROCESSED_DIR
from cricketiq.eval.metrics import evaluate
from cricketiq.eval.split import split_by_season

FEATURES = [
    "over", "innings_runs", "wickets_in_hand", "balls_remaining", "runs_needed",
    "target", "current_rr", "required_rr", "rr_diff", "runs_last30", "wkts_last30",
]


def load_xy(df):
    return df.select(FEATURES).to_numpy(), df["chase_won"].to_numpy()


def main() -> None:
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet")
    train, val, test = split_by_season(state)

    X_tr, y_tr = load_xy(train)
    X_va, y_va = load_xy(val)
    X_te, y_te = load_xy(test)

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        importance_type="gain", random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_te = model.predict_proba(X_te)[:, 1]

    m = evaluate(y_te, p_te)
    print(f"\n=== B1 LightGBM ({len(FEATURES)} features) - TEST 2025-26 ===")
    print(f"  Brier     {m['brier']:.4f}   (B0 was 0.1204)")
    print(f"  log-loss  {m['log_loss']:.4f}   (B0 was 0.3717)")
    print(f"  AUC       {m['auc']:.4f}   (B0 was 0.9147)")
    print(f"  ECE       {m['ece']:.4f}   (B0 was 0.0336)")
    print(f"  best_iteration {model.best_iteration_}")

    imp = model.feature_importances_
    imp = 100 * imp / imp.sum()
    print("\n  feature importance (% of gain):")
    for f, v in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
        print(f"    {f:16s} {v:5.1f}%")


if __name__ == "__main__":
    main()