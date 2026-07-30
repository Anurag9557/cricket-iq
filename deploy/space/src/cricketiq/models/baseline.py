"""B0 - resource baseline: logistic regression on 3 core features.

    python -m cricketiq.models.baseline

Trains on <=2023, evaluates on 2025-26 (test). The first real number on the board.
Features: wickets_in_hand, balls_remaining, required_rr (the classic 'resources').
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cricketiq.core.config import PROCESSED_DIR
from cricketiq.eval.metrics import evaluate
from cricketiq.eval.split import split_by_season

FEATURES = ["wickets_in_hand", "balls_remaining", "required_rr"]


def load_xy(df: pl.DataFrame):
    return df.select(FEATURES).to_numpy(), df["chase_won"].to_numpy()


def main() -> None:
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet")
    train, val, test = split_by_season(state)
    print(f"rows -> train {train.height} | val {val.height} | test {test.height}")

    X_tr, y_tr = load_xy(train)
    X_te, y_te = load_xy(test)

    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]

    m = evaluate(y_te, p_te)
    base_rate = float(y_tr.mean())
    brier_base = float(np.mean((y_te - base_rate) ** 2))

    print("\n=== B0 logistic (wkts_in_hand, balls_remaining, required_rr) - TEST 2025-26 ===")
    print(f"  Brier     {m['brier']:.4f}   (no-skill base rate = {brier_base:.4f})")
    print(f"  log-loss  {m['log_loss']:.4f}")
    print(f"  AUC       {m['auc']:.4f}")
    print(f"  ECE       {m['ece']:.4f}")
    print(f"  n         {m['n']}")
    coefs = model.named_steps["lr"].coef_[0]
    print(f"\n  scaled coefs: {dict(zip(FEATURES, coefs.round(3)))}")


if __name__ == "__main__":
    main()