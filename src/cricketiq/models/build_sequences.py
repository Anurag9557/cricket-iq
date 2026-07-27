"""Build padded per-match chase sequences for the M2 sequence model.

    python -m cricketiq.models.build_sequences

state.parquet -> data/processed/sequences.npz (load on Kaggle with np.load):
  X       [N, T, F] float32  padded per-ball feature sequences
  mask    [N, T]    bool      True = real ball, False = pad
  y       [N]       int8      chase_won (one label per match)
  season  [N]       int16     for the temporal split
plus feature_names, match_ids.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from cricketiq.core.config import PROCESSED_DIR

FEATURES = [
    "over", "innings_runs", "wickets_in_hand", "balls_remaining", "runs_needed",
    "target", "current_rr", "required_rr", "rr_diff", "runs_last30", "wkts_last30",
]


def main() -> None:
    state = pl.read_parquet(PROCESSED_DIR / "state.parquet").sort("match_id", "ball_seq")
    groups = list(state.group_by("match_id", maintain_order=True))

    N = len(groups)
    T = max(g.height for _, g in groups)
    F = len(FEATURES)
    print(f"{N} matches, T_max={T}, F={F}")

    X = np.zeros((N, T, F), dtype=np.float32)
    mask = np.zeros((N, T), dtype=bool)
    y = np.zeros(N, dtype=np.int8)
    season = np.zeros(N, dtype=np.int16)
    match_ids = []

    for i, (key, grp) in enumerate(groups):
        grp = grp.sort("ball_seq")
        feats = grp.select(FEATURES).to_numpy().astype(np.float32)
        t = feats.shape[0]
        X[i, :t] = feats
        mask[i, :t] = True
        y[i] = grp["chase_won"][0]
        season[i] = grp["season"][0]
        match_ids.append(key[0] if isinstance(key, tuple) else key)

    out = PROCESSED_DIR / "sequences.npz"
    np.savez_compressed(
        out, X=X, mask=mask, y=y, season=season,
        feature_names=np.array(FEATURES), match_ids=np.array(match_ids),
    )
    print(f"saved {out}")
    print(f"  X {X.shape}  mask {mask.shape}  y {y.shape}")
    print(f"  split: train<=2023 {int((season<=2023).sum())} | "
          f"val 2024 {int((season==2024).sum())} | test>=2025 {int((season>=2025).sum())}")
    print(f"  labels: won {int(y.sum())} / lost {int((y==0).sum())}")


if __name__ == "__main__":
    main()