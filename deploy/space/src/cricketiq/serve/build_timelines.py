"""
Phase 4.1 — precompute win-probability timelines.

Loads the SAVED B1 model (data/processed/b1.pkl — the exact model that produced the
docs/results.md ladder, persisted by models/gbm.py) and scores every delivery of
every match, writing per-match win-prob curves to data/processed/timelines.parquet.
This is the cheap, unbreakable source the live match-replay demo streams from: no
model training OR inference happens at request time — the API just reads this file.

Design: this file holds only the PER-BALL DYNAMIC data (score, wickets, rates,
runs-this-ball, wicket flag, win_prob and its delta). STATIC match facts (teams,
venue, date, winner, target) already live in matches.parquet; the API joins them by
match_id, so we never duplicate team names across ~782k rows.

Honesty note: predictions on train-era (<=2023) matches are in-sample. That's fine
for an illustrative replay ticker; every EVALUATION number in docs/results.md stays
strictly out-of-sample (test 2025-26).
"""
from __future__ import annotations

import joblib
import polars as pl

from cricketiq.core import config
from cricketiq.models.gbm import load_xy  # single source of truth for the feature contract

# Per-ball dynamic columns kept from state.parquet for the ticker + scoreboard.
# legal_bowled stays as a FACT (the chart's x-axis + the source for the API's
# display_ball label); the formatted "15.3" string is built in the serve layer.
KEEP = [
    "match_id", "season", "ball_seq", "over", "legal_bowled",
    "innings_runs", "wickets_lost", "wickets_in_hand",
    "balls_remaining", "runs_needed", "target",
    "current_rr", "required_rr",
]


def main() -> None:
    model_path = config.PROCESSED_DIR / "b1.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"{model_path} not found.\n"
            "Run `python -m cricketiq.models.gbm` first — it trains B1 and now saves "
            "the frozen model that the timelines and the API both load."
        )
    model = joblib.load(model_path)

    state = pl.read_parquet(config.PROCESSED_DIR / "state.parquet")
    X_all, _ = load_xy(state)
    win_prob = model.predict_proba(X_all)[:, 1]

    timelines = (
        state.select(KEEP)
        .with_columns(pl.Series("win_prob", win_prob.round(4)))
        .sort(["match_id", "ball_seq"])
        .with_columns(
            # per-ball events + win-prob change, all computed WITHIN each match
            runs_this_ball=pl.col("innings_runs").diff().over("match_id")
                .fill_null(pl.col("innings_runs")),
            wicket_fell=(
                pl.col("wickets_lost").diff().over("match_id")
                .fill_null(pl.col("wickets_lost")) > 0
            ),
            wp_delta=pl.col("win_prob").diff().over("match_id").fill_null(0.0).round(4),
        )
    )

    # fail fast on the two things that would silently corrupt the demo:
    assert timelines.height == state.height, "row count changed — a match got dropped/duplicated"
    assert timelines["runs_this_ball"].min() >= 0, (
        "negative runs_this_ball → deliveries out of order within a match"
    )

    out = config.PROCESSED_DIR / "timelines.parquet"
    timelines.write_parquet(out, compression="zstd")

    n_matches = timelines["match_id"].n_unique()
    print(f"saved {out}")
    print(f"  {timelines.height:,} deliveries across {n_matches:,} matches")
    print(f"  win_prob range [{timelines['win_prob'].min():.3f}, {timelines['win_prob'].max():.3f}]")
    print(f"  biggest single-ball win-prob swing: {timelines['wp_delta'].abs().max():.3f}")
    print("\nfirst 3 rows (opening balls — win_prob should be mid-range):")
    print(timelines.head(3))
    print("\nlast 3 rows (final balls — win_prob should be near 0 or 1):")
    print(timelines.tail(3))


if __name__ == "__main__":
    main()