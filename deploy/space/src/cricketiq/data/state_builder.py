"""Build per-ball chase-state features for the win-probability model.

    python -m cricketiq.data.state_builder

Reads   data/processed/deliveries.parquet, matches.parquet
Writes  data/processed/state.parquet   (one row per 2nd-innings delivery)

Each row = match state AFTER that delivery (cumulative through it) + label.
CORE features only (leak-free: within-innings + pre-chase target). Player-form /
venue-par enrichment comes later with as-of-date joins (see docs/leakage-audit.md).
"""
from __future__ import annotations

import polars as pl

from cricketiq.core.config import PROCESSED_DIR

MOMENTUM_WINDOW = 30        # deliveries (~5 overs)
REQUIRED_RR_CAP = 36.0      # 6 sixes/over. Phase-2 ablation: try 36 / 60 / 999(=uncapped)


def build() -> None:
    deliveries = pl.read_parquet(PROCESSED_DIR / "deliveries.parquet")
    matches = pl.read_parquet(PROCESSED_DIR / "matches.parquet")

    chase = (
        deliveries.filter(pl.col("innings") == 2)
        .join(
            matches.select("match_id", "season", "target", "chase_won"),
            on="match_id",
            how="left",
        )
        .sort("match_id", "ball_seq")          # order matters for cumulative sums
    )

    # cumulative state THROUGH each delivery (= state AFTER this ball)
    chase = chase.with_columns(
        innings_runs=pl.col("runs_total").cum_sum().over("match_id"),
        wickets_lost=pl.col("is_wicket").cast(pl.Int32).cum_sum().over("match_id"),
        legal_bowled=pl.col("is_legal").cast(pl.Int32).cum_sum().over("match_id"),
    )

    # derived state (clamp handles the real 7-ball-over quirk)
    chase = chase.with_columns(
        balls_remaining=(120 - pl.col("legal_bowled")).clip(lower_bound=0),
        wickets_in_hand=10 - pl.col("wickets_lost"),
        runs_needed=(pl.col("target") - pl.col("innings_runs")).clip(lower_bound=0),
    )

    # rates (guarded), phase, momentum
    chase = chase.with_columns(
        current_rr=pl.when(pl.col("legal_bowled") > 0)
            .then(pl.col("innings_runs") * 6 / pl.col("legal_bowled"))
            .otherwise(0.0),
        required_rr=pl.when(pl.col("runs_needed") == 0)                 # target reached -> need nothing
            .then(0.0)
            .when(pl.col("balls_remaining") > 0)                        # cap the death-overs explosion
            .then((pl.col("runs_needed") * 6 / pl.col("balls_remaining")).clip(upper_bound=REQUIRED_RR_CAP))
            .otherwise(REQUIRED_RR_CAP),                                # runs needed, no balls -> impossible
        phase=pl.when(pl.col("over") <= 6).then(pl.lit("powerplay"))
            .when(pl.col("over") <= 15).then(pl.lit("middle"))
            .otherwise(pl.lit("death")),
        runs_last30=pl.col("innings_runs")
            - pl.coalesce(pl.col("innings_runs").shift(MOMENTUM_WINDOW).over("match_id"), pl.lit(0)),
        wkts_last30=pl.col("wickets_lost")
            - pl.coalesce(pl.col("wickets_lost").shift(MOMENTUM_WINDOW).over("match_id"), pl.lit(0)),
    )

    chase = chase.with_columns(rr_diff=pl.col("required_rr") - pl.col("current_rr"))

    out = chase.select(
        "match_id", "season", "ball_seq", "over", "phase",
        "innings_runs", "wickets_lost", "wickets_in_hand",
        "legal_bowled", "balls_remaining", "runs_needed", "target",
        "current_rr", "required_rr", "rr_diff", "runs_last30", "wkts_last30",
        "chase_won",
    )
    out.write_parquet(PROCESSED_DIR / "state.parquet")

    print(f"state.parquet: {out.height} rows x {out.width} cols ({out['match_id'].n_unique()} matches)")
    print(out.head(8))
    print("\nrow-level label balance:", out.group_by("chase_won").len().sort("chase_won").to_dicts())


if __name__ == "__main__":
    build()