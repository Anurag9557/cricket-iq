"""
Win-probability moment tools (Phase 6) — the bridge from the model to the agent.

`stats.py` answers raw questions from the ball-by-ball data (batting, bowling, matchups).
This module answers questions about the MODEL's output: how much the win probability SWUNG
on a given ball, and which deliveries swung a match the most. Every value is read straight
from `timelines.parquet` (precomputed in Phase 4 by `serve/build_timelines.py`) — this module
computes NOTHING itself, it looks up the precomputed win_prob / wp_delta.

Why a tool and not LLM arithmetic: a win-probability swing is a number, and the verifier
rejects any number the model computes on its own (exactly what the stress test caught). Reading
the swing from here means the agent can state it and have it pass verification — which is what
makes verified auto-commentary (§6.1) possible at all.

A delivery is addressed by (match_id, ball_seq) — ball_seq is the 0-indexed delivery position
in the second innings, the same key `serve/data.py` and the `/winprob` endpoint use. `match_id`
is compared as a string so an int-or-string column both resolve.
"""
from __future__ import annotations

import polars as pl

from cricketiq.core import config

# matches serve/data.py: a swing of >= 8 percentage points is a "key moment"
KEY_MOMENT_THRESHOLD = 0.08

_timelines: pl.DataFrame | None = None


def _tl() -> pl.DataFrame:
    global _timelines
    if _timelines is None:
        path = config.PROCESSED_DIR / "timelines.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it with serve/build_timelines.py (Phase 4)."
            )
        _timelines = pl.read_parquet(path)
    return _timelines


def wp_delta(match_id: str, ball_seq: int) -> dict:
    """The win-probability swing on ONE delivery, read from the precomputed timeline.
    `wp_delta` is signed (positive = swung toward the chasing side); `win_prob` is the chasing
    side's probability AFTER the ball. Returns found=False if that ball isn't in the data."""
    row = _tl().filter(
        (pl.col("match_id").cast(pl.Utf8) == str(match_id)) & (pl.col("ball_seq") == int(ball_seq))
    )
    if row.height == 0:
        return {"found": False, "match_id": match_id, "ball_seq": ball_seq}
    r = row.row(0, named=True)
    return {
        "found": True, "match_id": match_id, "ball_seq": int(ball_seq), "over": r["over"],
        "win_prob": r["win_prob"], "wp_delta": r["wp_delta"],
        "runs_this_ball": r["runs_this_ball"], "wicket_fell": r["wicket_fell"],
        "is_key_moment": abs(r["wp_delta"]) >= KEY_MOMENT_THRESHOLD,
    }


def key_moments(match_id: str, top_n: int = 5) -> dict:
    """The biggest win-probability swings in a match, largest first — the deliveries a
    commentator would call. Reads the precomputed wp_delta; nothing is recomputed."""
    tl = _tl().filter(pl.col("match_id").cast(pl.Utf8) == str(match_id))
    if tl.height == 0:
        return {"found": False, "match_id": match_id, "moments": [],
                "n_deliveries": 0, "n_key_moments": 0}
    key = tl.filter(pl.col("wp_delta").abs() >= KEY_MOMENT_THRESHOLD).with_columns(
        _absd=pl.col("wp_delta").abs()
    )
    top = key.sort("_absd", descending=True).head(top_n)
    moments = [
        {"ball_seq": m["ball_seq"], "over": m["over"], "wp_delta": m["wp_delta"],
         "win_prob": m["win_prob"], "runs_this_ball": m["runs_this_ball"],
         "wicket_fell": m["wicket_fell"]}
        for m in top.iter_rows(named=True)
    ]
    return {"found": True, "match_id": match_id, "moments": moments,
            "n_deliveries": int(tl.height), "n_key_moments": int(key.height)}