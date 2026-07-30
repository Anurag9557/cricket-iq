"""
Serve-layer data access (Phase 4.2).

Loads the precomputed artifacts ONCE and hands the API plain Python structures.
No model, no training, no inference here — this layer only reads files. Keeping it
separate from the routes means the API never touches parquet or polars directly.
"""
from __future__ import annotations

import polars as pl

from cricketiq.core import config

# A delivery is a "key moment" when win-prob swings at least this much (Phase 4.3).
KEY_MOMENT_THRESHOLD = 0.08

# Static match facts surfaced to the UI (from matches.parquet).
_META_COLS = [
    "match_id", "league", "season", "date", "venue", "city",
    "team_bat_first", "team_chase", "winner", "win_by", "win_margin", "target",
]

_timelines: pl.DataFrame | None = None
_matches: pl.DataFrame | None = None


def load() -> None:
    """Read timelines + match metadata into memory. Called once at API startup."""
    global _timelines, _matches
    if _timelines is not None:
        return
    tl_path = config.PROCESSED_DIR / "timelines.parquet"
    mt_path = config.PROCESSED_DIR / "matches.parquet"
    if not tl_path.exists():
        raise FileNotFoundError(f"{tl_path} missing — run build_timelines first.")
    _timelines = pl.read_parquet(tl_path)
    _matches = pl.read_parquet(mt_path).select(_META_COLS)


def _tl() -> pl.DataFrame:
    if _timelines is None:
        load()
    return _timelines


def _mt() -> pl.DataFrame:
    if _matches is None:
        load()
    return _matches


def match_count() -> int:
    return _tl()["match_id"].n_unique()


def list_matches(league: str | None = None, limit: int = 100) -> list[dict]:
    """Replayable matches (those that have a timeline), newest first, with metadata."""
    meta = _tl().select("match_id").unique().join(_mt(), on="match_id", how="left")
    if league:
        meta = meta.filter(pl.col("league") == league)
    return meta.sort("date", descending=True).head(limit).to_dicts()


def get_meta(match_id: str) -> dict | None:
    row = _mt().filter(pl.col("match_id") == match_id)
    return row.to_dicts()[0] if row.height else None

# ---- precomputed verified commentary (Phase 6.1) -------------------------------
_COMMENTARY: dict[str, list[dict]] | None = None

def _load_commentary() -> dict[str, list[dict]]:
    global _COMMENTARY
    if _COMMENTARY is None:
        path = config.PROCESSED_DIR / "commentary.parquet"
        by_match: dict[str, list[dict]] = {}
        if path.exists():
            for row in pl.read_parquet(path).iter_rows(named=True):
                by_match.setdefault(str(row["match_id"]), []).append(row)
            for cards in by_match.values():
                cards.sort(key=lambda c: c["ball_seq"])
        _COMMENTARY = by_match
    return _COMMENTARY

def get_commentary(match_id: str) -> list[dict]:
    """Precomputed verified key-moment cards for one match, sorted by ball_seq."""
    return _load_commentary().get(str(match_id), [])


def get_timeline(match_id: str) -> list[dict]:
    """Every delivery of one match, in order, with a cricket-style over.ball label
    and a key-moment flag. Returns [] if the match has no timeline."""
    rows = _tl().filter(pl.col("match_id") == match_id).sort("ball_seq").to_dicts()

    out: list[dict] = []
    cur_over, ball_in_over = None, 0
    for r in rows:
        # over is 1-indexed (1..20); cricket writes the *completed* over, so over-1.
        # ball_in_over counts every delivery in the over (extras included) for v1.
        if r["over"] != cur_over:
            cur_over, ball_in_over = r["over"], 0
        ball_in_over += 1

        out.append({
            "ball_seq": r["ball_seq"],
            "display_ball": f"{r['over'] - 1}.{ball_in_over}",
            "over": r["over"],
            "score": r["innings_runs"],
            "wickets": r["wickets_lost"],
            "balls_remaining": r["balls_remaining"],
            "runs_needed": r["runs_needed"],
            "target": r["target"],
            "current_rr": r["current_rr"],
            "required_rr": r["required_rr"],
            "win_prob": r["win_prob"],
            "wp_delta": r["wp_delta"],
            "runs_this_ball": r["runs_this_ball"],
            "wicket_fell": r["wicket_fell"],
            "is_key_moment": abs(r["wp_delta"]) >= KEY_MOMENT_THRESHOLD,
        })
    return out