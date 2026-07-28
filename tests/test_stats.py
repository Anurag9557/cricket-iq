"""
Invariant tests for the stat tools (Phase 5.1).

These check internal CONSISTENCY (they don't hardcode a player's numbers, since those
depend on your data). For CORRECTNESS against reality, hand-verify one famous player
against ESPNcricinfo — see the note at the bottom. The invariant tests catch logic
bugs; the manual spot-check confirms the logic matches real cricket.
"""
import polars as pl

from cricketiq.core import config
from cricketiq.agent.tools import stats


def _busiest(col: str) -> str:
    """The id that appears in the most deliveries — a safe, data-independent subject."""
    d = pl.read_parquet(config.PROCESSED_DIR / "deliveries.parquet")
    return d.group_by(col).len().sort("len", descending=True)[col][0]


def test_batter_stats_consistent():
    s = stats.batter_stats(_busiest("batter"))
    assert s["balls"] > 0 and s["runs"] >= 0
    assert s["n"] == s["balls"]                       # n IS the sample size
    assert abs(s["strike_rate"] - round(s["runs"] * 100 / s["balls"], 2)) < 0.01
    if s["dismissals"] > 0:
        assert abs(s["average"] - round(s["runs"] / s["dismissals"], 2)) < 0.01


def test_bowler_economy_consistent():
    s = stats.bowler_stats(_busiest("bowler"))
    assert s["balls"] > 0 and s["wickets"] >= 0
    assert abs(s["economy"] - round(s["runs"] * 6 / s["balls"], 2)) < 0.01
    assert 0 <= s["dot_pct"] <= 100


def test_phase_split_covers_all_balls():
    """powerplay + middle + death balls must equal the total — validates phase logic."""
    bid = _busiest("batter")
    total = stats.batter_stats(bid)["balls"]
    parts = sum(stats.batter_stats(bid, phase=p)["balls"] for p in ("powerplay", "middle", "death"))
    assert parts == total


def test_matchup_is_subset():
    """A matchup can't have more balls or runs than the batter's total."""
    bid, bowid = _busiest("batter"), _busiest("bowler")
    mu = stats.matchup(bid, bowid)
    full = stats.batter_stats(bid)
    assert mu["balls"] <= full["balls"]
    assert mu["runs"] <= full["runs"]


def test_empty_query_is_safe():
    """A nonsense id returns zeros and n=0, not a crash or a divide-by-zero."""
    s = stats.batter_stats("__no_such_player__")
    assert s["balls"] == 0 and s["runs"] == 0 and s["n"] == 0
    assert s["strike_rate"] is None and s["average"] is None


# --- manual correctness check (run once, compare to ESPNcricinfo) -------------------
# from cricketiq.agent.tools import stats
# print(stats.batter_stats(stats.pid("Virat Kohli")))     # totals across your leagues
# print(stats.bowler_stats(stats.pid("Jasprit Bumrah"), phase="death"))