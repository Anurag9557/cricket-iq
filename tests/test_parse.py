"""Tests for parser output (data/processed/*.parquet).  Run: pytest tests/test_parse.py

- invariants: must hold for EVERY row if parsing is correct (no external data)
- golden matches: specific games hand-verified against ESPNcricinfo scorecards
"""
import polars as pl
import pytest

from cricketiq.core.config import PROCESSED_DIR


@pytest.fixture(scope="module")
def matches():
    return pl.read_parquet(PROCESSED_DIR / "matches.parquet")


@pytest.fixture(scope="module")
def deliveries():
    return pl.read_parquet(PROCESSED_DIR / "deliveries.parquet")


# ---------- invariants ----------

def test_innings_are_1_or_2(deliveries):
    assert set(deliveries["innings"].unique()) == {1, 2}


def test_over_in_1_to_20(deliveries):
    assert deliveries["over"].min() == 1
    assert deliveries["over"].max() == 20


def test_runs_total_is_batter_plus_extras(deliveries):
    bad = deliveries.filter(pl.col("runs_total") != pl.col("runs_batter") + pl.col("runs_extras"))
    assert bad.height == 0


def test_target_is_first_innings_plus_one(matches):
    assert matches.filter(pl.col("target") != pl.col("first_innings_runs") + 1).height == 0


def test_chase_won_agrees_with_winner(matches):
    expected = (pl.col("winner") == pl.col("team_chase")).cast(pl.Int64)
    assert matches.filter(pl.col("chase_won") != expected).height == 0


def test_chase_won_is_binary(matches):
    assert set(matches["chase_won"].unique()) <= {0, 1}


def test_legal_balls_rarely_exceed_120(deliveries):
    legal = deliveries.filter(pl.col("is_legal")).group_by("match_id", "innings").len()
    over = legal.filter(pl.col("len") > 120).height
    # A handful of real 7-ball overs (umpire miscounts) exist and are recorded faithfully.
    # A parser bug (e.g. duplicated deliveries) would push this ratio far higher, ~50%.
    assert over / legal.height < 0.01, f"{over}/{legal.height} innings exceed 120 legal balls"

def test_matches_and_deliveries_agree_on_first_innings(matches, deliveries):
    inn1 = (deliveries.filter(pl.col("innings") == 1)
            .group_by("match_id").agg(pl.col("runs_total").sum().alias("runs")))
    joined = matches.join(inn1, on="match_id", how="left")
    assert joined.filter(pl.col("first_innings_runs") != pl.col("runs")).height == 0


# ---------- golden matches (verify each on espncricinfo.com before trusting) ----------

GOLDEN = [
    # (match_id, first_innings_runs, winner, chase_won)
    ("1535465", 155, "Royal Challengers Bengaluru", 1),  # GT v RCB, 2026-05-31 — VERIFY 155
    # add 2 matches you know cold, with scores checked on ESPNcricinfo
]


@pytest.mark.parametrize("mid,fi_runs,winner,chase_won", GOLDEN)
def test_golden_match(matches, mid, fi_runs, winner, chase_won):
    row = matches.filter(pl.col("match_id") == mid)
    assert row.height == 1, f"{mid} not found (was it dropped?)"
    r = row.row(0, named=True)
    assert r["first_innings_runs"] == fi_runs
    assert r["winner"] == winner
    assert r["chase_won"] == chase_won