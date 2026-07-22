"""Tests for state.parquet (per-ball chase features).  Run: pytest tests/test_state.py

Behavioral invariants (catch real bugs) + cross-checks that tie this table back to
the parser outputs via an independent path. No tautological formula-restatements.
"""
import polars as pl
import pytest

from cricketiq.core.config import PROCESSED_DIR

REQUIRED_RR_CAP = 36.0


@pytest.fixture(scope="module")
def state():
    return pl.read_parquet(PROCESSED_DIR / "state.parquet")


@pytest.fixture(scope="module")
def matches():
    return pl.read_parquet(PROCESSED_DIR / "matches.parquet")


def _final_rows(state):
    return (state.with_columns(_mx=pl.col("ball_seq").max().over("match_id"))
            .filter(pl.col("ball_seq") == pl.col("_mx")).drop("_mx"))


def _step(state, col):
    return (state.sort("match_id", "ball_seq")
            .with_columns(step=pl.col(col) - pl.col(col).shift(1).over("match_id"))
            .drop_nulls("step")["step"])


# ---------- ranges ----------

def test_balls_remaining_range(state):
    assert state["balls_remaining"].min() >= 0 and state["balls_remaining"].max() <= 120


def test_wickets_in_hand_range(state):
    assert state["wickets_in_hand"].min() >= 0 and state["wickets_in_hand"].max() <= 10


def test_required_rr_capped(state):
    assert 0 <= state["required_rr"].min() and state["required_rr"].max() <= REQUIRED_RR_CAP


def test_chase_won_binary(state):
    assert set(state["chase_won"].unique()) <= {0, 1}


# ---------- behavioral (within-match progression) ----------

def test_legal_bowled_steps_by_0_or_1(state):
    # balls_remaining decreases IFF the delivery is legal -> legal_bowled steps 0 or 1
    s = _step(state, "legal_bowled")
    assert s.min() >= 0 and s.max() <= 1


def test_first_row_legal_bowled(state):
    first = state.filter(pl.col("ball_seq") == 0)
    assert set(first["legal_bowled"].unique()) <= {0, 1}


def test_innings_runs_monotonic(state):
    assert _step(state, "innings_runs").min() >= 0


def test_runs_needed_never_increases(state):
    # a scoring shot only lowers runs_needed; it can never rise
    assert _step(state, "runs_needed").max() <= 0


# ---------- cross-checks vs parser (independent path) ----------

def test_final_state_matches_win_margin(state, matches):
    final = _final_rows(state).join(
        matches.select("match_id", "win_by", "win_margin"), on="match_id", how="left",
    )
    runs_wins = final.filter(pl.col("win_by") == "runs")
    bad = runs_wins.filter(pl.col("innings_runs") != pl.col("target") - 1 - pl.col("win_margin"))
    # ~4 matches don't reconcile for legit SOURCE reasons, not parser bugs:
    #  - innings-level penalty_runs not in our deliveries-based innings_runs (1269656, 947121)
    #  - inconsistent target/margin in associate-nation matches (1457218, 1495657)
    # A real cumulative-sum bug would break thousands, not a handful.
    assert bad.height / runs_wins.height < 0.005, f"{bad.height}/{runs_wins.height} mismatch"
    
def test_final_state_matches_label(state):
    final = _final_rows(state)
    assert final.filter((pl.col("chase_won") == 1) & (pl.col("innings_runs") < pl.col("target"))).height == 0
    assert final.filter((pl.col("chase_won") == 0) & (pl.col("innings_runs") >= pl.col("target"))).height == 0


def test_golden_gt_rcb(state):
    r = _final_rows(state).filter(pl.col("match_id") == "1535465").row(0, named=True)
    assert r["chase_won"] == 1 and r["innings_runs"] >= 156   # RCB chased 156


if __name__ == "__main__":
    pass