"""
Tests for the answer verifier (Phase 5.2).

The whole point of a verifier is that it BITES. It must pass an answer whose every
number came from a tool, and FAIL an answer that slips in a number no tool produced —
whether hallucinated or hand-computed. The two `*_is_caught` tests are the ones that
matter: a verifier that only ever says 'pass' proves nothing.
"""
from cricketiq.agent.verify import verify

# shaped exactly like agent.ask() produces: a list of {tool, args, result}
FACTS = [{
    "tool": "get_batter_stats",
    "args": {"player_name": "Virat Kohli", "phase": "death"},
    "result": {"status": "ok", "source": "batter_stats", "player": "V Kohli",
               "player_id": "abc", "runs": 2487, "balls": 1296, "dismissals": 75,
               "average": 33.16, "strike_rate": 191.9, "n": 1296},
}]


def test_clean_answer_passes():
    ans = "Kohli scored 2,487 runs off 1,296 balls, average 33.16, strike rate 191.9 (n=1,296)."
    rep = verify(ans, FACTS)
    assert rep["verdict"] == "pass"
    assert not rep["unsupported"]


def test_rounded_number_still_passes():
    # 192 is a fair rounding of 191.9 — within half a unit, so not a fabrication
    assert verify("His strike rate is about 192.", FACTS)["verdict"] == "pass"


def test_fabricated_number_is_caught():
    # 205 came from nowhere — no tool produced it
    rep = verify("Kohli scored 2,487 runs at a strike rate of 205.", FACTS)
    assert rep["verdict"] == "fail"
    assert any(d["value"] == 205 for d in rep["unsupported"])


def test_hand_computed_number_is_caught():
    # the LLM doing its own arithmetic is a hallucination vector too: no tool returned 17.3
    rep = verify("That works out to a wicket every 17.3 balls.", FACTS)
    assert rep["verdict"] == "fail"
    assert any(d["value"] == 17.3 for d in rep["unsupported"])


def test_structural_numbers_not_flagged():
    # 16 and 20 are phase boundaries, not stat claims — structural bucket, not a failure
    rep = verify("In the death overs (16-20) he averaged 33.16.", FACTS)
    assert rep["verdict"] == "pass"
    assert {d["value"] for d in rep["structural"]} == {16, 20}


def test_wrong_by_a_little_is_caught():
    # 191.4 vs 191.9 differs by 0.5 in the tenths place — a real discrepancy, not rounding
    assert verify("Strike rate 191.4.", FACTS)["verdict"] == "fail"