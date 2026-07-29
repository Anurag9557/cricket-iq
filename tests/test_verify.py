"""
Tests for the answer verifier (Phase 5.2 + Phase 6).

The whole point of a verifier is that it BITES. It must pass an answer whose every number
came from a tool, and FAIL an answer that slips in a number no tool produced — whether
hallucinated or hand-computed. The `*_is_caught` / `*_still_caught` tests are the ones that
matter: a verifier that only ever says 'pass' proves nothing.

The Phase-6 block guards the win-prob wiring: numbers nested in a list of moment dicts, signed
swings narrated with the sign in words, and numbered-list ordinals that must NOT read as claims.
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
    assert verify("His strike rate is about 192.", FACTS)["verdict"] == "pass"


def test_fabricated_number_is_caught():
    rep = verify("Kohli scored 2,487 runs at a strike rate of 205.", FACTS)
    assert rep["verdict"] == "fail"
    assert any(d["value"] == 205 for d in rep["unsupported"])


def test_hand_computed_number_is_caught():
    rep = verify("That works out to a wicket every 17.3 balls.", FACTS)
    assert rep["verdict"] == "fail"
    assert any(d["value"] == 17.3 for d in rep["unsupported"])


def test_structural_numbers_not_flagged():
    rep = verify("In the death overs (16-20) he averaged 33.16.", FACTS)
    assert rep["verdict"] == "pass"
    assert {d["value"] for d in rep["structural"]} == {16, 20}


def test_wrong_by_a_little_is_caught():
    assert verify("Strike rate 191.4.", FACTS)["verdict"] == "fail"


# --- Phase 6: nested tool results (get_key_moments) + signed swings + list format ---

KM_FACTS = [{"tool": "get_key_moments", "args": {"match_id": "1512844"}, "result": {
    "status": "ok", "source": "key_moments", "match_id": 1512844,
    "n_deliveries": 125, "n_key_moments": 31, "n_shown": 2,
    "moments": [
        {"ball_seq": 121, "over": 20, "wp_delta": -0.363, "win_prob": 0.071,
         "runs_this_ball": 0, "wicket_fell": True, "wp_delta_pts": -36.3, "win_prob_pct": 7.1},
        {"ball_seq": 103, "over": 17, "wp_delta": 0.276, "win_prob": 0.628,
         "runs_this_ball": 6, "wicket_fell": False, "wp_delta_pts": 27.6, "win_prob_pct": 62.8},
    ]}}]


def test_nested_list_numbers_are_found():
    # numbers inside the moments list (a list of dicts) must be tool-backed, not flagged
    rep = verify("Over 20 saw a 36.3-point swing to 7.1%; over 17 a 27.6-point swing to 62.8%.", KM_FACTS)
    assert rep["verdict"] == "pass"
    assert not rep["unsupported"]


def test_signed_swing_matches_on_magnitude():
    # -36.3 in the tool, narrated 'fell 36.3 points' — the sign is carried by the word, not the digits
    assert verify("Win probability fell 36.3 points.", KM_FACTS)["verdict"] == "pass"


def test_numbered_list_ordinals_not_flagged():
    ans = "Top 2 swings:\n1) Over 20: 36.3 points to 7.1%\n2) Over 17: 27.6 points to 62.8%"
    assert verify(ans, KM_FACTS)["verdict"] == "pass"


def test_fabricated_nested_value_still_caught():
    # 99.9 is in no moment — must FAIL even surrounded by grounded nested numbers
    rep = verify("Over 20 swung 36.3 points, but over 17 swung 99.9 points.", KM_FACTS)
    assert rep["verdict"] == "fail"
    assert any(d["value"] == 99.9 for d in rep["unsupported"])