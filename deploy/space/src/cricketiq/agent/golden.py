"""
Golden set for the Phase 5.3 audit — the ruler.

A benchmark of specific T20 split questions whose TRUE answers are computed directly
from the deterministic tools, so the ground truth can never drift from the data. Each
spec renders (a) a natural-language question a user would actually ask and (b) the exact
number stats.py returns for it. The audit runner (next slice) puts a plain LLM and our
grounded agent to the same questions and scores each stated number against these truths.

Design choice #1 — INTRINSIC rate stats only (strike rate, economy, dot %, par), never
raw cumulative aggregates. 191.9 is a property of how Kohli BATS; 2,487 runs is a
property of our data SNAPSHOT. A plain LLM failing to recall an intrinsic death-over
strike rate is a fair demonstration; dinging it for not knowing our exact dataset's
cumulative totals would not be. Intrinsic stats make the eventual accuracy gap defensible.

Design choice #2 — truth stays LIVE; the RUN gets persisted, not the ruler. We do NOT
freeze truth to disk. Freezing would break the property that makes this ruler trustworthy
— that it always matches the engine (and data) the agent itself uses. If the data grows
or a stat's definition changes, a frozen truth would mark a CORRECT agent wrong. So the
benchmark DEFINITION (these specs) is version-controlled here, the truth is recomputed
live each run, and each audit RUN writes its own immutable record (specs + truths-at-that-
moment + answers + scores + data/commit fingerprint). That record is the reproducible
artifact; the ruler itself stays synchronized.

Build the ruler and check it by eye BEFORE measuring with it: a wrong truth here would
silently mis-grade every run.
"""
from __future__ import annotations

from cricketiq.agent.tools import stats

# `metric` = the field read out of the tool result. `rel_tol` = RELATIVE tolerance when
# grading: a stated value counts as correct if |stated - truth| <= rel_tol * |truth|
# (so 0.03 = within 3%, room for rounding but not for a lucky guess).
SPECS = [
    # --- Batter: strike rate by phase ---
    {"id": "kohli_death_sr",    "kind": "batter",  "player": "Virat Kohli",      "phase": "death",     "metric": "strike_rate", "rel_tol": 0.03},
    {"id": "sky_death_sr",      "kind": "batter",  "player": "Suryakumar Yadav", "phase": "death",     "metric": "strike_rate", "rel_tol": 0.03},
    {"id": "rohit_pp_sr",       "kind": "batter",  "player": "Rohit Sharma",     "phase": "powerplay", "metric": "strike_rate", "rel_tol": 0.03},
    {"id": "kohli_pp_sr",       "kind": "batter",  "player": "Virat Kohli",      "phase": "powerplay", "metric": "strike_rate", "rel_tol": 0.03},
    # --- Bowler: economy + dot % by phase ---
    {"id": "bumrah_death_econ", "kind": "bowler",  "player": "Jasprit Bumrah",   "phase": "death",     "metric": "economy",     "rel_tol": 0.03},
    {"id": "rashid_mid_econ",   "kind": "bowler",  "player": "Rashid Khan",      "phase": "middle",    "metric": "economy",     "rel_tol": 0.03},
    {"id": "bumrah_death_dot",  "kind": "bowler",  "player": "Jasprit Bumrah",   "phase": "death",     "metric": "dot_pct",     "rel_tol": 0.03},
    {"id": "rashid_death_dot",  "kind": "bowler",  "player": "Rashid Khan",      "phase": "death",     "metric": "dot_pct",     "rel_tol": 0.03},
    # --- Matchup: batter vs bowler ---
    {"id": "sky_v_rashid_sr",   "kind": "matchup", "batter": "Suryakumar Yadav", "bowler": "Rashid Khan", "metric": "strike_rate", "rel_tol": 0.03},
    # --- Venue: par score ---
    {"id": "wankhede_par",      "kind": "venue",   "venue": "Wankhede Stadium",  "metric": "par",         "rel_tol": 0.02},
    # Edge cases (unknown player / unknown venue / tiny sample) join in the runner slice,
    # where a correct REFUSAL — not a number — is the pass condition, so they grade differently.
]

_PHASE = {"powerplay": "the powerplay (overs 1-6)", "middle": "the middle overs (7-15)",
          "death": "the death overs (16-20)"}


def _batter_item(s):
    res = stats.batter_stats(stats.pid(s["player"]), phase=s["phase"])
    q = f"What is {s['player']}'s strike rate in {_PHASE[s['phase']]}?"
    return q, res[s["metric"]], res["n"]


def _bowler_item(s):
    res = stats.bowler_stats(stats.pid(s["player"]), phase=s["phase"])
    if s["metric"] == "economy":
        q = f"What is {s['player']}'s economy rate in {_PHASE[s['phase']]}?"
    else:  # dot_pct
        q = f"What percentage of {s['player']}'s deliveries in {_PHASE[s['phase']]} are dot balls?"
    return q, res[s["metric"]], res["n"]


def _matchup_item(s):
    res = stats.matchup(stats.pid(s["batter"]), stats.pid(s["bowler"]))
    q = f"What is {s['batter']}'s strike rate against {s['bowler']}?"
    return q, res[s["metric"]], res["n"]


def _venue_item(s):
    res = stats.venue_par(s["venue"])
    q = f"What is the average first-innings (par) score at {s['venue']}?"
    return q, res[s["metric"]], res["n"]


_DISPATCH = {"batter": _batter_item, "bowler": _bowler_item,
             "matchup": _matchup_item, "venue": _venue_item}


def build_golden() -> list[dict]:
    """Render every spec to {id, question, truth, n, metric, rel_tol, spec}. A spec that
    blows up (unresolvable name, venue string with no match) is kept but marked ok=False,
    so one bad item surfaces loudly instead of crashing the whole ruler. `spec` is carried
    on each item as in-memory provenance — the truth can always be regenerated from it."""
    items = []
    for s in SPECS:
        try:
            q, truth, n = _DISPATCH[s["kind"]](s)
            items.append({"id": s["id"], "question": q, "truth": truth, "n": n,
                          "metric": s["metric"], "rel_tol": s["rel_tol"], "spec": s, "ok": True})
        except Exception as e:
            items.append({"id": s.get("id"), "question": f"[FAILED spec: {s}]", "truth": None,
                          "n": 0, "metric": s.get("metric"), "rel_tol": s.get("rel_tol", 0),
                          "spec": s, "ok": False, "error": str(e)})
    return items


if __name__ == "__main__":
    items = build_golden()
    print(f"{'id':20} {'n':>6} {'truth':>8}  question")
    print("-" * 100)
    for it in items:
        if not it["ok"]:
            print(f"{it['id'] or '?':20} {'ERR':>6} {'-':>8}  {it['question']}  ({it['error']})")
            continue
        flag = "   <-- THIN DATA (n<30)" if it["n"] < 30 else ""
        print(f"{it['id']:20} {it['n']:>6} {str(it['truth']):>8}  {it['question']}{flag}")