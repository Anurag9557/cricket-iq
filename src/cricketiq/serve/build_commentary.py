"""
Precompute verified auto-commentary for key moments (Phase 6.1).

For the biggest win-probability swings in each selected match, we gather the facts already
computed elsewhere — the swing and win-prob (timelines), the chase context (score, runs needed,
required rate, from serve/data.get_timeline) — and a DETERMINISTIC reason phrase derived from the
top SHAP driver (serve/explain.explain_ball). The LLM then narrates ONE grounded broadcast line,
weaving in that ready-made reason.

Two layers of protection against the two failure modes:
  - NUMERIC hallucination — the verifier checks every number against the facts; on failure we
    repair, and if repair fails we drop to a deterministic template (so every card is grounded).
  - CAUSAL error — the model is NOT asked to infer cause from raw SHAP metadata (which produced
    backwards lines like "a six… as the rate gap rises"). We map (feature, probability-direction)
    to a plain-English phrase ourselves; the model only weaves in the phrase it is given.

PRECOMPUTED (heavy LLM work offline) so the replay just reads — no LLM/network in the request
path, the demo can't fail live. Output: data/processed/commentary.parquet.

Run:  python -m cricketiq.serve.build_commentary 1512844        # one match (eyeball first)
      python -m cricketiq.serve.build_commentary                # default: top-3 most eventful
"""
from __future__ import annotations

import json
import sys

import polars as pl

from cricketiq.agent.agent import client, MODEL
from cricketiq.agent.verify import verify
from cricketiq.core import config
from cricketiq.serve import data
from cricketiq.serve.explain import explain_ball

TOP_K = 10           # commentary for the K biggest swings per match (bounds LLM cost)
MAX_REPAIRS = 2

# fields read from each get_timeline row — checked at runtime against YOUR local data before any
# LLM call, so a field-name mismatch fails fast with the real keys instead of a late KeyError.
_NEEDED = ("ball_seq", "over", "wp_delta", "win_prob", "runs_this_ball", "wicket_fell",
           "score", "wickets", "balls_remaining", "runs_needed", "required_rr", "is_key_moment")

# Deterministic causal phrase per (feature, probability-direction). direction "up" = this driver
# pushed the chasing side's win prob UP (helped the chase); "down" = pushed it down (hurt). We hand
# the phrase to the LLM so it never has to infer cause — killing the backwards-causality errors.
# Phrases are complete and grammatical, meant to be dropped in verbatim. NOTE: keyed on the model's
# feature names — a documented coupling; could later move to stable semantic IDs from explain_ball().
_REASON = {   # feature: (phrase when it HELPED the chase, phrase when it HURT it)
    "required_rr":     ("with the asking rate in reach",        "with the asking rate climbing"),
    "rr_diff":         ("putting the chase ahead of the rate",  "leaving the chase behind the rate"),
    "wickets_in_hand": ("keeping wickets in hand",              "after wickets became scarce"),
    "runs_needed":     ("closing in on the target",             "with plenty still needed"),
    "balls_remaining": ("with balls to spare",                  "as the deliveries ran out"),
    "current_rr":      ("with runs flowing",                    "with the run rate dipping"),
    "runs_last30":     ("riding recent momentum",               "after a quiet spell"),
    "wkts_last30":     ("with wickets in hand of late",         "after losing quick wickets"),
    "target":          ("chasing a modest target",              "chasing a big total"),
    "innings_runs":    ("now well set in the chase",            "still building the chase"),
    "over":            ("deep in the innings",                  "early in the chase"),
}

SYSTEM = (
    "You are a T20 cricket commentator. You are given the numeric facts of ONE key delivery and a "
    "ready-made REASON phrase. Write ONE punchy broadcast line of 12–20 words: describe the delivery "
    "and how it swung the chase. Use the reason phrase VERBATIM if it improves the line; OMIT it if "
    "it would sound forced — the event often tells the story on its own. State the swing cleanly: "
    "either the new win probability as a percent OR the change in points, never both mashed into one "
    "clause. Round numbers ('26 points', '72%', not '26.2'), and vary your opening across deliveries. "
    "HARD RULE: use only numbers that appear in the facts; never invent or calculate one. Never name "
    "a statistic code or say the words 'driver' or 'SHAP'."
)


def _reason(driver: dict | None) -> str | None:
    if not driver:
        return None
    helped = driver.get("direction") == "up"
    pair = _REASON.get(driver.get("feature"))
    if pair:
        return pair[0] if helped else pair[1]
    label = driver.get("label", "the situation")           # generic fallback for an unmapped feature
    return f"with {label} {'favouring' if helped else 'working against'} the chase"


def _numeric(b: dict) -> dict:
    """Only the numbers the commentary may cite — this is what the verifier checks against."""
    return {
        "over": b["over"],
        "wp_delta_pts": round(b["wp_delta"] * 100, 1),
        "win_prob_pct": round(b["win_prob"] * 100, 1),
        "runs_this_ball": b["runs_this_ball"], "wicket_fell": b["wicket_fell"],
        "score": b["score"], "wickets": b["wickets"],
        "balls_remaining": b["balls_remaining"], "runs_needed": b["runs_needed"],
        "required_rr": b["required_rr"],
    }


def _event_type(num: dict) -> str:
    """Deterministic event label so the UI renders an icon without parsing the sentence."""
    if num["wicket_fell"]:
        return "wicket"
    if num["runs_this_ball"] >= 4:
        return "boundary"
    if num["runs_this_ball"] == 0:
        return "dot"
    return "other"


def _fallback(num: dict) -> str:
    """Deterministic, grounded card from the facts alone — used only if the LLM won't stop inventing
    numbers. Guaranteed to pass verification (every number is a fact)."""
    event = "A wicket" if num["wicket_fell"] else (f"{num['runs_this_ball']} runs" if num["runs_this_ball"] >= 4 else "A dot under pressure")
    return (f"Over {num['over']}: {event} — win probability moves {num['wp_delta_pts']:+.0f} points to "
            f"{num['win_prob_pct']:.0f}% ({num['runs_needed']} needed off {num['balls_remaining']}).")


def _narrate(num: dict, reason: str | None):
    """LLM narrates from the numbers + the ready-made reason; verify; repair; else fallback.
    Returns (text, status, repairs) — repairs = repair rounds before it passed (0 = first pass)."""
    facts = [{"tool": "key_moment", "args": {}, "result": num}]
    user = (f"Facts (only these numbers may be used): {json.dumps(num)}\n"
            f"Reason to weave in: {reason}\nWrite the one-line commentary.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(1 + MAX_REPAIRS):
        text = (client.chat.completions.create(model=MODEL, messages=messages)
                .choices[0].message.content or "").strip()
        v = verify(text, facts)
        if v["verdict"] == "pass":
            return text, "verified", attempt
        bad = ", ".join(d["text"] for d in v["unsupported"])
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": f"These numbers are not in the facts: {bad}. Rewrite using ONLY the given numbers."})
    return _fallback(num), "fallback", MAX_REPAIRS


def _top_key_moments(timeline: list[dict]) -> list[dict]:
    keys = [b for b in timeline if b.get("is_key_moment")]
    keys.sort(key=lambda b: abs(b["wp_delta"]), reverse=True)     # biggest swings first
    return sorted(keys[:TOP_K], key=lambda b: b["ball_seq"])       # store chronologically


def build(match_ids: list[str]) -> None:
    rows = []
    for mid in match_ids:
        mid = str(mid)
        timeline = data.get_timeline(mid)
        for row in timeline:                                      # verify the interface vs local data (every row)
            missing = [k for k in _NEEDED if k not in row]
            if missing:
                print(f"ABORT: a get_timeline row is missing {missing}.\nActual keys: {sorted(row)}")
                return
        keys = _top_key_moments(timeline)
        print(f"\nmatch {mid}: {len(keys)} moments")
        for b in keys:
            num = _numeric(b)
            ex = explain_ball(mid, b["ball_seq"]) or {"drivers": []}
            top = ex["drivers"][0] if ex.get("drivers") else None
            reason = _reason(top)
            # drop a reason that contradicts the swing: the top driver can offset the event (a wicket
            # falling while the required rate eased), which reads as "a wicket putting the chase
            # ahead". If the reason's direction disagrees with the ball's net swing, let the event
            # speak for itself rather than pair it with a contradicting cause.
            if reason is not None and top and (b["wp_delta"] > 0) != (top.get("direction") == "up"):
                reason = None
            commentary, status, repairs = _narrate(num, reason)
            rows.append({"match_id": mid, "ball_seq": b["ball_seq"], "over": b["over"],
                         "event_type": _event_type(num),
                         "commentary": commentary, "status": status, "repairs": repairs,
                         "win_prob_pct": num["win_prob_pct"], "wp_delta_pts": num["wp_delta_pts"],
                         "runs_needed": num["runs_needed"], "balls_remaining": num["balls_remaining"],
                         "driver": (top["label"] if top else None), "reason": reason})
            print(f"  [{status:8} r{repairs}] over {b['over']:>2} ({_event_type(num)}): {commentary}")

    df = pl.DataFrame(rows)
    out = config.PROCESSED_DIR / "commentary.parquet"
    df.write_parquet(out)
    n_ver = df.filter(pl.col("status") == "verified").height
    print(f"\nwrote {len(rows)} cards -> {out}  ({n_ver} verified, {len(rows) - n_ver} fallback)")


def _default_matches(n: int = 3) -> list[str]:
    """The most eventful matches (most key moments) make the best demo."""
    tl = pl.read_parquet(config.PROCESSED_DIR / "timelines.parquet")
    counts = (tl.filter(pl.col("wp_delta").abs() >= 0.08)
                .group_by("match_id").len().sort("len", descending=True).head(n))
    return [str(r["match_id"]) for r in counts.iter_rows(named=True)]


if __name__ == "__main__":
    build(sys.argv[1:] or _default_matches())