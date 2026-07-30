"""
CricketIQ agent — Phase 5.2 + Phase 6 win-prob tools.

An LLM that answers natural-language cricket questions by calling the deterministic tools.
Per the locked design the model NEVER writes a number itself: it emits tool CALLS (which
tool, which player/match, which phase/ball); we execute the deterministic tool; the values
come back from data. The model only plans and narrates.

Tools are exposed by INTENT (get_batter_stats(player_name, ...)) — the model uses player
NAMES, never registry IDs; name->id resolution + data disambiguation is hidden inside. The
Phase-6 win-prob tools (get_wp_delta, get_key_moments) read the precomputed timelines and
hand the model ready-made percentage forms (win_prob_pct, wp_delta_pts) so natural commentary
("swung 36 points, to 7%") stays grounded instead of being the model's own arithmetic.

Every result carries a uniform envelope: {status, source, ...ids..., ...values...}.

Run:  python -m cricketiq.agent.agent "How does Virat Kohli score in the death overs?"
"""
from __future__ import annotations

import json
import sys

from dotenv import load_dotenv
from openai import OpenAI

from cricketiq.agent.tools import stats, moments

load_dotenv()                 # pull OPENAI_API_KEY out of .env
client = OpenAI()
MODEL = "gpt-5-mini"          # swap to any tool-calling chat model your account has
MAX_TOOL_ROUNDS = 6


# ---------- intent tools: names in, registry IDs hidden, uniform envelope out ----------

def _pick(name):
    """Best-ranked candidate for a name (resolve() ranks by data), or None."""
    hits = stats.resolve(name, limit=5)
    return hits[0] if hits else None


def get_batter_stats(player_name, phase=None, season=None):
    p = _pick(player_name)
    if not p:
        return {"status": "not_found", "query": player_name}
    s = stats.batter_stats(p["id"], phase=phase, season=season)
    return {"status": "ok", "source": "batter_stats", "player": p["name"], "player_id": p["id"],
            "phase": phase, "season": season, **s}


def get_bowler_stats(player_name, phase=None, season=None):
    p = _pick(player_name)
    if not p:
        return {"status": "not_found", "query": player_name}
    s = stats.bowler_stats(p["id"], phase=phase, season=season)
    return {"status": "ok", "source": "bowler_stats", "player": p["name"], "player_id": p["id"],
            "phase": phase, "season": season, **s}


def get_matchup(batter_name, bowler_name, phase=None):
    b, w = _pick(batter_name), _pick(bowler_name)
    if not b:
        return {"status": "not_found", "query": batter_name}
    if not w:
        return {"status": "not_found", "query": bowler_name}
    s = stats.matchup(b["id"], w["id"], phase=phase)
    return {"status": "ok", "source": "matchup", "batter": b["name"], "batter_id": b["id"],
            "bowler": w["name"], "bowler_id": w["id"], "phase": phase, **s}


def get_venue_par(venue, season=None):
    r = stats.venue_par(venue, season=season)
    return {"status": "ok" if r["n"] else "not_found", "source": "venue_par",
            "venue": venue, "season": season, **r}


def get_wp_delta(match_id, ball_seq):
    r = moments.wp_delta(str(match_id), int(ball_seq))
    if not r.get("found"):
        return {"status": "not_found", "match_id": match_id, "ball_seq": ball_seq}
    r["match_id"] = int(match_id) if str(match_id).isdigit() else match_id   # numeric -> echo-safe for verify
    return {"status": "ok", "source": "wp_delta",
            "win_prob_pct": round(r["win_prob"] * 100, 1),      # percentage forms so the model
            "wp_delta_pts": round(r["wp_delta"] * 100, 1),      # narrates from the tool, not arithmetic
            **r}


def get_key_moments(match_id, top_n=5):
    r = moments.key_moments(str(match_id), top_n=top_n)
    if not r.get("found"):
        return {"status": "not_found", "match_id": match_id}
    r["match_id"] = int(match_id) if str(match_id).isdigit() else match_id   # numeric -> echo-safe for verify
    r["n_shown"] = len(r["moments"])                                         # so a 'Top N' header count is tool-backed
    for m in r["moments"]:
        m["wp_delta_pts"] = round(m["wp_delta"] * 100, 1)
        m["win_prob_pct"] = round(m["win_prob"] * 100, 1)
    return {"status": "ok", "source": "key_moments", **r}


TOOL_FNS = {
    "get_batter_stats": get_batter_stats, "get_bowler_stats": get_bowler_stats,
    "get_matchup": get_matchup, "get_venue_par": get_venue_par,
    "get_wp_delta": get_wp_delta, "get_key_moments": get_key_moments,
}

_PHASE = {"type": "string", "enum": ["powerplay", "middle", "death"],
          "description": "phase: powerplay=overs 1-6, middle=7-15, death=16-20"}
_SEASON = {"type": "integer", "description": "restrict to one year, e.g. 2024"}

TOOLS = [
    {"type": "function", "function": {"name": "get_batter_stats",
        "description": "Batting: runs, balls, dismissals, average, strike_rate, n — optionally by phase/season.",
        "parameters": {"type": "object", "properties": {"player_name": {"type": "string"}, "phase": _PHASE, "season": _SEASON}, "required": ["player_name"]}}},
    {"type": "function", "function": {"name": "get_bowler_stats",
        "description": "Bowling: balls, runs, wickets, economy, dot_pct, n — optionally by phase/season.",
        "parameters": {"type": "object", "properties": {"player_name": {"type": "string"}, "phase": _PHASE, "season": _SEASON}, "required": ["player_name"]}}},
    {"type": "function", "function": {"name": "get_matchup",
        "description": "Head-to-head of a batter vs a specific bowler: balls, runs, dismissals, strike_rate, n.",
        "parameters": {"type": "object", "properties": {"batter_name": {"type": "string"}, "bowler_name": {"type": "string"}, "phase": _PHASE}, "required": ["batter_name", "bowler_name"]}}},
    {"type": "function", "function": {"name": "get_venue_par",
        "description": "Average first-innings total (par) at a venue, optionally by season.",
        "parameters": {"type": "object", "properties": {"venue": {"type": "string"}, "season": _SEASON}, "required": ["venue"]}}},
    {"type": "function", "function": {"name": "get_wp_delta",
        "description": "Win-probability swing on ONE delivery: wp_delta_pts (signed percentage-point change), win_prob_pct (chasing side's % after the ball), runs_this_ball, wicket_fell. Needs match_id and ball_seq (0-indexed 2nd-innings delivery).",
        "parameters": {"type": "object", "properties": {"match_id": {"type": "string"}, "ball_seq": {"type": "integer"}}, "required": ["match_id", "ball_seq"]}}},
    {"type": "function", "function": {"name": "get_key_moments",
        "description": "The biggest win-probability swings in a match, largest first — the deliveries that decided it. Each has wp_delta_pts, win_prob_pct, over, runs_this_ball, wicket_fell. Needs match_id; optional top_n.",
        "parameters": {"type": "object", "properties": {"match_id": {"type": "string"}, "top_n": {"type": "integer"}}, "required": ["match_id"]}}},
]

SYSTEM = (
    "You are CricketIQ, a T20 analyst. Answer ONLY from the tools — never state a number "
    "you didn't get from a tool result. Each result has a 'status': if it is not 'ok', do "
    "not fabricate — tell the user the player, venue, or match isn't in the data. Mention "
    "the sample size n where given, and caution briefly if it's small (under ~30 balls). For "
    "win-probability, use the tools' win_prob_pct and wp_delta_pts fields (percentages and "
    "percentage-point swings) — do not convert probabilities yourself. Data covers six men's "
    "T20 leagues (IPL, T20I, BBL, Blast, PSL, CPL). Be concise and specific."
)


def ask(question: str, verbose: bool = True) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    facts = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content, "facts": facts}
        messages.append(msg)
        for tc in msg.tool_calls:
            name, args = tc.function.name, {}
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = TOOL_FNS.get(name)
                result = fn(**args) if fn else {"status": "error", "message": f"unknown tool {name}"}
            except Exception as e:                       # never let a bad call crash the loop
                result = {"status": "error", "message": str(e)}
            facts.append({"tool": name, "args": args, "result": result})
            if verbose:
                print(f"  [tool] {name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return {"answer": "(stopped: too many tool rounds)", "facts": facts}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How does Virat Kohli score in the death overs?"
    print(f"Q: {q}\n")
    print("\nA:", ask(q)["answer"])