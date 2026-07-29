"""
CricketIQ answer verifier — Phase 5.2.

The agent grounds every number in a tool call, but nothing FORCES its prose to
match what the tools returned — it could copy a value wrong, average two of them
in its head, or slip one in from memory. This module is the check: it pulls every
number out of the final answer and confirms each one traces to a value a tool
actually returned.

It deliberately does NOT re-run the tools. Re-running is circular — the tools are
deterministic, so they always agree with themselves. The failure we hunt is in the
OTHER direction: a number in the narration that no tool produced. That is exactly
what a hallucinated (or silently hand-computed) statistic looks like.

A few numbers in an answer are structural, not claims — the phase boundaries
("overs 16-20"), the small-sample threshold ("~30 balls"), the league count. Those
get their own bucket: never counted as fabrications, but never silently credited as
tool-backed either, so a human sees exactly what matched what.

What this GUARANTEES vs what it does NOT. It proves every number in the prose is a
value some tool returned — no inventing, no hand-arithmetic. It does NOT prove the
narrator bound each value to the right label: 'runs' and 'balls' are both real tool
outputs, so an answer that swaps them ('1296 runs, 2487 balls') still passes. In
short, `tool_supported` means "this value appeared in tool output", NOT "this claim is
semantically correct". Closing that gap needs the narrator to emit structured
(stat, value) claims we check against the KEYED tool output — a later slice. The risk
is small (the tool JSON hands the model the correct keys) and, importantly, measurable:
the golden-set audit will show whether a swap ever actually happens, which is the right
reason to defer it rather than redesign now.

Nor does it verify logical INFERENCES over the values. "Kohli's death SR is higher than
his powerplay SR" is safe — it's directly readable from the two numbers — but "more than
40% higher" is a threshold judgment the verifier does not check: the answer "Yes" carries
no number, so it passes provenance. This module establishes the provenance of numeric
claims, not the soundness of conclusions drawn from them. That is the right scope for this
project (the alternative is a theorem prover); the boundary is stated, not hidden.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

# numbers as they appear in prose: 2,487 | 1,296 | 191.9 | 8.0 | 152 | 33.16
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# numbers that come from the prompt/question, not from a stat tool. Reported
# separately so they never count as fabrications — and never as tool-backed either.
# Known limitation: a real stat that happens to equal one of these is credited
# structural rather than flagged. Documented on purpose; the golden set avoids it.
STRUCTURAL = {1, 6, 7, 15, 16, 20, 30}


def _to_float(tok: str) -> float:
    return float(tok.replace(",", ""))


def _decimals(tok: str) -> int:
    tok = tok.replace(",", "")
    return len(tok.split(".")[1]) if "." in tok else 0


def _tool_values(facts: list[dict]) -> list[float]:
    """Every numeric value any tool returned — the set the prose is allowed to use. Walks
    nested lists/dicts too: some tools (get_key_moments) return a LIST of moment dicts and the
    real numbers live inside it, not at the result's top level."""
    vals = []

    def walk(x):
        if isinstance(x, bool):
            return                             # bool is an int subclass — never a stat
        if isinstance(x, (int, float)):
            vals.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    for f in facts:
        walk(f.get("result") or {})
    return vals


def _backed_by(p_val: float, p_tok: str, tool_vals: list[float]):
    """The tool value that backs this prose number within rounding, or None.
    Tolerance = half a unit in the prose number's LAST shown place, so a legitimately rounded
    value ('192' for 191.9) matches but a wrong one ('195') does not. Matches on MAGNITUDE — a
    signed swing stored as -36.3 is narrated 'fell 36.3 points', its sign carried by words, not
    digits — which is a qualitative claim the verifier doesn't police anyway."""
    tol = 0.5 * (10 ** (-_decimals(p_tok))) + 1e-9
    best, best_d = None, None
    for v in tool_vals:
        d = min(abs(p_val - v), abs(p_val + v))    # p matches a tool value or its sign-flip
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = v, d
    return best


def verify(answer: str, facts: list[dict]) -> dict:
    """Classify every number in `answer` as tool-backed, structural, or unsupported."""
    tool_vals = _tool_values(facts)
    # drop line-leading list markers ('1)', '2.', '10) ') so a numbered list's ordinals aren't
    # read as claims. Requires whitespace after the marker, so decimals ('36.3') are untouched.
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", answer or "")
    tool_supported, structural, unsupported = [], [], []
    for tok in _NUM.findall(text):
        val = _to_float(tok)
        backed = _backed_by(val, tok, tool_vals)
        if backed is not None:
            tool_supported.append({"text": tok, "value": val, "tool_value": backed})
        elif val == int(val) and int(val) in STRUCTURAL:
            structural.append({"text": tok, "value": val})
        else:
            unsupported.append({"text": tok, "value": val})
    return {
        "verdict": "fail" if unsupported else "pass",
        "n_numbers": len(tool_supported) + len(structural) + len(unsupported),
        "tool_supported": tool_supported,
        "structural": structural,
        "unsupported": unsupported,
    }


def _grouped(labels: list[str]) -> str:
    """Collapse repeats for readable output: ['20','20','20'] -> '20 x3'. Display only —
    the report dict itself keeps every occurrence, so the audit trail stays complete."""
    return ", ".join(f"{s} x{n}" if n > 1 else s for s, n in Counter(labels).items())


def _print_report(q: str, answer: str, rep: dict) -> None:
    print(f"Q: {q}\n\nA: {answer}\n")
    mark = "PASS" if rep["verdict"] == "pass" else "FAIL"
    print(f"VERIFICATION: {mark}  ({rep['n_numbers']} numbers | "
          f"{len(rep['tool_supported'])} tool-backed | {len(rep['structural'])} structural | "
          f"{len(rep['unsupported'])} unsupported)")
    if rep["tool_supported"]:
        print("  tool-backed : " + _grouped([f"{d['text']}->{d['tool_value']}" for d in rep["tool_supported"]]))
    if rep["structural"]:
        print("  structural  : " + _grouped([d["text"] for d in rep["structural"]]))
    if rep["unsupported"]:
        print("  UNSUPPORTED : " + _grouped([d["text"] for d in rep["unsupported"]]) + "   <-- no tool produced this")


if __name__ == "__main__":
    from cricketiq.agent.agent import ask                 # imported lazily: only main needs the LLM
    q = " ".join(sys.argv[1:]) or "How does Virat Kohli score in the death overs?"
    out = ask(q, verbose=False)
    _print_report(q, out["answer"], verify(out["answer"], out["facts"]))