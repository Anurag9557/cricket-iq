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
    """Every numeric value any tool returned — the set the prose is allowed to use."""
    vals = []
    for f in facts:
        for v in (f.get("result") or {}).values():
            if isinstance(v, bool):
                continue                       # bool is an int subclass — never a stat
            if isinstance(v, (int, float)):
                vals.append(float(v))
    return vals


def _backed_by(p_val: float, p_tok: str, tool_vals: list[float]):
    """The tool value that backs this prose number within rounding, or None.
    Tolerance = half a unit in the prose number's LAST shown place, so a legitimately
    rounded value ('192' for 191.9) matches but a wrong one ('195') does not."""
    tol = 0.5 * (10 ** (-_decimals(p_tok))) + 1e-9
    best = None
    for v in tool_vals:
        if abs(p_val - v) <= tol and (best is None or abs(p_val - v) < abs(p_val - best)):
            best = v
    return best


def verify(answer: str, facts: list[dict]) -> dict:
    """Classify every number in `answer` as tool-backed, structural, or unsupported."""
    tool_vals = _tool_values(facts)
    tool_supported, structural, unsupported = [], [], []
    for tok in _NUM.findall(answer or ""):
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