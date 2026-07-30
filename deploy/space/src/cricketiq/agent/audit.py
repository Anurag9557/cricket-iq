"""
CricketIQ audit runner — Phase 5.3, the headline.

Puts systems built on the SAME underlying model to the same benchmark, so the only
variable is grounding. Three arms:
  - GROUNDED        : our agent (tools + verifier).
  - PLAIN (offered) : the model from memory, explicitly offered an out ("...or say NO DATA").
  - PLAIN (pressed) : the model from memory, just asked for the number — how a real user asks.

Outcomes (four, not two — the fourth was learned the hard way):
  - correct   : within the item's relative tolerance of the truth
  - wrong     : a bare number, but off — a CONFIDENTLY WRONG stat (the dangerous case)
  - clarified : asked which competition/scope it means — an honest non-answer, NOT a fabrication
  - refused   : NO DATA / no number

'clarified' exists because the pressed model often answers a scope question with a scope
question ("IPL, T20Is, or all T20s?"). That is reasonable behaviour — it is surfacing the
ambiguity our tools resolve by design — and must never be scored as a hallucination. The
extractor treats a response that LEADS with a number as an assertion and a question as a
clarification; any other prose is a refusal. Reading only a leading number means numbers
buried in prose ('T20', '(16-20)', 'real-time', a year) can never masquerade as the answer.

Grading is DECOUPLED from measurement: every run persists raw answers + grounded facts, and
`--regrade` re-scores a saved run with the current logic and zero API calls. When the grader
improves, we re-grade history instead of re-running (and re-paying for) it.

Records land under audit_runs/ with model, git commit, and data fingerprint — the receipt.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime

from cricketiq.agent.agent import ask, client, MODEL
from cricketiq.agent.verify import verify
from cricketiq.agent.golden import build_golden
from cricketiq.core import config

PLAIN_SYSTEM = (
    "You are an expert T20 cricket analyst with broad knowledge of players across the "
    "major men's T20 competitions (IPL, T20Is, BBL, Blast, PSL, CPL). Answer as accurately "
    "as you can from your own knowledge."
)

# 'offered' — invites an honest refusal. Used by the grounded arm and the plain-offered arm.
_FMT_OUT = ("\n\nFor this question, ignore any usual formatting: respond with ONLY the single "
            "numeric value that answers it — no sample size, no words, no units. If you do not "
            "have the data, respond with exactly: NO DATA")
# 'pressed' — asks for the number, offers no out. How a real user asks; the hallucination test.
_FMT_NUM = "\n\nReply with only the single numeric value that answers this — no words, no units."

# Edge cases: entities that do not exist. Pass condition is a REFUSAL, not a number.
EDGE = [
    {"id": "unknown_player", "expect": "refusal",
     "question": "What is Jandamluck Fernsby's strike rate in the death overs?"},
    {"id": "unknown_venue", "expect": "refusal",
     "question": "What is the average first-innings par score at the Narnia Cricket Ground?"},
]

_CLARIFY_TOKENS = ("do you mean", "did you mean", "which format", "which competition",
                   "could you clarify", "can you clarify", "please clarify", "clarify")
# A response that LEADS with a number (optionally hedged) is an assertion of THAT number.
_LEADS_NUM = re.compile(r"(?i)^\s*(?:about|approx(?:imately)?|around|roughly|~|≈)?\s*(\d[\d,]*(?:\.\d+)?)")


def plain_answer(question: str, fmt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": question + fmt}],
    )
    return resp.choices[0].message.content or ""


def interpret(text: str):
    """(kind, number). A response that LEADS with a number is an assertion of that number;
    a question ('do you mean...?') is a clarification, not a fabrication; any other prose is a
    refusal/hedge. Reading ONLY a leading number means numbers buried inside prose — 'T20',
    '(16-20)', 'real-time', a year — can never masquerade as the answer, which is the leak
    that inflated an earlier run's wrong-count."""
    if not text or not text.strip():
        return ("refused", None)
    if "?" in text or any(p in text.lower() for p in _CLARIFY_TOKENS):
        return ("clarified", None)
    m = _LEADS_NUM.match(text)
    return ("answered", float(m.group(1).replace(",", ""))) if m else ("refused", None)


def _grounded_refused(facts: list) -> bool:
    """The agent correctly found the entity absent — read from tool facts, not prose."""
    return any((f.get("result") or {}).get("status") == "not_found" for f in facts)


def classify(item: dict, kind: str, num) -> str:
    if item.get("expect") == "refusal":
        return "correct" if kind != "answered" else "wrong"   # any asserted number about a non-entity is fabrication
    if kind == "clarified":
        return "clarified"
    if kind == "refused" or num is None:
        return "refused"
    truth, rel_tol = item["truth"], (item.get("rel_tol") or 0.03)
    return "correct" if abs(num - truth) <= rel_tol * abs(truth) else "wrong"


def run_one(item: dict) -> dict:
    base = item["question"]

    # grounded arm: our agent (tools + verifier). Store facts for provenance + re-grading.
    g = ask(base + _FMT_OUT, verbose=False)
    gk, gn = interpret(g["answer"])
    if _grounded_refused(g["facts"]):
        gk, gn = "refused", None
    grounded = {"answer": g["answer"], "facts": g["facts"], "number": gn,
                "outcome": classify(item, gk, gn), "grounded": verify(g["answer"], g["facts"])["verdict"]}

    # plain, offered an honest out
    po = plain_answer(base, _FMT_OUT)
    ok, on = interpret(po)
    offered = {"answer": po, "number": on, "outcome": classify(item, ok, on)}

    # plain, pressed for a number (realistic)
    pw = plain_answer(base, _FMT_NUM)
    wk, wn = interpret(pw)
    pressed = {"answer": pw, "number": wn, "outcome": classify(item, wk, wn)}

    return {"id": item["id"], "question": base, "expect": item.get("expect"),
            "truth": item.get("truth"), "rel_tol": item.get("rel_tol"),
            "grounded": grounded, "offered": offered, "pressed": pressed}


def _regrade_record(r: dict) -> None:
    """Re-derive all three arms' outcomes from stored raw answers + facts, in place."""
    item = {"truth": r.get("truth"), "rel_tol": r.get("rel_tol"), "expect": r.get("expect")}
    gk, gn = interpret(r["grounded"]["answer"])
    if _grounded_refused(r["grounded"].get("facts", [])):
        gk, gn = "refused", None
    r["grounded"]["number"], r["grounded"]["outcome"] = gn, classify(item, gk, gn)
    for arm in ("offered", "pressed"):
        k, n = interpret(r[arm]["answer"])
        r[arm]["number"], r[arm]["outcome"] = n, classify(item, k, n)


_MARK = {"correct": "C", "wrong": "W", "clarified": "?", "refused": "R"}


def _cell(rec: dict) -> str:
    v = rec["number"] if rec["number"] is not None else "-"
    return f"{v} {_MARK[rec['outcome']]}"


def _line(label: str, c: Counter, n: int, extra: str = "") -> str:
    return (f"{label:24}: {c['correct']}/{n} correct | {c['wrong']} wrong | "
            f"{c['clarified']} clarified | {c['refused']} refused{extra}")


def _report(records: list) -> dict:
    n = len(records)
    gc = Counter(r["grounded"]["outcome"] for r in records)
    oc = Counter(r["offered"]["outcome"] for r in records)
    wc = Counter(r["pressed"]["outcome"] for r in records)
    verified = sum(1 for r in records if r["grounded"].get("grounded") == "pass")

    print(f"\n{'id':18} {'truth':>7}  {'grounded':>11}  {'plain(out)':>11}  {'plain(press)':>12}")
    print("-" * 70)
    for r in records:
        t = "refuse" if r["expect"] == "refusal" else str(r["truth"])
        print(f"{r['id']:18} {t:>7}  {_cell(r['grounded']):>11}  {_cell(r['offered']):>11}  {_cell(r['pressed']):>12}")

    print("\n=== HEADLINE ===  (C=correct  W=confidently wrong  ?=clarified  R=refused)")
    print(_line("Grounded agent", gc, n, f" | verified {verified}/{n}"))
    print(_line("Plain LLM (offered out)", oc, n))
    print(_line("Plain LLM (pressed)", wc, n, "   <-- realistic hallucination test"))
    return {"n": n, "grounded": dict(gc), "offered": dict(oc), "pressed": dict(wc),
            "grounded_verified": verified}


def _meta() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    dp = config.PROCESSED_DIR / "deliveries.parquet"
    fp = {"file": dp.name, "bytes": dp.stat().st_size} if dp.exists() else {}
    return {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL, "git_commit": commit, "data": fp}


def main():
    items = build_golden() + EDGE
    records = [run_one(it) for it in items]
    summary = _report(records)
    meta = _meta()
    os.makedirs("audit_runs", exist_ok=True)
    stamp = meta["timestamp"].replace(":", "").replace("-", "")
    path = os.path.join("audit_runs", f"run_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary, "records": records}, fh, indent=2)
    print(f"\nrun record -> {path}")


def regrade(path: str | None = None):
    """Re-score a saved run with the CURRENT grading logic — no API calls, deterministic."""
    if path is None:
        files = glob.glob(os.path.join("audit_runs", "*.json"))
        if not files:
            print("no run records in audit_runs/")
            return
        path = max(files, key=os.path.getmtime)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for r in data["records"]:
        _regrade_record(r)
    print(f"re-graded {path} with current logic (no API calls):")
    _report(data["records"])


if __name__ == "__main__":
    if "--regrade" in sys.argv:
        regrade()
    else:
        main()