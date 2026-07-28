"""
Verified agent with a REPAIR loop — Phase 5.3b.

The stress battery found one clean failure mode: asked "by how much", the model COMPUTES a
number no tool returned (a difference, a percentage) and states it. The verifier catches it.
Repair closes the loop: on a verify FAIL, we hand the offending number(s) back with the
verifier's own invariant — every number you state must come from a tool — and re-verify, up
to MAX_REPAIRS. If it still fails, we do NOT emit the model's broken answer: we fall back to a
deterministic statement of the verified tool figures. The system would rather say less than say
something wrong, so an unsupported number can never leave it.

Two deliberate design choices (both from adversarial review):
  - Invariant, not policy. The repair prompt enforces "every number must be tool-supported",
    NOT "never compute". The day a get_difference tool exists, computing-via-tool is legitimate
    and nothing here changes — the repair still just enforces the verifier's rule.
  - Deterministic terminal state. MAX_REPAIRS caps the loop; the safe fallback guarantees the
    final answer always passes verification. That makes the (soon-to-be LangGraph) machine total.

Reuses the tools, schema, system prompt, model and client from agent.py, so the baseline agent
stays untouched (the audit still compares against it).

Run:  python -m cricketiq.agent.repair
"""
from __future__ import annotations

import json

from cricketiq.agent.agent import client, MODEL, TOOLS, TOOL_FNS, SYSTEM
from cricketiq.agent.verify import verify

MAX_TOOL_ROUNDS = 6
MAX_REPAIRS = 2

REPAIR_INSTRUCTION = (
    "Your previous answer contained numeric value(s) that no tool returned: {bad}. "
    "Invariant: every number you state must appear verbatim in a tool result from this "
    "conversation. If a tool can supply the needed value, call it. Otherwise answer using only "
    "tool-provided numbers and qualitative comparisons ('lower', 'higher', 'more'), and do not "
    "state a figure the tools did not give you."
)


def _run_tool_rounds(messages: list):
    """Drive the tool-calling loop until the model returns a final text answer.
    Returns (answer, facts_this_call, messages)."""
    facts = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content, facts, messages
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = TOOL_FNS.get(name)
                result = fn(**args) if fn else {"status": "error", "message": f"unknown tool {name}"}
            except Exception as e:
                result = {"status": "error", "message": str(e)}
            facts.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "(stopped: too many tool rounds)", facts, messages


def _safe_fallback(facts: list) -> str:
    """A deterministic answer built only from verified tool numbers — guaranteed to pass
    verification. Used when repair is exhausted, so the system never emits an unsupported
    number even if the model keeps recomputing."""
    lines = []
    for f in facts:
        r = f.get("result") or {}
        nums = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not nums:
            continue
        who = r.get("player") or r.get("batter") or r.get("venue") or f.get("tool", "result")
        lines.append(f"{who}: " + ", ".join(f"{k} {v}" for k, v in nums.items()))
    body = "\n".join(lines) if lines else "(no verified figures were retrieved)"
    return ("I can't give that answer without stating a number no tool provided. "
            "Here are the verified figures I retrieved:\n" + body)


def _snap(answer: str, v: dict) -> dict:
    return {"answer": answer, "verdict": v["verdict"], "unsupported": [d["text"] for d in v["unsupported"]]}


def ask_verified(question: str, verbose: bool = True) -> dict:
    """Answer, verify, and REPAIR up to MAX_REPAIRS times until every number is tool-backed.
    resolution is one of: 'clean' (passed first try), 'repaired' (passed after repair),
    'fell_back' (repair exhausted -> deterministic safe answer)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    answer, facts, messages = _run_tool_rounds(messages)
    all_facts = list(facts)
    v = verify(answer, all_facts)
    history = [_snap(answer, v)]

    if v["verdict"] == "pass":
        return {"answer": answer, "facts": all_facts, "verdict": "pass",
                "repairs": 0, "resolution": "clean", "history": history}

    repairs = 0
    while v["verdict"] == "fail" and repairs < MAX_REPAIRS:
        repairs += 1
        bad = ", ".join(d["text"] for d in v["unsupported"])
        if verbose:
            print(f"  [repair {repairs}] unsupported: {bad} -> re-grounding")
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": REPAIR_INSTRUCTION.format(bad=bad)})
        answer, facts, messages = _run_tool_rounds(messages)
        all_facts += facts                       # a repair round may legitimately call more tools
        v = verify(answer, all_facts)             # verify against ALL facts gathered
        history.append(_snap(answer, v))

    if v["verdict"] == "pass":
        resolution = "repaired"
    else:
        answer = _safe_fallback(all_facts)        # never emit the model's unsupported number
        v = verify(answer, all_facts)
        resolution = "fell_back"
        history.append(_snap(answer, v))

    return {"answer": answer, "facts": all_facts, "verdict": v["verdict"],
            "repairs": repairs, "resolution": resolution, "history": history}


# The four questions the stress battery caught the agent computing numbers on.
_DEMO = [
    "How much lower is Jasprit Bumrah's death-over economy than Rashid Khan's?",
    "How much higher is Virat Kohli's death-overs strike rate than his powerplay strike rate?",
    "Who has the better death-over strike rate, Virat Kohli or Suryakumar Yadav, and by how many points?",
    "Is Kohli's death-over strike rate more than 40% higher than his powerplay strike rate?",
]


def main():
    rows = []
    for q in _DEMO:
        print(f"\n{'=' * 72}\n{q}")
        r = ask_verified(q, verbose=True)
        print(f"  first : {r['history'][0]['verdict'].upper()} "
              f"(unsupported: {', '.join(r['history'][0]['unsupported']) or '-'})")
        print(f"  final : {r['verdict'].upper()} after {r['repairs']} repair(s) [{r['resolution']}]")
        print(f"  answer: {r['answer']}")
        rows.append(r)

    print(f"\n{'=' * 72}\n{'#':>2}  {'initial':>8}  {'repairs':>7}  {'final':>6}  {'resolution':>10}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['history'][0]['verdict'].upper():>8}  {r['repairs']:>7}  "
              f"{r['verdict'].upper():>6}  {r['resolution']:>10}")

    failed = [r for r in rows if r["history"][0]["verdict"] == "fail"]
    recovered = [r for r in failed if r["resolution"] == "repaired"]
    fellback = [r for r in rows if r["resolution"] == "fell_back"]
    avg = sum(r["repairs"] for r in rows) / len(rows) if rows else 0
    print(f"\nprovenance failures recovered by repair : {len(recovered)}/{len(failed)}")
    print(f"average repair iterations               : {avg:.2f}")
    print(f"fell back to safe facts (unrecovered)   : {len(fellback)}")


if __name__ == "__main__":
    main()