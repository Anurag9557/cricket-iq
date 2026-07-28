"""
CricketIQ verified agent as a LangGraph state machine — Phase 5 final piece.

The flow that was a while-loop in repair.py, expressed as an explicit graph:

        agent ──has tool calls?──► tools ──┐
          ▲                                │
          └────────────────────────────────┘
          │ no tool calls
          ▼
        verify ──pass──────────────► END
          │ fail
          ├── repairs < MAX ──► repair ──► agent
          └── repairs = MAX ──► fallback ──► END

The branch is real — verify pass/fail with a bounded repair cycle and a deterministic safe
terminal — which is exactly what LangGraph is for, and why it was deferred until now instead
of bolted on early. Every node reuses logic already built and tested: the tool schema and
system prompt from agent.py, the verifier from verify.py, and the repair instruction + safe
fallback from repair.py. The graph adds structure, not new behaviour.

Run:  python -m cricketiq.agent.graph            # demo a clean + repaired question
      python -m cricketiq.agent.graph --mermaid  # print the state-machine diagram
"""
from __future__ import annotations

import json
import sys
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from cricketiq.agent.agent import client, MODEL, TOOLS, TOOL_FNS, SYSTEM
from cricketiq.agent.verify import verify
from cricketiq.agent.repair import REPAIR_INSTRUCTION, MAX_REPAIRS, _safe_fallback


class AgentState(TypedDict):
    messages: list
    facts: list
    answer: str
    verdict: str
    repairs: int
    resolution: str
    unsupported: list


# ---------- nodes (each returns a partial state update) ----------

def _to_dict(msg) -> dict:
    """Normalize an SDK assistant message to a plain dict, so graph state holds ONE message
    representation throughout. Mixing SDK objects and dicts works for in-memory invoke() but
    would break under a serializing checkpointer — and one representation is simply cleaner."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def agent_node(state: AgentState) -> dict:
    """One model turn. Emits tool calls, or a final answer."""
    resp = client.chat.completions.create(model=MODEL, messages=state["messages"], tools=TOOLS)
    msg = resp.choices[0].message
    assistant = _to_dict(msg)
    update: dict[str, Any] = {"messages": state["messages"] + [assistant]}
    if not msg.tool_calls:
        update["answer"] = msg.content or ""
    return update


def tools_node(state: AgentState) -> dict:
    """Execute the deterministic tools the model just called; accumulate facts."""
    last = state["messages"][-1]
    new_msgs, new_facts = [], []
    for tc in last["tool_calls"]:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
            fn = TOOL_FNS.get(name)
            result = fn(**args) if fn else {"status": "error", "message": f"unknown tool {name}"}
        except Exception as e:
            result = {"status": "error", "message": str(e)}
        new_facts.append({"tool": name, "args": args, "result": result})
        new_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    return {"messages": state["messages"] + new_msgs, "facts": state["facts"] + new_facts}


def verify_node(state: AgentState) -> dict:
    """Check every number in the answer against the accumulated tool facts."""
    v = verify(state["answer"], state["facts"])
    out: dict[str, Any] = {"verdict": v["verdict"], "unsupported": [d["text"] for d in v["unsupported"]]}
    if v["verdict"] == "pass":
        out["resolution"] = "clean" if state["repairs"] == 0 else "repaired"
    return out


def repair_node(state: AgentState) -> dict:
    """Hand the unsupported number(s) back with the verifier's invariant; loop to agent."""
    bad = ", ".join(state["unsupported"])
    instruction = {"role": "user", "content": REPAIR_INSTRUCTION.format(bad=bad)}
    return {"messages": state["messages"] + [instruction], "repairs": state["repairs"] + 1}


def fallback_node(state: AgentState) -> dict:
    """Repair exhausted: emit a deterministic statement of the facts, and ENFORCE that it's
    grounded rather than assume it — verify and assert, so a future regression in _safe_fallback
    fails loudly instead of silently returning 'pass' over an unsupported number."""
    answer = _safe_fallback(state["facts"])
    v = verify(answer, state["facts"])
    assert v["verdict"] == "pass", "invariant broken: safe fallback emitted an unsupported number"
    return {"answer": answer, "verdict": "pass", "resolution": "fell_back"}


# ---------- routing ----------

def route_after_agent(state: AgentState) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else "verify"


def route_after_verify(state: AgentState) -> str:
    if state["verdict"] == "pass":
        return "done"
    return "repair" if state["repairs"] < MAX_REPAIRS else "fallback"


# ---------- assembly ----------

def _build():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("verify", verify_node)
    g.add_node("repair", repair_node)
    g.add_node("fallback", fallback_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", route_after_verify, {"done": END, "repair": "repair", "fallback": "fallback"})
    g.add_edge("repair", "agent")
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = _build()


def ask_graph(question: str) -> dict:
    init: AgentState = {"messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": question}],
                        "facts": [], "answer": "", "verdict": "", "repairs": 0,
                        "resolution": "clean", "unsupported": []}
    final = GRAPH.invoke(init, {"recursion_limit": 25})
    return {"answer": final["answer"], "facts": final["facts"], "verdict": final["verdict"],
            "repairs": final["repairs"], "resolution": final["resolution"]}


_DEMO = [
    "What is Virat Kohli's strike rate in the death overs?",                     # clean path
    "How much lower is Jasprit Bumrah's death-over economy than Rashid Khan's?",  # repair path
    "Who has the better death-over strike rate, Virat Kohli or Suryakumar Yadav, and by how many points?",
]


def main():
    if "--mermaid" in sys.argv:
        print(GRAPH.get_graph().draw_mermaid())
        return
    for q in _DEMO:
        r = ask_graph(q)
        print(f"\n{'=' * 72}\n{q}")
        print(f"  resolution: {r['resolution']} | verdict: {r['verdict']} | repairs: {r['repairs']}")
        print(f"  answer: {r['answer']}")


if __name__ == "__main__":
    main()