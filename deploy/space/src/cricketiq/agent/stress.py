"""
Stress battery for the agent — Phase 5.3b: find REAL failures before building repair.

Hard questions the tools don't directly answer, in three families that tend to break a
tool-calling agent:
  - comparisons / differences : tempt the LLM to COMPUTE a number no tool returned
                                ("Bumrah is 0.24 lower") -> verifier should flag it.   [repair target]
  - out-of-scope stats        : no matching tool (highest score, sixes, chasing) -> the
                                agent should refuse, or it mis-maps and answers the wrong thing.
  - superlatives / lists      : need a scan we don't have ("which bowler is best").

We run each through the agent + verifier and watch for:
  - VERIFY: FAIL              -> the answer states an unsupported (derived/invented) number.
  - a confident answer with 0 tool calls -> likely fabrication or a mis-mapped tool.
The failures we find here DEFINE what repair + the state machine must actually handle. If the
agent instead refuses cleanly everywhere, that robustness is itself a result worth reporting.

Run:  python -m cricketiq.agent.stress
"""
from cricketiq.agent.agent import ask
from cricketiq.agent.verify import verify

QUESTIONS = [
    # --- comparisons / arithmetic: the model must state a difference no tool produced ---
    "How much lower is Jasprit Bumrah's death-over economy than Rashid Khan's?",
    "How much higher is Virat Kohli's death-overs strike rate than his powerplay strike rate?",
    "Who has the better death-over strike rate, Virat Kohli or Suryakumar Yadav, and by how many points?",
    "Is Kohli's death-over strike rate more than 40% higher than his powerplay strike rate?",
    # --- out-of-scope stats: no such tool exists ---
    "What is Virat Kohli's highest T20 score?",
    "How many sixes has Rohit Sharma hit in the powerplay?",
    "What is MS Dhoni's strike rate when chasing a target?",
    # --- superlative / list: needs a scan we don't have ---
    "Which bowler has the best death-over economy in your data?",
    # --- ambiguous scope: no phase given ---
    "What is Virat Kohli's strike rate?",
    # --- two-hop compare ---
    "Does Suryakumar Yadav score faster against Rashid Khan than in the death overs generally?",
]


def main():
    fails = 0
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 72}\n[{i}] {q}")
        out = ask(q, verbose=True)
        v = verify(out["answer"], out["facts"])
        print(f"\nA: {out['answer']}")
        unsup = ", ".join(d["text"] for d in v["unsupported"])
        verdict = v["verdict"].upper()
        if v["verdict"] == "fail":
            fails += 1
        print(f"VERIFY: {verdict}"
              + (f" | unsupported: {unsup}" if unsup else "")
              + f" | tools called: {len(out['facts'])}")
    print(f"\n{'=' * 72}\nverify FAILs: {fails}/{len(QUESTIONS)}  "
          f"(each FAIL is a real repair target; 0 FAILs = the agent is robust here)")


if __name__ == "__main__":
    main()