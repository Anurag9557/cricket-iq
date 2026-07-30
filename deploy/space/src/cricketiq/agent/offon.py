"""
Verification OFF vs ON — the Phase 5.4 headline.

Measures the VERIFIER'S value: how often the tool-calling agent states an UNGROUNDED number
(one no tool produced — a computed difference or a fabrication) with verification OFF, versus
with the verify+repair loop ON.

Both arms come from ONE run per question: ask_verified() answers, then verifies and repairs.
Its FIRST answer (history[0]) is the pre-repair state = verification OFF; its final verdict is
post-repair = verification ON. Same initial generation, so it's a fair before/after.

ON is ~0% by construction — repair re-grounds, and the safe fallback guarantees no ungrounded
number survives. The gap is the number: "X% of answers carried an ungrounded stat; with
verification, ~0%." The question set is a deliberate MIX (lookups the agent grounds on its own,
comparisons that tempt it to compute, edge cases it should refuse) so the OFF rate reflects
realistic usage, not a set cherry-picked to inflate it.

Run:  python -m cricketiq.agent.offon
"""
from __future__ import annotations

from collections import Counter

from cricketiq.agent.repair import ask_verified

QUESTIONS = [
    # --- simple lookups: the agent should ground these on its own (PASS even OFF) ---
    "How does Virat Kohli score in the death overs?",
    "What is Jasprit Bumrah's economy in the death overs?",
    "What is Rohit Sharma's strike rate in the powerplay?",
    "What's the par score at Wankhede Stadium?",
    "How does Suryakumar Yadav do against Rashid Khan?",
    "What percentage of Rashid Khan's death-over deliveries are dot balls?",
    "How many wickets does Jasprit Bumrah take at the death?",
    "What is MS Dhoni's strike rate in the death overs?",
    "What were the biggest win-probability swings in match 1512844?",
    "By how much did the win probability swing on ball 121 of match 1512844?",
    # --- comparisons / arithmetic: tempt the model to COMPUTE (ungrounded OFF, repaired ON) ---
    "How much lower is Bumrah's death economy than Rashid Khan's?",
    "How much higher is Kohli's death strike rate than his powerplay strike rate?",
    "Who has the better death strike rate, Kohli or Suryakumar Yadav, and by how many points?",
    "What's the difference in dot-ball percentage between Bumrah and Rashid at the death?",
    "Is Kohli's death strike rate more than 40% higher than his powerplay strike rate?",
    # --- out-of-scope / edge: the agent should refuse, not fabricate ---
    "What is Virat Kohli's highest T20 score?",
    "How many sixes has Rohit Sharma hit in the powerplay?",
    "What is Jandamluck Fernsby's strike rate in the death overs?",
]


def main():
    rows = []
    for q in QUESTIONS:
        r = ask_verified(q, verbose=False)
        off = r["history"][0]["verdict"]          # before repair = verification OFF
        on = r["verdict"]                          # after repair + fallback = verification ON
        rows.append((q, off, on, r["repairs"], r["resolution"]))
        print(f"  {off.upper():4} -> {on.upper():4}  [{r['resolution']}]  {q[:60]}")

    n = len(rows)
    off_fail = sum(1 for _, off, _, _, _ in rows if off == "fail")
    on_fail = sum(1 for _, _, on, _, _ in rows if on == "fail")
    res = Counter(r[4] for r in rows)

    print(f"\n{'=' * 72}\n=== VERIFICATION OFF vs ON  (n={n}) ===")
    print(f"OFF (ask only)       : {off_fail}/{n} answers carried an ungrounded stat  ({off_fail / n * 100:.0f}%)")
    print(f"ON  (verify + repair): {on_fail}/{n}  ({on_fail / n * 100:.0f}%)   "
          f"[{res['clean']} clean, {res['repaired']} repaired, {res['fell_back']} safe-fallback]")
    print(f"\nHEADLINE: verification took the ungrounded-stat rate from {off_fail / n * 100:.0f}% to "
          f"{on_fail / n * 100:.0f}% over {n} questions.")


if __name__ == "__main__":
    main()