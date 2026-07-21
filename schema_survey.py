"""Survey ALL downloaded Cricsheet files and report structure variation.

Run:  python schema_survey.py

Answers what parse.py must handle: outcome shapes, DLS count, ties/no-results,
innings counts (super overs / abandoned), wicket kinds, extras types,
optional-field presence, squad sizes, date range, balls-per-over.
"""
from __future__ import annotations

import json
from collections import Counter

from cricketiq.core.config import RAW_DIR

LEAGUES = ["ipl", "t20i", "bbl", "blast", "psl", "cpl"]


def survey() -> None:
    n_files = n_bad = dls = 0
    per_league = Counter(); innings_counts = Counter(); outcome_keys = Counter()
    by_types = Counter(); result_types = Counter(); wicket_kinds = Counter()
    extras_types = Counter(); squad_sizes = Counter(); balls_per_over = Counter()
    has_actual = Counter(); has_field = Counter(); years = []

    for league in LEAGUES:
        for path in (RAW_DIR / league).glob("*.json"):
            n_files += 1
            per_league[league] += 1
            try:
                m = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                n_bad += 1
                continue

            info = m.get("info", {})
            for f in ("registry", "player_of_match", "city", "venue", "toss"):
                if f in info:
                    has_field[f] += 1
            balls_per_over[info.get("balls_per_over", "missing")] += 1
            for players in info.get("players", {}).values():
                squad_sizes[len(players)] += 1
            if info.get("dates"):
                years.append(str(info["dates"][0])[:4])

            oc = info.get("outcome", {})
            outcome_keys[tuple(sorted(oc.keys()))] += 1
            if "by" in oc:
                for k in oc["by"]:
                    by_types[k] += 1
            if oc.get("method") == "D/L":
                dls += 1
            if "result" in oc:
                result_types[oc["result"]] += 1

            inns = m.get("innings", [])
            innings_counts[len(inns)] += 1
            for inn in inns:
                for over in inn.get("overs", []):
                    for d in over.get("deliveries", []):
                        has_actual["yes" if "actual_delivery" in d else "no"] += 1
                        for ex in d.get("extras", {}):
                            extras_types[ex] += 1
                        for w in d.get("wickets", []):
                            wicket_kinds[w.get("kind", "??")] += 1

    print(f"FILES: {n_files} total, {n_bad} failed to parse")
    print(f"per league: {dict(per_league)}")
    print(f"year range: {min(years)} .. {max(years)}")
    print(f"\nINNINGS per match: {dict(sorted(innings_counts.items()))}  (2=normal, 0/1=abandoned, 3-4=super over)")
    print("\nOUTCOME key-combos:")
    for keys, c in outcome_keys.most_common():
        print(f"  {list(keys)}: {c}")
    print(f"  by-type: {dict(by_types)}   DLS: {dls}")
    print(f"  result-types: {dict(result_types)}")
    print(f"\nWICKET kinds: {dict(wicket_kinds)}")
    print(f"EXTRAS types: {dict(extras_types)}")
    print(f"SQUAD sizes: {dict(sorted(squad_sizes.items()))}")
    print(f"BALLS/over: {dict(balls_per_over)}")
    print(f"actual_delivery on deliveries: {dict(has_actual)}")
    print(f"\ninfo fields present (of {n_files}): {dict(has_field.most_common())}")


if __name__ == "__main__":
    survey()