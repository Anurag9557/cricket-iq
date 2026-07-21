"""Scratch tool: pretty-print the structure of ONE Cricsheet match file.

    python explore_match.py                 # most recent IPL match
    python explore_match.py 1426312.json    # a specific file in data/raw/ipl/
"""
import json
import sys

from cricketiq.core.config import RAW_DIR

folder = RAW_DIR / "ipl"
if len(sys.argv) > 1:
    path = folder / sys.argv[1]
else:
    path = max(folder.glob("*.json"), key=lambda p: int(p.stem))  # highest id = newest

match = json.loads(path.read_text(encoding="utf-8"))

print(f"File: {path.name}")
print(f"Top-level keys: {list(match.keys())}\n")

info = match["info"]
print("=== INFO ===")
print(f"  teams   : {info['teams']}")
print(f"  date    : {info['dates']}")
print(f"  venue   : {info.get('venue')} ({info.get('city')})")
print(f"  toss    : {info['toss']}")
print(f"  outcome : {info['outcome']}")
print(f"  method  : {info['outcome'].get('method', 'normal')}   # 'D/L' = DLS-affected")
print(f"  squads  : {[(t, len(p)) for t, p in info['players'].items()]}")
reg = info["registry"]["people"]
print(f"  registry: {len(reg)} name->id entries, e.g. {list(reg.items())[0]}\n")

print("=== INNINGS ===")
print(f"  count: {len(match['innings'])}")
for i, inn in enumerate(match["innings"]):
    print(f"  innings {i}: batting={inn['team']}, overs_recorded={len(inn['overs'])}")

print("\n=== FIRST OVER of innings 0, ball by ball ===")
over0 = match["innings"][0]["overs"][0]
print(f"  'over' field = {over0['over']}   <-- is this 0 or 1?")
legal = 0
for d in over0["deliveries"]:
    extras = d.get("extras", {})
    if not ("wides" in extras or "noballs" in extras):
        legal += 1
    tag = f"  WICKET({d['wickets'][0]['kind']})" if "wickets" in d else ""
    if extras:
        tag += f"  extras={extras}"
    print(f"    {d['batter']:<18} vs {d['bowler']:<18} runs={d['runs']['total']} legal={legal}{tag}")
print(f"\n  legal balls this over: {legal}")