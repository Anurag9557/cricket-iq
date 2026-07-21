"""Parse raw Cricsheet JSON into two flat Parquet tables.

    python -m cricketiq.data.parse

Reads   data/raw/<league>/*.json
Writes  data/processed/matches.parquet      (one row per KEPT match)
        data/processed/deliveries.parquet   (one row per ball of kept matches)

Keep/drop rules and per-ball logic follow docs/parser-spec.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cricketiq.core.config import RAW_DIR, PROCESSED_DIR, ERA_START_YEAR

LEAGUES = ["ipl", "t20i", "bbl", "blast", "psl", "cpl"]
NOT_OUT_KINDS = {"retired hurt", "retired not out"}  # batter leaves, no wicket charged


def parse_match(path: Path, league: str):
    """Parse ONE match file.

    Returns (match_row, delivery_rows) if kept, or (None, drop_reason) if dropped.
    """
    m = json.loads(path.read_text(encoding="utf-8"))
    info = m["info"]

    # --- match-level filters (docs/parser-spec.md) ---
    if info.get("balls_per_over") != 6:
        return None, "not_6_balls"
    innings = m.get("innings", [])
    if len(innings) != 2:
        return None, "not_2_innings"
    outcome = info.get("outcome", {})
    if "winner" not in outcome or "by" not in outcome:
        return None, "no_winner"
    if outcome.get("method") == "D/L":
        return None, "dls"
    year = int(info["dates"][0][:4])
    if year < ERA_START_YEAR:
        return None, "pre_era"

    reg = info["registry"]["people"]          # name -> stable id

    def resolve(name):                         # fall back to the name if unregistered
        return reg.get(name, name)

    bat_first = innings[0]["team"]
    chase = innings[1]["team"]
    by_type, by_margin = next(iter(outcome["by"].items()))   # {'runs': 8} -> ('runs', 8)
    match_id = path.stem
    first_innings_runs = 0

    delivery_rows = []
    for inn_idx, inn in enumerate(innings):
        innings_no = inn_idx + 1
        ball_seq = 0
        for over in inn["overs"]:
            over_no = over["over"] + 1          # normalize Cricsheet 0-index -> cricket 1-20
            for d in over["deliveries"]:
                extras = d.get("extras", {})
                is_legal = "wides" not in extras and "noballs" not in extras

                wkt = d.get("wickets", [])
                is_wicket, player_out, wicket_kind = False, None, None
                if wkt:
                    wicket_kind = wkt[0].get("kind")
                    player_out = resolve(wkt[0].get("player_out"))
                    is_wicket = wicket_kind not in NOT_OUT_KINDS

                runs = d["runs"]
                if innings_no == 1:
                    first_innings_runs += runs["total"]

                delivery_rows.append({
                    "match_id": match_id,
                    "innings": innings_no,
                    "ball_seq": ball_seq,
                    "over": over_no,
                    "batter": resolve(d["batter"]),
                    "bowler": resolve(d["bowler"]),
                    "non_striker": resolve(d["non_striker"]),
                    "runs_batter": runs["batter"],
                    "runs_extras": runs["extras"],
                    "runs_total": runs["total"],
                    "is_legal": is_legal,
                    "is_wicket": is_wicket,
                    "player_out": player_out,
                    "wicket_kind": wicket_kind,
                })
                ball_seq += 1

    match_row = {
        "match_id": match_id,
        "league": league,
        "date": info["dates"][0],
        "season": year,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_bat_first": bat_first,
        "team_chase": chase,
        "toss_winner": info["toss"]["winner"],
        "toss_decision": info["toss"]["decision"],
        "winner": outcome["winner"],
        "win_by": by_type,
        "win_margin": by_margin,
        "first_innings_runs": first_innings_runs,
        "target": first_innings_runs + 1,
        "chase_won": int(outcome["winner"] == chase),
    }
    return match_row, delivery_rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    match_rows, delivery_rows, drops, errors = [], [], {}, []

    for league in LEAGUES:
        for path in (RAW_DIR / league).glob("*.json"):
            try:
                match_row, payload = parse_match(path, league)
            except Exception as e:
                errors.append((path.name, str(e)))
                continue
            if match_row is None:
                drops[payload] = drops.get(payload, 0) + 1
            else:
                match_rows.append(match_row)
                delivery_rows.extend(payload)

    pl.DataFrame(match_rows).write_parquet(PROCESSED_DIR / "matches.parquet")
    pl.DataFrame(delivery_rows).write_parquet(PROCESSED_DIR / "deliveries.parquet")

    print(f"KEPT    {len(match_rows):>5d} matches, {len(delivery_rows):>9d} deliveries")
    print("DROPPED:")
    for reason, n in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"  {reason:14s} {n:>5d}")
    total = len(match_rows) + sum(drops.values()) + len(errors)
    print(f"ERRORS  {len(errors)}   (checked {total} files)")
    for name, msg in errors[:3]:
        print(f"    {name}: {msg}")


if __name__ == "__main__":
    main()