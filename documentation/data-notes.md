# Cricsheet JSON — data notes (Phase 1.2)

Example match: 1535465.json — Gujarat Titans vs Royal Challengers Bengaluru,
2026-05-31, Narendra Modi Stadium, Ahmedabad. RCB won by 5 wickets (chasing).

## Top-level shape
3 keys: `meta`, `info`, `innings`.
- meta    = data version / created date → ignore
- info    = everything ABOUT the match
- innings = the ball-by-ball record (a list, usually 2)

## info block — what I can use
- teams   : list of 2 team names
- dates   : list (usually 1) → drives the temporal train/val/test split
- venue + city
- toss    : {winner, decision}
- outcome shapes I must handle:
    - {winner: X, by: {wickets: N}}  → chasing team won   (my match: RCB by 5)
    - {winner: X, by: {runs: N}}     → team batting first won
    - tie / no result                → NO winner
    - may carry method: "D/L"        → DLS-affected → EXCLUDE
- players : {team: [names]} — squad size can be 12 (Impact Player rule), not always 11
- registry.people : {name: id} → store IDs, not names (names collide / vary spelling)

## innings / overs / deliveries
- innings[0] = 1st innings; innings[1] = the CHASE (my model's prediction target)
- a won/all-out chase ENDS EARLY: my match innings[1] = only 18 overs recorded
- over  : {over: <0-indexed!>, deliveries: [...]}   — over 0 = the 1st over
- delivery: {batter, bowler, non_striker, runs: {batter, extras, total}}
    - extras  (only if any): {wides | noballs | byes | legbyes}
    - wickets (only if one) : [{player_out, kind, fielders?}]

## Gotchas → each becomes a line in parse.py
1. over is 0-indexed → +1 to line up with config.PHASES (overs 1–20)
2. wides/no-balls are NOT legal balls → my over 0 had 9 deliveries, 3 wides, 6 legal
   balls_remaining = 120 − (legal balls bowled)   [T20 = 20×6]
3. chase can end before 120 balls (won or all out) → don't assume 20 full overs
4. outcome may be tie / no result → no winner label
5. DLS matches (outcome.method == "D/L") → exclude
6. squad size may be 12, not 11 → don't hardcode