# parse.py spec — derived from schema_survey.py over 7,690 files (Phase 1.2)

## Keep a match only if ALL of:
- exactly 2 innings                         (drops 107 abandoned + 59 super-over)
- outcome has 'winner' AND 'by'             (drops 168 tie/no-result → no label)
- outcome.method != 'D/L'                   (drops 276 DLS)
- year(dates[0]) >= 2008                    (drops 2005–07; data starts 2005)
- balls_per_over == 6                       (assert; all 7,690 pass)

## Per-ball rules
- over is 0-indexed  → over_no = over['over'] + 1
- legal ball  = 'wides' not in extras AND 'noballs' not in extras
                 (byes/legbyes DO count as legal)
- balls_remaining (chase) = 120 - legal_balls_bowled_so_far
- wicket falls = delivery has 'wickets' AND kind not in
                 {'retired hurt', 'retired not out'}     # retired out DOES count
- player_out stored explicitly (run-out can dismiss the non-striker)
- runs: team += runs['total'] (incl. extras); batter += runs['batter']
- target (chase) = 1st-innings total runs + 1

## Label
- chase_won = (innings[1].team == outcome.winner)  → 1 else 0
- classes are ~balanced: 3,790 chase wins vs 3,661 defends

## Trust / don't-trust
- always present: registry, venue, toss     → safe features
- sometimes missing: city (306), player_of_match (880) → don't depend on
- actual_delivery: present on 100% of deliveries → cross-check only (we compute ourselves)