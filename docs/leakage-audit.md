# Leakage Audit — state.parquet (Phase 1.7)

## Task
Predict P(chasing team wins | match state at ball t). One row per 2nd-innings
delivery; label = chase_won (same for every ball of a match).

## The rule
A feature at ball t may use ONLY:
- deliveries 1..t of THIS chase innings, and
- information fixed BEFORE the chase began (the target).
It may NOT use: future deliveries (t+1..end), the final outcome, or any
cross-match aggregate computed over data that includes this or later matches.

## Feature-by-feature certification
| feature                     | built from                       | leak-free? |
|-----------------------------|----------------------------------|------------|
| innings_runs                | cum-sum runs_total, balls 1..t   | ✓ past+present |
| wickets_lost / _in_hand     | cum-sum is_wicket, 1..t          | ✓ |
| legal_bowled / balls_remaining | cum-sum is_legal, 1..t        | ✓ |
| runs_needed                 | target − innings_runs            | ✓ target pre-chase |
| current_rr / required_rr / rr_diff | ratios of the above       | ✓ |
| runs_last30 / wkts_last30   | window balls t−29..t             | ✓ past+present |
| over / phase                | position of ball t               | ✓ |
| target                      | innings[1].target.runs (pre-chase) | ✓ |
| season                      | match metadata                   | ✓ (split only, not a feature) |
| chase_won                   | the LABEL                        | — prediction target |

Every core feature is a function of (chase deliveries 1..t) + (pre-chase target).
Leak-free by construction.

## Three things that look like leakage but aren't
1. "State after ball t includes ball t's outcome." Not leakage — ball t is
   present-time. Leakage would be ball t+1. We predict the final result from the
   present state; the current ball is part of the present.
2. Terminal rows (runs_needed=0 → won; wickets_in_hand=0 → lost). The state
   genuinely determines the result there, and the model correctly reads ~1/0.
   That's correctness (the live ticker SHOULD show 100% at the winning run), not
   leakage.
3. Correlated rows (~112 per match share one label). Not leakage, but two
   consequences: effective sample size << row count, and the split MUST be by
   match, never by shuffling rows.

## Split rule (enforced in Phase 2)
Split by season → by match: train ≤2023, val 2024, test 2025–26. ALL rows of a
match go to ONE split. NEVER shuffle-split rows — that scatters one match across
train and test (leakage + trivially easy). `season` is carried in state.parquet
for exactly this.

## The real danger: enrichment features (NOT built yet)
When player/venue features are added next, these are the landmines:
- batter/bowler form (SR, economy, avg): MUST be as-of-date — only matches
  strictly BEFORE this match's date. Using all matches = massive leakage.
- venue par / venue average: computed from TRAINING seasons only (or as-of-date),
  never the full dataset — else test info leaks into a training feature.
- team / matchup aggregates: same as-of-date rule.
Each enrichment feature gets its own audit line + a test that no future match
contributes, BEFORE it ships.

## Status
Core state features certified leak-free. Enrichment deferred with the
as-of-date rule flagged above.