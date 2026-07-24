# Results — CricketIQ win-probability model

Test: 2025–26, 145,985 held-out chase-deliveries (split by match — see leakage-audit.md).
Train ≤2023 (545,310 rows) · Val 2024 (91,247). Metrics via the verified harness
(tests/test_eval.py). Lower Brier/log-loss/ECE better; higher AUC better.

## Model ladder (TEST 2025–26)

| model | features | Brier | log-loss | AUC | ECE |
|-------|----------|------:|---------:|----:|----:|
| no-skill (base rate) | — | 0.2499 | ~0.693 | 0.500 | — |
| **B0 logistic** | wkts_in_hand, balls_remaining, required_rr | **0.1204** | **0.3717** | **0.9147** | 0.0336 |

B0 cuts Brier 52% below no-skill and reaches 0.915 AUC with just three "resource"
features (the DLS intuition: wickets + balls + required rate).

## B0 coefficients (scaled, per std-dev) — sanity check
- required_rr    −3.61  → higher rate needed ⇒ less likely to win (dominant)
- wickets_in_hand +1.25 → more wickets ⇒ more likely
- balls_remaining −1.28 → at FIXED required rate, more balls = longer chase to
  sustain = harder ⇒ lower win prob (interpret holding required_rr constant)

## Next
- B1 (LightGBM, all features) — beat B0.
- Isotonic calibration — drive ECE → ~0; reliability diagrams.
- Ablations (±momentum, ±phase; per-phase Brier).

| **B1 LightGBM** | full 11-feature state | **0.1144** | **0.3520** | **0.9226** | 0.0261 |


## B1 feature importance (% of gain)
required_rr 57.5 · rr_diff 17.1 · target 7.5 · wickets_in_hand 5.6 · runs_needed 5.5
· wkts_last30 2.0 · innings_runs 1.9 · current_rr 1.5 · runs_last30 0.9
· balls_remaining 0.5 · over 0.1

The required-rate gap (required_rr + rr_diff ≈ 75%) dominates — a chase is mostly
"ahead of or behind the rate." Momentum (runs/wkts last 30) adds a small but real
~3%. B1 beats B0 on every metric; the 3 resources already carry most of the signal.

