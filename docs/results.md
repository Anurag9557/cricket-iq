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

## Calibration (fit on val 2024, evaluated on test 2025–26)

| variant | Brier | log-loss | ECE |
|---------|------:|---------:|----:|
| **raw B1** | **0.1144** | **0.3520** | **0.0261** |
| isotonic | 0.1161 | 0.3575 | 0.0355 |
| Platt | 0.1177 | 0.3714 | 0.0461 |

Raw LightGBM is already well-calibrated (ECE 2.6%). BOTH explicit calibrators made
held-out calibration WORSE — they overfit the 2024 validation season's miscalibration,
which didn't transfer to 2025–26 (temporal distribution shift). Decision: ship raw B1,
no post-hoc calibration. Reliability curve: docs/reliability.png.

## B1 per-phase (TEST 2025–26)

| phase | n | Brier | log-loss | AUC |
|-------|--:|------:|---------:|----:|
| powerplay (1–6) | 50,211 | 0.1572 | 0.4699 | 0.8536 |
| middle (7–15) | 69,683 | 0.1038 | 0.3262 | 0.9353 |
| death (16–20) | 26,091 | 0.0601 | 0.1941 | 0.9775 |
| overall | 145,985 | 0.1144 | 0.3520 | 0.9226 |

Model confidence rises through the innings (Brier 0.157→0.060, AUC 0.85→0.98).

## Feature-group ablation (TEST)

| features | Brier | AUC |
|----------|------:|----:|
| full (11) | 0.1144 | 0.9226 |
| − momentum (9) | 0.1146 | 0.9222 |
| resources-only, LightGBM (3) | 0.1198 | 0.9165 |
| resources-only, B0 logistic (3) | 0.1204 | 0.9147 |

~90% of the B0→B1 gain came from ADDED FEATURES (0.1198→0.1144), not the model
(logistic→tree on the same 3 was only 0.1204→0.1198). Momentum adds ~0.0002 — nearly
redundant. The 3 resources are already strongly (near-linearly) informative.
