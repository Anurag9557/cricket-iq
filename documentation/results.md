# Results — CricketIQ: predictive models & verified agent

Test: 2025–26, 145,985 held-out chase-deliveries (split by match — see leakage-audit.md).
Train ≤2023 (545,310 rows) · Val 2024 (91,247). Metrics via the verified harness
(tests/test_eval.py). Lower Brier/log-loss/ECE better; higher AUC better.

## Model ladder (TEST 2025–26)

| model | features | Brier | log-loss | AUC | ECE |
|-------|----------|------:|---------:|----:|----:|
| no-skill (base rate) | — | 0.2499 | ~0.693 | 0.500 | — |
| B0 logistic | wkts_in_hand, balls_remaining, required_rr | 0.1204 | 0.3717 | 0.9147 | 0.0336 |
| **B1 LightGBM** (shipped) | full 11-feature state | **0.1144** | **0.3520** | **0.9226** | **0.0261** |
| M2 GRU (causal) | same 11 features, sequential | 0.1142 | 0.3515 | 0.9231 | 0.0295 |

B0 cuts Brier 52% below no-skill and reaches 0.915 AUC with just three "resource"
features (the DLS intuition: wickets + balls + required rate).

## B0 coefficients (scaled, per std-dev) — sanity check
- required_rr    −3.61  → higher rate needed ⇒ less likely to win (dominant)
- wickets_in_hand +1.25 → more wickets ⇒ more likely
- balls_remaining −1.28 → at FIXED required rate, more balls = longer chase to
  sustain = harder ⇒ lower win prob (interpret holding required_rr constant)

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

Approximately 90% of the B0→B1 Brier improvement is attributable to the additional
features rather than the model family: features moved it 0.1198→0.1144, while
logistic→tree on the same 3 features moved it only 0.1204→0.1198. Momentum adds ~0.0002
— nearly redundant. The 3 resources are already strongly (near-linearly) informative.

M2 (sequence model) — a rigorously-run negative result. A causal GRU (hidden 64, per-ball masked BCE, early-stopped on val), fed the identical 11 features as B1, matches LightGBM to within sampling noise (Brier 0.1142 vs 0.1144, AUC 0.9231 vs 0.9226) and is slightly worse calibrated (ECE 0.0295 vs 0.0261). Because the 145,985 test deliveries come from only 1,329 independent matches, differences this small are unlikely to be practically meaningful. Interpretation: the engineered state (required run-rate, rr_diff, rolling last-30 momentum) already encodes the sequential signal, so recurrent memory over ball-order adds nothing — the same conclusion as the Phase-2 feature ablation, reached via a different model family. B1 remains the model of record; M2 is the deep-learning control that proves the simpler model is enough.

## Verified agent — verification OFF vs ON (Phase 5)

The agent answers in natural language but obtains every numeric statistic from deterministic tools rather than generating it itself: a verifier re-checks every number in the answer against the tool returns, and a repair loop re-grounds any that don't trace to a tool — falling back, if repair is exhausted, to a response that states only verified figures. This measures the **verifier's value**: the *ungrounded-statistic rate* (answers stating a number no tool produced) with verification OFF vs ON, over a mixed 18-question benchmark (simple lookups, comparisons that tempt the model to compute, and out-of-scope edge cases). Each question runs once through the verify+repair loop — its pre-repair answer is OFF, its final verdict is ON.

| verification | ungrounded-stat rate | resolution breakdown |
|--------------|---------------------:|----------------------|
| **OFF** (tools only) | **33%** (6 / 18) | — |
| **ON** (verify + repair) | **0%** (0 / 18) | 12 clean · 5 repaired · 1 safe-fallback |

The benchmark deliberately mixes questions that succeed on a plain tool lookup with questions that tempt the model into unsupported arithmetic (comparisons, "by how much…"), so the OFF rate reflects realistic usage rather than a set built to maximise failures.

Every OFF failure was one of those comparison/derivation questions, where the model introduced a derived numeric value (a difference or a percentage) that no tool returned directly. In this evaluation the verify+repair loop reduced the ungrounded-stat rate from 33% to 0%: repair re-grounded five of the six, and the sixth — which the model kept re-asserting — was replaced by a fallback that states only verified figures, so no ungrounded number reached the final answer. All six ungrounded numeric claims in this benchmark were detected before the response. The 33% OFF figure depends on the question mix (~28% comparisons) and is a single run. Note this measures *numeric grounding* — whether each stated number traces to a tool — not overall factual correctness or answer quality; a 0% ungrounded rate does not mean the agent is never wrong.

*Separately, on a golden set of specific split questions the grounded agent is 12/12 correct and every number verified, while the same model without tools never produces a correct player split — that measures grounding's value; the table above measures the verifier's.*