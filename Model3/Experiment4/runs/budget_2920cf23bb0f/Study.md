# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **target_not_met_within_budget**.

Selected preparation: `half_mixed`; steps: 394; final time: 31.125.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered tail window | Mean tail error |
|---|---:|---|---:|
| vacuum | 496 | True | 0.043056275 |
| filled | 496 | True | 0.043056275 |
| left_excitation | 488 | True | 0.032575963 |
| neel | 498 | True | 0.02607373 |
| half_mixed | 394 | True | 0.018264661 |
| profile_mixed | 394 | True | 0.01833976 |

| R | Status | Sampled tail error | Two SE |
|---:|---|---:|---:|
| 64 | mean_target_not_met | 0.042324118 | 0.026724485 |

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
