# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **target_not_met_within_budget**.

Selected preparation: `half_mixed`; steps: 490; final time: 43.125.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered tail window | Mean tail error |
|---|---:|---|---:|
| vacuum | 490 | True | 0.026673361 |
| filled | 490 | True | 0.026673361 |
| left_excitation | 490 | True | 0.02737217 |
| neel | 480 | True | 0.024141968 |
| half_mixed | 490 | True | 0.015682747 |
| profile_mixed | 500 | True | 0.016134973 |

| R | Status | Sampled tail error | Two SE |
|---:|---|---:|---:|
| 64 | mean_target_not_met | 0.03800775 | 0.026749749 |

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
