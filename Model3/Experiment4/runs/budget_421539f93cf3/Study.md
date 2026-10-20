# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **mean_target_met**.

Selected preparation: `half_mixed`; steps: 490; final time: 43.125.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered tail window | Mean tail error |
|---|---:|---|---:|
| half_mixed | 490 | True | 0.015682747 |

| R | Status | Sampled tail error | Two SE |
|---:|---|---:|---:|
| 64 | sampling_budget_not_met | 0.03800775 | 0.026749749 |
| 4096 | sampling_budget_not_met | 0.016521947 | 0.0027493578 |

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
