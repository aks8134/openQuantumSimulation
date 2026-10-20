# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **target_not_met_within_budget**.

Selected preparation: `profile_mixed`; steps: 380; final time: 33.4375.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered tail window | Mean tail error |
|---|---:|---|---:|
| vacuum | 500 | True | 0.039559186 |
| filled | 500 | True | 0.039559186 |
| left_excitation | 500 | True | 0.038517462 |
| neel | 495 | True | 0.024265822 |
| half_mixed | 405 | True | 0.016986646 |
| profile_mixed | 380 | True | 0.016328006 |

| R | Status | Sampled tail error | Two SE |
|---:|---|---:|---:|
| 64 | mean_target_not_met | 0.044126969 | 0.02665286 |

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
