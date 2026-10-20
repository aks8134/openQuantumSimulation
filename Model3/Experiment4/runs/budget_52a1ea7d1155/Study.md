# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **mean_target_met**.

Selected preparation: `half_mixed`; steps: 117; final time: 22.65.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered mean window | Mean window error |
|---|---:|---|---:|
| half_mixed | 117 | True | 0.015922293 |

Forward mode: the mean is validated after output using the same final timestep. This continuation is classical planning work and is excluded from preparation depth. Sampling checks apply only to the output state.
Classical continuation: 200 steps; validation ends at physical time 32.65.

| R | Status | Sampled output error | Two SE |
|---:|---|---:|---:|
| 8192 | passed | 0.017116024 | 0.0023250806 |

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
