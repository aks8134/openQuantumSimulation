# Experiment4: budget-constrained steady-state search

Step budget per trajectory: 500. Search status: **mean_target_met**.

Selected preparation: `half_mixed`; steps: 200; final time: 21.2125.

Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.

The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.

| Preparation | Steps | Covered tail window | Mean tail error |
|---|---:|---|---:|
| half_mixed | 200 | True | 0.015080872 |

Forward mode: the mean is validated after output using the same final timestep. This continuation is classical planning work and is excluded from preparation depth. Sampling checks apply only to the output state.
Classical continuation: 320 steps; validation ends at physical time 31.2125.

Plan only: no sampled trajectories were replayed; this is not a sampling-validated success.

| R | Status | Sampled tail error | Two SE |
|---:|---|---:|---:|

Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof.
