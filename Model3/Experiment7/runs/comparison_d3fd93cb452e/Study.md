# Experiment7: automatic two-stage feedback versus cosine annealing

Total tolerance 0.02; sampling allowance 0.004.
Both methods plan independently of trajectory seeds and R. No earlier schedules are loaded.

| Method | Steps | Output time | Forward mean max | Planning seconds |
|---|---:|---:|---:|---:|
| two_stage (mean_target_met) | 73 | 10.776081 | 0.0158263033 | 91.150 |
| cosine (mean_target_met) | 72 | 22.809543 | 0.0159253878 | 82.216 |

| Method | Replay | Status | Sampled output error | Sampling distance | Two SE |
|---|---|---|---:|---:|---:|
| two_stage | seed29_R8192 | passed | 0.0160325878 | 0.00323109949 | 0.00237792795 |
| two_stage | seed30_R8192 | passed | 0.0159577773 | 0.0014968217 | 0.00237330854 |
| two_stage | seed31_R8192 | passed | 0.0169250787 | 0.00166113818 | 0.00237807252 |
| cosine | seed29_R8192 | passed | 0.0171097482 | 0.00272229504 | 0.00268756283 |
| cosine | seed30_R8192 | passed | 0.0174284472 | 0.00235330039 | 0.00266951234 |
| cosine | seed31_R8192 | passed | 0.0168577158 | 0.00340830102 | 0.00264118323 |

Search jointly selects preparation length and timestep parameters with seeded differential evolution. The two-stage method replans after short committed prefixes and retains its own feasible continuation. Cosine selects its two endpoints and length, then follows that schedule, allowing early stopping only when the same output checks pass.

Search observes a sparse, exactly advanced future window; every retained candidate and actual output is separately checked by direct mean evolution at every future integration node. Globally minimal depth is not established. The schedule search does not compile circuits; separately generated verification and resource measurements are in [VerifiedComparison.md](VerifiedComparison.md).

Forward continuation uses the SAME final timestep for at least the configured physical-time window. It is classical work and is excluded from preparation steps. Sampling is checked at output only. Two-SE values are descriptive, not simultaneous confidence bounds. Initial mixture bits and measurement shots are not sampled; hardware noise is excluded.
