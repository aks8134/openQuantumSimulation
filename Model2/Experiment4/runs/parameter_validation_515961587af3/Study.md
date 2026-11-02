# Model2 Experiment4: cosine preparation search

N=10, J=1, gamma=2, p=0.25, exchange prefactor=0.25; preparation `half_mixed`.
Rate-weighted single-jump dilation, `jump-first`, `even-odd` Hamiltonian factor.
All populations and currents/J: total tolerance 0.02 = mean 0.016 + sampling 0.004.

| Family | Steps | Output time | Output mean error | Forward maximum | Permanent mean bound | Sampled checks |
|---|---:|---:|---:|---:|---:|---|
| warped_cosine | 44 | 183.96811 | 0.015732686 | 0.015732686 | 0.024189115 | passed |

The smallest passing count found is a bounded numerical-search result, not a proven global minimum. Shorter failed searches do not establish impossibility. Constant steps use the same model, bounds, objective and accuracy checks.

The forward check directly evaluates every node for at least 10 physical-time units at the final timestep, including the output. These classical continuation steps add no preparation gates. The separate spectral-norm bound, when passing, covers all later ideal mean nodes under contractive rate-weighted steps preserving pI; it is a floating-point bound, not an interval certificate or a many-body state-distance claim.

The target is pI with zero current. Equal-reservoir rate-weighted mean steps preserve it even at finite step, so annealing is not needed to remove a stationary bias. Half-mixture is a nonstationary input when p differs from 0.5. Large preparation steps need not reproduce the physical transient.

| Family | Seed | R | Sampled error | Sampling distance | Maximum two SE | Pass |
|---|---:|---:|---:|---:|---:|---|
| warped_cosine | 29 | 16384 | 0.017412737 | 0.001950975 | 0.0027314743 | True |
| warped_cosine | 30 | 16384 | 0.016117764 | 0.0016589289 | 0.0027334562 | True |
| warped_cosine | 31 | 16384 | 0.018873023 | 0.0031403372 | 0.0027355423 | True |

Schedule selection uses deterministic means only. The fixed seed list is not reselected. Larger ensembles are tried only if a smaller R fails; every attempted replay is retained. Mixture preparation is analytically averaged. Two standard errors are descriptive, not simultaneous confidence guarantees. Measurement shots, device noise, routing and native compiled depth are outside this experiment.

Reproducible configuration, optimizer trials, complete step lists, means, forward checks and replay summaries are in `study.json`. Figures show actual observables as well as errors.
