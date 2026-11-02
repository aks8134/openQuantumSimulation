# Model2 Experiment4: cosine preparation search

N=10, J=1, gamma=0.5, p=0.25, exchange prefactor=0.25; preparation `half_mixed`.
Rate-weighted single-jump dilation, `jump-first`, `even-odd` Hamiltonian factor.
All populations and currents/J: total tolerance 0.02 = mean 0.016 + sampling 0.004.

| Family | Steps | Output time | Output mean error | Forward maximum | Permanent mean bound | Sampled checks |
|---|---:|---:|---:|---:|---:|---|
| warped_cosine | 22 | 62.750056 | 0.014684415 | 0.014684415 | 0.020099065 | passed |

The smallest passing count found is a bounded numerical-search result, not a proven global minimum. Shorter failed searches do not establish impossibility. Constant steps use the same model, bounds, objective and accuracy checks.

The forward check directly evaluates every node for at least 10 physical-time units at the final timestep, including the output. These classical continuation steps add no preparation gates. The separate spectral-norm bound, when passing, covers all later ideal mean nodes under contractive rate-weighted steps preserving pI; it is a floating-point bound, not an interval certificate or a many-body state-distance claim.

The target is pI with zero current. Equal-reservoir rate-weighted mean steps preserve it even at finite step, so annealing is not needed to remove a stationary bias. Half-mixture is a nonstationary input when p differs from 0.5. Large preparation steps need not reproduce the physical transient.

| Family | Seed | R | Sampled error | Sampling distance | Maximum two SE | Pass |
|---|---:|---:|---:|---:|---:|---|
| warped_cosine | 29 | 16384 | 0.017211248 | 0.0025268338 | 0.0042978789 | False |
| warped_cosine | 30 | 16384 | 0.016157327 | 0.0040448159 | 0.0042951194 | False |
| warped_cosine | 31 | 16384 | 0.020117018 | 0.0057877567 | 0.0043135646 | False |
| warped_cosine | 29 | 32768 | 0.015379679 | 0.0017684899 | 0.0030274678 | True |
| warped_cosine | 30 | 32768 | 0.01616174 | 0.002128671 | 0.0030321779 | True |
| warped_cosine | 31 | 32768 | 0.016840135 | 0.0024776669 | 0.0030342652 | True |

Schedule selection uses deterministic means only. The fixed seed list is not reselected. Larger ensembles are tried only if a smaller R fails; every attempted replay is retained. Mixture preparation is analytically averaged. Two standard errors are descriptive, not simultaneous confidence guarantees. Measurement shots, device noise, routing and native compiled depth are outside this experiment.

Reproducible configuration, optimizer trials, complete step lists, means, forward checks and replay summaries are in `study.json`. Figures show actual observables as well as errors.
