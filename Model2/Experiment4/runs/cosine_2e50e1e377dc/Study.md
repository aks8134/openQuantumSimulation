# Model2 Experiment4: cosine preparation search

N=10, J=1, gamma=2, p=0.25, exchange prefactor=0.25; preparation `half_mixed`.
Rate-weighted single-jump dilation, `jump-first`, `even-odd` Hamiltonian factor.
All populations and currents/J: total tolerance 0.02 = mean 0.016 + sampling 0.004.

| Family | Steps | Output time | Output mean error | Forward maximum | Permanent mean bound | Sampled checks |
|---|---:|---:|---:|---:|---:|---|
| cosine | 80 | 106.83222 | 0.015901744 | 0.015901744 | 0.031832036 | not run |
| warped_cosine | 81 | 107.5588 | 0.015393345 | 0.015393345 | 0.031033967 | not run |

The smallest passing count found is a bounded numerical-search result, not a proven global minimum. Shorter failed searches do not establish impossibility. Constant steps use the same model, bounds, objective and accuracy checks.

The forward check directly evaluates every node for at least 10 physical-time units at the final timestep, including the output. These classical continuation steps add no preparation gates. The separate spectral-norm bound, when passing, covers all later ideal mean nodes under contractive rate-weighted steps preserving pI; it is a floating-point bound, not an interval certificate or a many-body state-distance claim.

The target is pI with zero current. Equal-reservoir rate-weighted mean steps preserve it even at finite step, so annealing is not needed to remove a stationary bias. Half-mixture is a nonstationary input when p differs from 0.5. Large preparation steps need not reproduce the physical transient.

| Family | Seed | R | Sampled error | Sampling distance | Maximum two SE | Pass |
|---|---:|---:|---:|---:|---:|---|

Schedule selection uses deterministic means only. The fixed seed list is not reselected. Larger ensembles are tried only if a smaller R fails; every attempted replay is retained. Mixture preparation is analytically averaged. Two standard errors are descriptive, not simultaneous confidence guarantees. Measurement shots, device noise, routing and native compiled depth are outside this experiment.

Reproducible configuration, optimizer trials, complete step lists, means, forward checks and replay summaries are in `study.json`. Figures show actual observables as well as errors.
