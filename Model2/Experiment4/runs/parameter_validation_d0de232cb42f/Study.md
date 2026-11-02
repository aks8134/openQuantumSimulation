# Model2 Experiment4: cosine preparation search

N=10, J=1, gamma=0.392699, p=0.25, exchange prefactor=0.25; preparation `half_mixed`.
Rate-weighted single-jump dilation, `jump-first`, `even-odd` Hamiltonian factor.
All populations and currents/J: total tolerance 0.02 = mean 0.016 + sampling 0.004.

| Family | Steps | Output time | Output mean error | Forward maximum | Permanent mean bound | Sampled checks |
|---|---:|---:|---:|---:|---:|---|
| constant | 20 | 62.831853 | 0.015625 | 0.015625 | 0.015625 | passed |

The smallest passing count found is a bounded numerical-search result, not a proven global minimum. Shorter failed searches do not establish impossibility. Constant steps use the same model, bounds, objective and accuracy checks.

The forward check directly evaluates every node for at least 10 physical-time units at the final timestep, including the output. These classical continuation steps add no preparation gates. The separate spectral-norm bound, when passing, covers all later ideal mean nodes under contractive rate-weighted steps preserving pI; it is a floating-point bound, not an interval certificate or a many-body state-distance claim.

The target is pI with zero current. Equal-reservoir rate-weighted mean steps preserve it even at finite step, so annealing is not needed to remove a stationary bias. Half-mixture is a nonstationary input when p differs from 0.5. Large preparation steps need not reproduce the physical transient.

| Family | Seed | R | Sampled error | Sampling distance | Maximum two SE | Pass |
|---|---:|---:|---:|---:|---:|---|
| constant | 29 | 16384 | 0.019744873 | 0.0049133301 | 0.0066523669 | False |
| constant | 30 | 16384 | 0.019470215 | 0.0048522949 | 0.0066553182 | False |
| constant | 31 | 16384 | 0.022735596 | 0.0071105957 | 0.0066695684 | False |
| constant | 29 | 32768 | 0.018356323 | 0.0043487549 | 0.00469568 | False |
| constant | 30 | 32768 | 0.018554688 | 0.0035247803 | 0.0046964954 | False |
| constant | 31 | 32768 | 0.020309448 | 0.0046844482 | 0.0047055158 | False |
| constant | 29 | 65536 | 0.016242981 | 0.0041732788 | 0.0033103329 | False |
| constant | 30 | 65536 | 0.016937256 | 0.0038375854 | 0.0033152285 | True |
| constant | 31 | 65536 | 0.018493652 | 0.0028686523 | 0.003319657 | True |
| constant | 29 | 131072 | 0.01833725 | 0.0027122498 | 0.0023486447 | True |
| constant | 30 | 131072 | 0.017353058 | 0.0028495789 | 0.0023454258 | True |
| constant | 31 | 131072 | 0.016788483 | 0.0011634827 | 0.0023429192 | True |

Schedule selection uses deterministic means only. The fixed seed list is not reselected. Larger ensembles are tried only if a smaller R fails; every attempted replay is retained. Mixture preparation is analytically averaged. Two standard errors are descriptive, not simultaneous confidence guarantees. Measurement shots, device noise, routing and native compiled depth are outside this experiment.

Reproducible configuration, optimizer trials, complete step lists, means, forward checks and replay summaries are in `study.json`. Figures show actual observables as well as errors.
