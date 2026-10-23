# Noiseless Aer population check

Saved run: `ibm_phoenix_20261020T190507_427277Z`; same 4 paths, all 10 sites, 72 cosine steps.

## Result

Maximum exact density-matrix versus saved ideal difference over all paths/sites: **2.498e-15**.
Full native circuit conditional-estimator discrepancy: maximum **0.0232322**, largest standardized discrepancy **2.062 SE**.
Exact check below 1e-10: **True**. Native sample check within four nominal SE at every site/path: **True**.

[Averaged populations](populations.png) ([PDF](populations.pdf)); [each trajectory](per_path_populations.png) ([PDF](per_path_populations.pdf)).

| Site | Saved ideal | Aer exact density | Full native Aer conditional estimate | Conditional SE | Saved Phoenix | Physical target |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.585298 | 0.585298 | 0.574082 | 0.006559 | 0.622742 | 0.580000 |
| 2 | 0.472304 | 0.472304 | 0.464520 | 0.005572 | 0.554596 | 0.500000 |
| 3 | 0.539824 | 0.539824 | 0.531056 | 0.005308 | 0.562225 | 0.500000 |
| 4 | 0.522405 | 0.522405 | 0.515413 | 0.004581 | 0.543365 | 0.500000 |
| 5 | 0.534246 | 0.534246 | 0.528295 | 0.004675 | 0.551147 | 0.500000 |
| 6 | 0.475177 | 0.475177 | 0.472370 | 0.004482 | 0.503082 | 0.500000 |
| 7 | 0.550738 | 0.550738 | 0.544969 | 0.004484 | 0.476105 | 0.500000 |
| 8 | 0.515943 | 0.515943 | 0.514133 | 0.004913 | 0.470490 | 0.500000 |
| 9 | 0.489764 | 0.489764 | 0.483242 | 0.005268 | 0.454010 | 0.500000 |
| 10 | 0.380747 | 0.380747 | 0.380649 | 0.005763 | 0.357880 | 0.420000 |

## What was simulated

The original twelve-qubit boundary-bath bodies were regenerated directly from the saved step sizes and jump choices. No new random boundary trajectories were chosen. No hardware account or live backend was accessed.

For the deterministic check, every ignored preparation measurement was replaced by its exact nonselective dephasing channel. Each bath collision was replaced by the exact reduced system channel computed from the actual Qiskit two-qubit gate matrix and the bath's actual initial bit. The bath is immediately reset and never used jointly with the other bath, so this partial trace gives exactly the same system density matrix. Aer propagated the resulting ten-qubit density matrix; no sampled shots or initial-state approximation enter this check.

Independently, the full twelve-qubit bodies were compiled to a compact CZ-connected line using the saved optimization level/transpiler seed. Aer statevector sampling used 512 shots/path, retaining all initial preparation measurements, 72 bath resets, and terminal Z measurements. Simulator-only expectation-value snapshots before final measurement give a lower-variance conditional population estimator; raw measured population counts and their separate shot SEs are also saved. Both estimators average over sampled preparation and reset outcomes. Snapshot values are not hardware-accessible observables and are not extra shots.

## Scope and limits

The original Phoenix ISA QPY files are absent. This validates regenerated logical operations and an offline CZ-line compilation; it does not certify the historical Phoenix compiled circuits or reproduce device calibrations, timings, routing decisions, or errors. The compressed regenerated QPY is saved separately and is not a replacement for the missing original artifact.

The deterministic comparison is against the same four saved paths, not the infinite-trajectory mean or physical stationary target. The latter can differ because of trajectory selection, finite physical time, and timestep bias. Agreement here would support circuit/reference consistency, not establish a unique cause of hardware error.

The bath-channel matrices are copied to contiguous storage before passing them to Aer. In this installed Aer 0.17.2 environment, passing the original strided submatrix views produced an incorrect non-trace-preserving result. Small-system tests against explicit system-plus-bath density matrices detected this; the contiguous-copy implementation passed those checks. This affected the new analysis helper, not the original hardware circuit or saved ideal calculation.

The four-SE sampling criterion is a diagnostic using empirical marginal standard errors, not a certified simultaneous confidence bound. Raw counts and conditional estimators have different variances and must not have their SEs added. There is no device noise model, mitigation, or hardware execution in this analysis.

Per-path JSON files retain counts, conditional covariances, mappings, resource metrics, Aer metadata and timing. Population CSV contains both estimators. All original run files were hash-verified unchanged.
