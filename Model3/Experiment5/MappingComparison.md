# Paired mapping evaluation, unchanged 48-step schedule

`run_schedule_comparison.py --comparison mapping` uses the candidate schedule in the existing self-contained input bundle for **both** arms. Baseline means the current physical chain; candidate means the alternative chain. The initial half-mixture, logical operations, exact branch choices, ideal covariance predictions, compilation options and shots agree between the arms. Requested PUB order is randomized within each pair, and every pair remains in one batch. Actual device dispatch order is not verified.

The alternative bath-system-bath chain is:

`49 – 39 – 38 – 48 – 58 – 57 – 56 – 66 – 76 – 86 – 85 – 84`.

This avoids q95/q105 and changes physical roles along the chain. It was selected from the full saved calibration while preserving the current minimum system T1/T2 and worst-readout constraints, as well as overall coherence floors. It is a different choice from the unconstrained CZ-ranked path in the earlier audit. [Selection provenance and metrics](inputs/mapping48_selection.json) record the saved-calibration basis; this is not a guarantee of improved hardware populations. Both chains must pass live calibration filters before any new submission.

| Saved-calibration metric | Current chain | Alternative |
|---|---:|---:|
| Worst CZ error | 0.1919% | 0.1912% |
| Worst system readout error | 1.1475% | 1.0986% |
| Minimum T1, all 12 qubits | 97.85 μs | 174.80 μs |
| Minimum T2, all 12 qubits | 122.99 μs | 128.16 μs |

The two arms overlap physically. This tests the complete layout change, including site/bath roles, rather than isolating one bad qubit or comparing disjoint chips.

## Sampling and workload

The recommended command uses **1024 identical branch paths × 32 shots per path per chain**. Relative to eight shots, this halves conditional measurement standard errors. It does **not** remove the finite-trajectory sampling limit or certify the original total 0.02 budget. The purpose is to test whether changing the physical chain reduces the hardware-to-common-ideal discrepancy. Increasing the trajectory count beyond 1024 is rejected because the supplied bundle contains only 1024 saved paths.

Both arms retain 48 steps, 960 expected CZ gates and 48 bath resets per trajectory. Three preparation controls per chain use 2048 shots each. The total workload is **2054 circuits, 77,824 shots, five sequential pair-preserving batches** with sizes `[512, 512, 512, 512, 6]`. This is a shot/circuit count, not a hardware-time prediction. Using `--shots 8` gives 28,672 total shots with the same circuits and batches.

## Windows PowerShell

After syncing the updated script, run this single line from the repository root:

```powershell
python Model3/Experiment5/run_schedule_comparison.py --backend ibm_phoenix --comparison mapping --alternative-path 49 39 38 48 58 57 56 66 76 86 85 84 --trajectory-count 1024 --shots 32 --batch-size 512 --verify-previous-run Model3/Experiment5/hardware_runs/ibm_phoenix_comparison_20261026T005025_553744Z
```

The previous-run check reads and verifies all five original QPY chunks locally **before connecting to IBM**. It does not submit those old circuits. If the files are missing or their hashes/records disagree, it stops without a new job. Original `circuits_*.qpy` files must be in that previous run's directory on the execution machine. They do not need to be committed to Git.

The runner records both physical paths, current calibration metrics, a calibration snapshot covering their union, software versions, per-batch QPY hashes, native counts, job IDs and raw outcome counts. Resume uses the saved manifest and submits only batches with no known job ID; an ambiguous submission is never automatically retried. Unknown submission IDs require reconciliation with IBM.

New raw files are saved under `hardware_runs/ibm_phoenix_mapping_comparison_<timestamp>/`. Population plots and analysis are saved under `analysis/hardware/<run>/mapping_comparison/`. Transfer the small `results_bundle.zip` for offline analysis; keep QPY chunks locally for circuit verification. The bundle omits QPY intentionally.

The primary outcome is the paired change in discrepancy from the common ideal, alongside aggregate maximum/mean population discrepancies. Also report physical-target error. A signed population shift alone is not necessarily an accuracy improvement. Uncertainty estimates condition on this run and do not include day-to-day drift. If the result is inconclusive, do not describe the alternative as an accuracy improvement.

Validation used local targets and mocked Runtime only: 30 relevant tests passed, including separate native mappings, six controls, identical sources, pair-preserving batches, missing-QPY rejection and unchanged schedule-comparison behavior. No hardware execution was performed while preparing this workflow.
