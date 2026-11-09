# Measured Aer benchmark: 256 trajectories × 32 shots

**The real Aer run completed: 768 circuits and 24,576 actual final-measurement outcomes.**
This is a timing and memory benchmark; the full 16,384-trajectory configuration has not been run.

| Quantity | Measured result |
| --- | ---: |
| End-to-end wall time, including imports, source construction, circuit construction and output | 394.484 s (6.575 min) |
| Aer chunk execution, decoding and checkpoint time | 365.100 s |
| Import time | 11.623 s |
| Peak process resident RAM | 202.586 MiB |
| CPU thread limit | 4 |
| Total CPU time, user + system | 177.621 s |
| Average CPU use, CPU seconds / wall seconds | 0.450 cores |
| Completed checkpoints | 16, each containing 16 trajectories |
| Chunk duration, minimum / median / maximum | 21.914 / 22.694 / 24.119 s |

Peak RAM is Linux `getrusage(RUSAGE_SELF).ru_maxrss`, covering the Python process and its Aer threads. It is a measured peak for this 256-path run, not a measured full-run peak. The wrapper measured elapsed time with a monotonic clock. CPU time includes all process threads. The environment exposed eight logical CPUs; Aer was capped at four threads, with one circuit executing at a time and up to four shots in parallel. NumPy/BLAS thread limits were one.

Settings: the frozen Model2 44-step cosine schedule, ten system qubits, half-mixed preparation, trajectory seed 29, simulator seed 2026, bases Z/XY/YX, 32 shots per basis, 16 trajectories per chunk. Software: Qiskit 2.5.2, Aer 0.17.2, NumPy 2.5.2. This uses the previously verified exact bath-channel reduction with stochastic preparation and channel outcomes. Outcomes came from actual Aer statevector executions and final measurements. No hardware noise model, mitigation or calibration controls were included; no IBM service was contacted.

For 16,384 trajectories at the same 32 shots and four-thread setting, scaling the measured scientific workload by 64 while keeping imports fixed gives **6.810 hours**. This is an extrapolation, conditional on comparable CPU load and per-path cost. The full run would contain 49,152 circuits and 1,572,864 final outcomes. Fixed-size circuit chunks keep circuit/state working memory bounded, while source and result arrays grow with the path count; full-run RAM is not established by this benchmark.

The observed maximum discrepancies were 0.0291748047 against the stationary physical target, 0.0181055501 against the infinite-path mean, and 0.0138746446 against the same 256-path ideal. The original empirical checks returned **False**. These 256 trajectories are a performance benchmark, not the recommended precision ensemble. One simulator realization does not estimate a repeated-experiment failure probability.

All 16 sample-archive hashes, study and source hashes, bit ranges, shot counts and path coverage were checked. Population and current estimates were independently reconstructed from the saved packed measurement bits and exactly matched the saved analysis.

Compact machine-readable results: [aer_benchmark_256_summary.json](aer_benchmark_256_summary.json).
Raw study: `Model2/Experiment5/runs/benchmark_256x32_4threads_3825ec14f2/samples/aer_d2cbc7ef84e0/study.json`.
Raw benchmark: `Model2/Experiment5/runs/benchmark_256x32_4threads_3825ec14f2/benchmark.json`.
Start/finish UTC: 2026-11-08T04:08:31.538523+00:00 / 2026-11-08T04:15:06.013988+00:00.

## Four versus eight threads

A second actual Aer benchmark repeated all 256 trajectories, all three bases and 32 shots per basis with an eight-thread limit. Physics, trajectory seed, simulator seed, chunk size, source hash, scientific code hash and software versions matched the four-thread run. Only the Aer/OpenMP thread limit changed; NumPy/BLAS remained limited to one thread.

| Measurement | Four threads | Eight threads |
| --- | ---: | ---: |
| End-to-end wall time | 6 min 34.5 sec | 6 min 40.2 sec |
| Scientific wall time, excluding imports | 382.850 s | 391.666 s |
| Aer chunks, decoding and checkpoint time | 365.100 s | 374.846 s |
| Peak process RAM | 202.586 MiB | 197.309 MiB |
| Total CPU time | 177.621 s | 184.425 s |
| Average CPU use, CPU time / wall time | 0.450 cores | 0.461 cores |
| Actual final outcomes | 24,576 | 24,576 |
| Projected full 16,384-path wall time | 6.810 h | 6.966 h |

The **4-thread run was faster by a factor of 1.015** in this comparison. The wall-time reduction on changing from four to eight threads was -1.45%. This is one sequential run per setting, so it measures observed performance under the respective system loads; it does not establish a universal thread-scaling factor. A requested thread limit does not imply that the same number of CPU cores remained busy throughout the run.

All 16 sample archives in each run passed hash, shape, range, shot-count and trajectory-coverage checks. Populations and currents were reconstructed from the saved packed measurement bits and exactly matched each run's saved results. Every saved chunk's Aer metadata reported the requested parallel-shot count (four or eight). Simulator bitstrings need not match when thread counts change.

The eight-thread benchmark's maximum errors were 0.0291748047 against the stationary target, 0.0181055501 against the infinite-path mean, and 0.0138746446 against its same-path ideal. Its original empirical checks returned **False**; 256 paths remain a performance benchmark rather than the full precision ensemble. No full 16,384-path run or hardware job was started.

Results: [eight-thread summary](aer_benchmark_256_8threads_summary.json), [comparison JSON](aer_benchmark_256_thread_comparison.json).
Eight-thread raw study: `Model2/Experiment5/runs/benchmark_256x32_8threads_691e142b24/samples/aer_0094aa6fe582/study.json`.
Eight-thread start/finish UTC: 2026-11-08T04:22:54.126170+00:00 / 2026-11-08T04:29:34.327009+00:00.
