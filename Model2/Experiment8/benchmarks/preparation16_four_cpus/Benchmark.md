# Preparation benchmark: 16 trajectories, four allocated CPUs

Four workers were fastest in this single matched trial: **78.24 seconds versus
198.84 seconds serially (2.54x)**. Two workers took **114.40 seconds (1.74x)**.
All 223 saved circuit signatures and each group's instruction vectors matched
across modes. This trial used a prototype pool around the shared compilation
kernel. Subsequent [production integration](../../ParallelPreparation.md) adds
`--workers 4` and validated interrupted-group recovery; those additional costs
were not measured by this earlier benchmark.

| Workers | Wall time | Speedup | Peak summed RSS | Peak summed USS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 198.84 s | 1.00x | 358.3 MiB | 354.0 MiB |
| 2 | 114.40 s | 1.74x | 1065.6 MiB | 868.5 MiB |
| 4 | 78.24 s | 2.54x | 1734.6 MiB | 1405.5 MiB |

Each mode compiled the same **16 distinct full 44-step trajectories**, all
13 variants per trajectory, and **15 job-local controls**: 223 circuits. QPY
files were freshly written and hashed, then reloaded and compared outside the
timed interval. Automatic circuit names are excluded from the semantic digest;
instructions, parameters, mappings, global phase, DD labels and metadata match.

The local fixture has 120 qubits and the saved Phoenix path durations and
constraints. Other device connectivity/calibration is synthetic. No IBM
service, QPU or Aer run was used. This measures preparation throughput, not
hardware accuracy or Phoenix compiler acceptance.

Slurm job **57157007** had **four allocated CPU slots, no inherited CPU quota,
and a 4 GiB memory allocation**. Versions: Qiskit 2.6.2, Runtime 0.51.0,
NumPy 2.5.2, Python 3.14.7. Native library threads and nested Qiskit processes
were limited to one per worker. Wall time includes worker startup/imports,
compilation, checks, writes/hashes and shutdown. The JSON records CPU affinity.

The login host exposed eight CPUs but imposed an inherited **0.5-core quota**.
Its preliminary complete serial/partial parallel run was interrupted and excluded.
These results come solely from the completed matched compute-node trial.

RAM figures are sampled maxima of parent plus recursive children. Summed RSS
counts shared pages per process; USS counts private pages. The nominal sampling
interval is 0.2 s, but GIL/sampling delays can miss short peaks; actual sample
counts are recorded. Figures exclude the full 16,834-path manifest/source and
Sampler submission memory. The timing excludes full-manifest checkpoints,
plan creation and provider submission serialization.

Four workers are a useful candidate on a machine with available CPUs and RAM.
This one trial does not establish an optimal worker count or guarantee the same
speedup on Windows, and it is not a measured full-run ETA.

Ten regression tests passed in 115.93 s, including native folded/DD circuits and
resume. Four CPU quota tests were rerun successfully after a Windows adjustment
to the test fixture. See [the protocol](../../ParallelPreparationBenchmark.md)
for the reproducible command. Raw outputs remain under .local-artifacts; the
shareable summary records their full report hash and per-group timings.
