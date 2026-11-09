# Local preparation worker benchmark

The [completed four-CPU trial](benchmarks/preparation16_four_cpus/Benchmark.md)
measured 198.84 s serially, 114.40 s with two workers, and 78.24 s with four
workers (2.54x speedup). All circuit signatures matched.

`benchmark_preparation.py` compares one, two and four worker processes using
the same `preparation.compile_group` function as the production runner. It
does not call IBM, run Aer, submit jobs or modify an existing hardware run.
The benchmark uses isolated workers. Production preparation now also accepts
`--workers 4`; see [ParallelPreparation.md](ParallelPreparation.md) for resumable
production use. The measurements below predate that integration.

Each mode compiles the same 16 distinct seed-29 trajectories, with all 44
steps, bath resets and 13 raw/twirled/folded variants per trajectory. All 15
zero/one/half-mixture control circuits are also included: 223 circuits per mode.
The physics, twirling seeds, factors and selective XX DD policy are unchanged.

The fixture is a synthetic 120-qubit `GenericBackendV2` with the saved Phoenix
path, its gate durations and timing constraints. It includes the saved links
inside that path; other device connectivity and calibration errors are synthetic.
This is not the Phoenix backend target and does not establish Phoenix compiler
acceptance or hardware accuracy. The benchmark measures scaling on the host
that runs it, rather than promising the same speedup on another computer.

Wall time includes worker startup/shutdown, native compilation, transformation,
ALAP scheduling, DD insertion and gate/timing checks, ISA and footprint checks,
QPY serialization and file hashing. At most one trajectory/control group per
worker is in flight. Native library and Qiskit thread/process limits are set to
one per worker to avoid nested parallelism. Process spawning works on Windows
as well as Linux.

After each timed mode, the script reloads public QPY files and verifies the
saved row metadata and native operations. It compares hashes of every circuit's
instructions, physical/classical mapping, parameters, DD labels, global phase
and metadata with the serial result; auto-generated circuit names are excluded.
The comparison itself is outside the timed interval.

Memory measurements use optional `psutil`. Summed RSS includes shared pages
in each process; USS measures private pages. Both sample the parent and its
recursive children at a nominal 0.2-second interval; sampling/GIL delays can
miss short peaks. Reports explicitly mark unavailable
memory measurements when `psutil` is absent.

The report records visible CPUs, affinity and inherited Linux cgroup quotas.
For a run that must have four real CPU slots, add `--require-cpus 4`; compilation
is refused if the known effective capacity is smaller. A visible count of eight
CPUs can coexist with a half-core account quota on a cluster login node. Such
a run cannot establish scaling across four available cores.

Use the existing `.venv-dd` and a fresh, ignored local output directory:

```powershell
python -u Model2/Experiment8/benchmark_preparation.py --trajectories 16 --workers 1 2 4 --output-directory ".local-artifacts/exp8-prep-benchmark16"
```

`--timing-context` can select another saved Experiment8 numeric context without
loading a service or real backend pickle. The directory must not already exist;
every worker mode performs fresh compilation. Progress prints each completed
group and a final wall time/speedup. The full `benchmark.json`, raw QPY files
and synthetic target stay in that output directory.

This small benchmark excludes production plan creation, full-manifest
checkpointing and provider submission serialization. Its throughput should not
be treated as an exact full-run ETA. An existing hardware manifest, its QPY
files and job IDs are never opened or changed by the benchmark.
