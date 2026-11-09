# Parallel preparation and interrupted-job recovery

`prepare --workers 4` runs up to four independent trajectory/control groups
at once. Each trajectory includes all 13 matched variants. Only the parent
updates the global manifest. The 24-job production grouping, trajectory order,
physics, shots, seeds, twirling, folds, SPAM controls and DD policy are unchanged.
The default remains one worker. Worker count is an invocation option and may
be changed when resuming a saved plan.

Stop the old Windows preparation with **Ctrl+C**, keep its entire run folder,
and sync the updated code. Use the existing `.venv-dd`; no dependencies change:

```powershell
$ddManifest = "Model2/Experiment8/hardware_runs/mitigation_dd44_20261109T183941_893159Z/manifest.json"
python -u Model2/Experiment8/run_hardware.py prepare --resume "$ddManifest" --workers 4
```

Do not run two preparation commands against the same folder simultaneously.
Preparation contacts the backend to read its current target/calibration but
never submits QPU jobs. After it reports a fully compiled plan, the existing
`submit --manifest "$ddManifest" --hardware` command queues the saved jobs in
Runtime Batch mode.

## What resumes

* Completed jobs retain their recorded hashes and are validated without
  recompilation.
* Newly completed groups have a `.qpy` file and a sealed `.prepare.json`
  receipt. A receipt contains source/compiler context, exact row identities,
  QPY hash/size, checks and native instruction counts. Valid receipts are
  adopted even if the process died before the job's manifest checkpoint.
  This fast recovery performs no transpilation or scheduling.
* Older complete QPY files without receipts receive **one-time full native
  and scheduling validation** against the saved source and seeds. Recovery
  checks every folded/twirled circuit, labelled DD operation, delay, physical
  and classical mapping and global phase. Reset/block timing uses the same
  independent paired-schedule check as fresh preparation. The original QPY
  file is not rewritten. This validation can take substantial CPU time; its
  new receipt avoids repeating that work on later resumes.
* Unfinished `.partial` files are ignored. Only a missing final QPY is compiled
  afresh. A mismatched, corrupted or incompatible complete file causes a
  refusal; it is not silently overwritten.

SDK versions and saved target timing must still match. Calibration policy
remains the saved `record-only` choice for this run. `--workers` does not relax
connectivity, native ISA, mapping, intentional-reset, DD or resource checks.

Receipts are the per-group durable journal; the large global manifest remains
checkpointed at job boundaries. This avoids rewriting a full 219,202-row plan
after every trajectory. A complete QPY written just before a crash but lacking
its final receipt follows the older-file validation path once.

## Worker resources and progress

Each worker holds one bounded group, writes it atomically, and returns only a
small receipt name to the parent. At most `workers` groups are outstanding.
Workers load only the target and the exact compile-only source; backend
services/accounts are never serialized. Backend transpiler defaults, including
Phoenix's `ibm_dynamic_circuits` translation plugin, are passed explicitly to
target-only compilation. Nested Qiskit processes and native library threads
are limited to one per worker.

Progress reports saved trajectories out of 16,834, current job, recovered/new
groups, active workers, elapsed time and an ETA from the session's observed
rate. Early recovery/startup can make that ETA volatile. Long active groups
produce a heartbeat after 20 seconds without a completion.

Press **Ctrl+C once** to stop a parallel run. It stops dispatching groups and
allows already active groups to finish their receipts. A forced termination
also retains complete receipts; in-flight work may have to be repeated. The
same resume command is used afterward, with `--workers 2` if less concurrency
is desired.

The [small four-CPU benchmark](benchmarks/preparation16_four_cpus/Benchmark.md)
found 2.54x speedup with four workers and sampled process RAM around 1.69 GiB.
Those measurements predate production integration and do not include the full
manifest/source, one-time legacy recovery or submission serialization. They
are not a Windows/full-run memory or timing guarantee.
