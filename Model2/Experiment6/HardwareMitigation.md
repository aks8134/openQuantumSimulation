# Model2 cosine44 hardware mitigation

[run_hardware.py](run_hardware.py) runs the population mitigation comparison on
`ibm_phoenix`, using the frozen Model2 input and the tested Model3 CZ twirling,
folding and statistics helpers. The default is an offline plan. Hardware
submission requires `--execute` and a configured IBM account.

This is a **64-trajectory mitigation pilot**, separate from the earlier
16,384-trajectory raw experiment. It estimates mitigation effects against the
exact same-path ideal populations. It does not establish sufficient trajectories
for 0.02 total observable error, and it does not measure currents.

For the **full 16,384-trajectory** workload, use the separate
[offline job estimator](estimate_jobs.py) described below before choosing
hardware packing. The runner's default remains the small pilot.

## Frozen physics and protocol

| Setting | Value |
| --- | --- |
| System | N=10, J=1, exchange prefactor 1/4, Δ=0 |
| Reservoirs | Γ/J=2 at both ends, pL=pR=0.25 |
| Initial state | Half mixed at every system site; ignored preparation measurements |
| Schedule | 44-step warped cosine, jump first, then even/odd exchange layers |
| Initial/final dt | 4.207281841881579 / 3.760760459292396 |
| Warp power | 16.424973388132297 |
| Total physical time | 183.96811308446507 |
| Trajectory/transpiler seeds | 29 / 11 |
| Trajectories | First 64 independent, prefix-stable seed streams from the Model2 generator |
| Shots | 256 per trajectory per raw arm or twirled noise factor |
| Randomizations | 4 per factor, 64 shots each |
| CZ factors | 1, 1.25, 1.5; 880, 1,100, 1,320 CZs per evolution circuit |
| Intentional resets | All 44 retained at every factor |
| Controls | Zero, one, held-out half mixture; 2,048 shots each per raw/twirled treatment per job |
| Runtime options | Gate/measurement twirling off, DD off, initialization on |
| Bootstrap | 2,000 descriptive paired path/control replicates |

The source is the hash-locked [cosine44 specification](../Experiment5/inputs/cosine44.json).
The two bath qubits remain outside the ten system readouts. Each twirled circuit
has manual Pauli CZ frames and a balanced terminal X mask; analysis decodes that
mask. Selected CZs become CZ·CZ·CZ with barriers. Only one-qubit frame
optimization follows folding, so folded CZ pairs survive compilation.

All six predefined profiles are reported: raw, raw + SPAM, twirled, twirled +
SPAM, linear ZNE, and linear ZNE + SPAM. The fixed ZNE weights are
`[17/6, 1/3, -13/6]`; their shot-SE amplification is about 3.58 relative to one
equally sampled scale. No estimate is clipped or fitted to the target.

Each job has its own raw and twirled zero/one response. SPAM inversion occurs
inside each job before trajectory-weighted averaging. The half mixture is a
held-out check, not a calibration parameter. Response contrast below 0.5 stops
the corrected analysis while preserving the collected counts. The previous
FakeKingston comparison did not demonstrate a clear ZNE improvement; this
hardware pilot preserves the comparisons needed to assess that directly.

## Exact default workload

Each trajectory uses one raw circuit and twelve twirled circuits: four variants
at each of three factors. Each job adds fifteen control circuits.

| Component | Circuits | Shots |
| --- | ---: | ---: |
| Raw evolution, 64 paths | 64 | 16,384 |
| Twirled evolution, three factors | 768 | 49,152 |
| Controls, repeated in two jobs | 30 | 24,576 |
| **Total** | **862** | **90,112** |

The default maximum of 512 circuits per job produces two balanced jobs, each
with 32 paths, 431 circuits and 45,056 shots. Jobs are submitted and collected
sequentially; this runner does not create a Runtime Batch or enqueue both jobs
at once. The circuit cap is configurable and is an implementation choice.

The compiled native programs must fit the configured fraction (default 80%)
of IBM's weighted per-qubit instruction allowance, and the shot count must fit
the execution allowance. These checks include the manually expanded variants
and controls. The counting model follows
[IBM's published validation rules](https://quantum.cloud.ibm.com/docs/en/guides/job-limits).
All jobs are compiled and checked before the first submission. A failing
instruction check requires a new plan with a smaller `--batch-size`; the runner
does not silently repartition an existing run. Passing this public heuristic
does not guarantee provider compiler acceptance.

The shot budget above is exact for these settings. A reliable QPU-time or
dollar estimate requires live compiled durations and the account's charging
terms. The local compilation check is not a paid hardware timing measurement.

## Estimate jobs for the full experiment

For a **Phoenix-specific estimate**, fetch its live target and compile a
sample locally. In PowerShell, reuse the earlier raw run's mapping:

```powershell
python Model2/Experiment6/estimate_jobs.py `
  --backend ibm_phoenix `
  --reference-manifest $runManifest `
  --sample-trajectories 32 `
  --trajectory-count 16384 --shots 32 `
  --output-directory Model2/Experiment6/job_estimates/ibm_phoenix
```

This mode uses the configured IBM account to fetch `ibm_phoenix` with
fractional gates disabled. It constructs the frozen Model2 bodies, transpiles
them on the selected physical chain, and counts all thirteen raw/twirled/folded
variants per sampled trajectory. Controls are compiled once and reserved in
every proposed job. The counting code checks native operation support, fixed
layout, CZ factors and intentional resets against that target.

**Compilation runs locally. No Sampler is constructed and no QPU job is
submitted.** The public execution/instruction allowances stay the same in the
calculation; the per-qubit native footprint now comes from the requested
backend, mapping, compiler settings and calibration snapshot. A backend with
the same CZ basis may give a similar job count; that count is recomputed.

The default sample is `min(32, trajectory-count)`. A 32-path count extrapolated
to 16,384 paths remains explicitly labelled as a backend-specific projection.
For a count of every requested native trajectory, change the compilation size:

```powershell
python Model2/Experiment6/estimate_jobs.py `
  --backend ibm_phoenix `
  --reference-manifest $runManifest `
  --sample-trajectories 16384 `
  --trajectory-count 16384 --shots 32 `
  --output-directory Model2/Experiment6/job_estimates/ibm_phoenix_full
```

Full compilation is substantially more CPU work. It keeps only one native
template and variant at a time rather than retaining a whole experiment's
circuits. The new `compilation_<UTC>/` directory saves the exact Model2 source,
selected path, backend snapshot and a hash-checked `footprints.json`. Numeric
footprint checkpoints are saved every 32 completed paths; incomplete samples
carry `compilation_complete: false`. There are no QPY archives, executable
hardware manifests, job IDs or submissions in this mode. Existing hardware
run directories remain unchanged. The source, snapshot hashes, physical path,
transpiler seed, optimization level, versions and observed backend status
are retained in the profile and report.

Omit `--reference-manifest` to select a calibration-screened path on the named
backend; `--physical-path` fixes an explicit chain. A reference must come from
a matching raw Model2 run on that backend. `--calibration-policy record-only`
allows an explicitly fixed/reference path without quality filtering; native
connectivity and ISA checks remain. The estimator can compile a target while
its backend is in maintenance because it requests no hardware execution.
The current protocol requires native CZ support and refuses unsupported
backends without substituting a different mitigation protocol.

`--account-file`, `--seed`, `--seed-transpiler`, `--optimization-level` and the
runner's coherence/error thresholds configure backend compilation. These
options require `--backend`. A saved `compilation_<UTC>/footprints.json` can
later be supplied to `--footprints` for an entirely offline recalculation.
Counting every trajectory removes sample expansion; the published memory
heuristic still does not prove provider compiler acceptance.

For an **offline FakeKingston baseline**, use:

```powershell
python Model2/Experiment6/estimate_jobs.py --trajectory-count 16384 --shots 32
```

This command works without an IBM account or the large local simulation files. Its
bundled [footprint summary](inputs/fake_kingston_mitigation_footprints.json)
was independently recounted from the saved **32-path FakeKingston** pilot QPY.
It retains individual path vectors, the complete control vector, physical
mapping, frozen specification hash, archive hashes and compiler versions.
The summary's integrity hash is checked before use.

For 16,384 paths, four variants per factor, factors 1/1.25/1.5, 32 shots per
arm/factor and 2,048 control shots per preparation/treatment/job, the saved
[report](job_estimates/JobEstimate.md) gives:

| Packing assumption | Jobs | Shots including controls |
| --- | ---: | ---: |
| Published instruction allowance, no circuit cap | 16 | 2,293,760 |
| 80% instruction allowance, no circuit cap | 20 | 2,342,912 |
| Largest observed footprint at every position, 80% | 21 | 2,355,200 |
| Current runner's 512-circuit cap | 432 | 7,405,568 |

The first three rows are **footprint projections**, not full Phoenix audits.
The largest-observed calculation tests sensitivity to sample variability;
it does not bound circuits absent from the sample. The current-cap row uses
the runner's balanced path grouping and reports whether its instruction and
execution counts pass. Instruction storage is not multiplied by shots.

The estimator writes `job_estimates/JobEstimate.md`, `job_estimate.json` and
per-scenario job CSV files. Each proposed job retains all thirteen variants of
a path, complete job-local controls, and at least two paths. Reports include
a lower bound and state whether the construction attains it for the supplied
counting model. No QPU-time, dollar or statistical-accuracy estimate is made.

To replace the FakeKingston sample with a **fully compiled mitigation** run:

```powershell
python Model2/Experiment6/estimate_jobs.py --manifest $mitManifest --trajectory-count 16384 --shots 32
```

Use the new Experiment6 `$mitManifest`, not the raw Experiment5 `$runManifest`.
The script reads the saved ISA QPY in bounded job-sized pieces, verifies hashes
and variant identities, and counts every compiled path. It does not query the
backend or submit. If that archive contains fewer than the requested 16,384
paths, the report explicitly marks the remaining footprint expansion as a
projection. Distinct observed job controls are reserved through their
componentwise maximum. Missing or incomplete native archives are rejected.

To independently recount the original local FakeKingston pilot or export
another small portable summary:

```powershell
python Model2/Experiment6/estimate_jobs.py --fake-run Model2/Experiment6/runs/fake_zne_f39521cdd46b --export-footprints Model2/Experiment6/inputs/recounted_footprints.json
```

`--footprints PATH` reads an exported summary; `--instruction-fraction`,
`--control-shots`, `--circuit-cap` and `--output-directory` customize the
planning calculation. Outputs must stay outside the input archive directory.

**The estimator does not install the memory-packed partitions into the
hardware runner.** It leaves its 64-path default and 512-circuit cap unchanged,
and never modifies saved manifests, job IDs, source or native programs.
To attempt the full workload in twenty balanced jobs, explicitly raise the
runner's circuit cap as shown below. This uses balanced groups rather than
importing the estimator's greedy CSV partition. Every actual compiled job
must still pass the instruction and execution checks before submission.

### Full workload with twenty balanced jobs

For 16,384 paths and the fixed four-randomization/three-factor protocol,
`--batch-size 10675` fits 820 complete paths and fifteen controls. The runner
balances the paths across twenty jobs: four have 820 paths and sixteen have
819 paths. This gives **213,292 circuits and 2,342,912 shots**, with 2,048
control shots per preparation/treatment/job. The 80% instruction threshold
is enforced on every actual native job, not just on sampled footprints.

Using the Phoenix path from the reported 32-path target compilation:

```powershell
python Model2/Experiment6/run_hardware.py `
  --backend ibm_phoenix `
  --trajectory-count 16384 --shots 32 --seed 29 `
  --randomizations 4 --control-shots 2048 `
  --batch-size 10675 --instruction-fraction 0.8 `
  --physical-path 106 116 115 114 113 112 111 101 91 81 80 90 `
  --execute
```

The runner rechecks the explicit chain against the current calibration and
compiles all jobs before submitting any. If actual instruction counts exceed
the threshold, it stops without submitting and requires a new plan with a
smaller cap. Passing the public counting heuristic does not guarantee IBM
compiler acceptance. These large jobs also require substantially more local
memory than 512-circuit jobs: the native circuit list for one Runtime job
is retained in memory for serialization and submission.

For compilation without subsequent submission, replace `--execute` with
`--compile-only`. Execute that saved plan later with
`python Model2/Experiment6/run_hardware.py --resume "<new manifest.json path>" --execute`.

## Commands from the repository root

Create an offline plan, with no account lookup or backend query:

```powershell
python Model2/Experiment6/run_hardware.py
```

To compare on the physical chain selected for the earlier raw Model2 run,
reuse its manifest through your existing PowerShell `$runManifest` variable:

```powershell
python Model2/Experiment6/run_hardware.py --backend ibm_phoenix --reference-manifest $runManifest --compile-only
```

This imports only the physical mapping from an intact matching raw Model2
manifest. It creates a separate mitigation directory and recompiles against
the requested live backend. No raw job IDs, results or QPY are imported.
The reference must match the backend, seed and frozen Model2 specification.

The command prints the **new mitigation** `manifest.json` path. Save that path
as `$mitManifest`, then execute the saved plan:

```powershell
$mitManifest = "<printed mitigation manifest.json path>"
python Model2/Experiment6/run_hardware.py --resume $mitManifest --execute
```

Alternatively, create, compile and execute in one command:

```powershell
python Model2/Experiment6/run_hardware.py --execute --backend ibm_phoenix --reference-manifest $runManifest
```

Omit `--reference-manifest $runManifest` to select a new calibration-screened
chain. A fixed chain can instead be supplied with `--physical-path`, ordered as
left bath, ten system sites, right bath. New or reference paths are screened
against live T1 ≥ 80 µs, T2 ≥ 60 µs, CZ error ≤ 0.005 and readout error ≤ 0.02.
There are no inherited Model3 qubit exclusions. An explicitly fixed/reference
path may use `--calibration-policy record-only` on a new plan when retaining
that path is required; connectivity and native-ISA checks still apply.

New-plan CLI options allow changing trajectories, shots, randomizations,
controls and batch size. Shots must divide evenly across the randomizations,
and each job must hold a complete control set and at least two matched paths.
Inspect the printed workload whenever changing these settings. Resume uses
saved settings and refuses scientific parameter overrides.

## Recovery and analysis

For a process that was killed, use the cross-experiment
[completion audit](../HardwareRecovery.md) on the original checkout:

```powershell
python Model2/audit_hardware_runs.py --collect-existing --analyze
```

It retrieves only saved jobs reporting `DONE`, backs up the manifest before
saving recovered counts, and produces a labelled completed-job subset analysis
plus resume/inspection commands. It does not submit, retry or cancel jobs.

Each job has a separately checkpointed QPY archive, streamed SHA256, physical
mapping, circuit identities, resource assessment, compiler versions, requested
shots, raw outcome counts, provider job ID and available usage metrics. Large
archives are read through the bounded QPY loader; the entire experiment is not
loaded into memory. The original compilation snapshot is retained alongside a
separate execution snapshot. Backend status is checked before each new
submission, and maintenance blocks submission.

Plain `--resume` is offline. `--resume ... --execute` retrieves a known job ID
before considering pending jobs. An interrupted submission with an unknown ID
stops for reconciliation; it is never automatically repeated.

Inspect a saved failed job without submitting or changing its retry state:

```powershell
python Model2/Experiment6/run_hardware.py --resume $mitManifest --inspect-job <job_id>
```

An explicit retry is available for a saved terminal `ERROR` with code `6056`
and no collected observations, after the backend is active:

```powershell
python Model2/Experiment6/run_hardware.py --resume $mitManifest --execute --retry-failed-job <job_id>
```

The old ID, error and usage diagnostic remain in `failed_attempts`. Other
errors are not automatically classified as safe retries. This does not resolve
the provider's internal compiler fault in the previous raw experiment.

All collected runs are analyzed automatically. To rebuild analysis offline:

```powershell
python Model2/Experiment6/run_hardware.py --resume $mitManifest --analyze-only
```

[analyze_hardware.py](analyze_hardware.py) also accepts a completed run directory.
Outputs are `Analysis.md`, per-site population CSV, JSON with all six profiles
and job-local controls, and population/noise-scaling plots in PNG and PDF.
The raw run contains `hardware_results.json` and `results_bundle.zip`, including
the source, specification, counts, snapshots, QPY archives and analysis.
Analysis requires the hashed source and intact plan; it verifies native QPY
when available and explicitly records missing archives for portable analysis.

## Local validation

Thirteen focused tests cover offline defaults, the exact budget, Model2 source
shape, native CZ/reset counts, masks, QPY integrity, interrupted compilation,
known/ambiguous job recovery, explicit 6056 retries, maintenance, per-job SPAM
inversion, all six profile outputs and reference mapping. Mocked execution
checks that all jobs compile before any submission and ignores preparation
measurement bits.

An additional offline smoke check compiled two Model2 paths against the full
156-qubit FakeKingston target, using its documented pilot chain and relaxed
quality thresholds. All 41 variants/controls passed ISA, layout, reset/fold,
QPY reload and resource validation. Its peak was 53,208 weighted instructions
per qubit and 832 executions. It did not use a live account or submit a job;
its circuits are not intended for Phoenix. The local compiler versions were
Qiskit 2.5.2 and qiskit-ibm-runtime 0.49.0.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_mit_mpl \
  .venv/bin/python -m pytest -q tests/test_model2_hardware_mitigation.py
```

Estimator tests exercise full-run accounting, shot-versus-instruction limits,
singleton handling, lower-bound claims, capped-job failures, profile integrity,
offline execution, raw-manifest rejection and read-only compiled-archive input.
The live-target path uses a mocked service with a real local CZ target, forbids
Sampler construction and matches its vectors exactly against the independently
saved native QPY. Tests also cover backend identity, native CZ requirements,
explicit backend-only settings and invalid requests before account access:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_mit_mpl \
  .venv/bin/python -m pytest -q tests/test_model2_job_estimate.py
```
