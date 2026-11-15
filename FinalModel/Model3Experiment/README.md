# Configurable Model 3 hardware experiment

This experiment runs populations **and normalized bond currents** on
`ibm_phoenix` using the mitigation protocol of the manuscript's existing
Model 3 hardware run. Its implementation reference is
[`Model3/Experiment5/run_mitigation.py`](../../Model3/Experiment5/run_mitigation.py)
and [`mitigation.py`](../../Model3/Experiment5/mitigation.py).
`Model3/Experiment7` is the earlier schedule-optimization study, not the
hardware mitigation runner.

The default [configuration](config.json) uses the exact selected **72-step**
FinalModel schedule, N=10, J=Gamma=1, H=J/2 sum(XX+YY), reservoirs 0.9/0.1,
and initial state `(I/2)^10`. It uses **16,384 paths and 16 shots per path,
basis and raw/twirled noise factor**, as requested. These values are separate
from the statistical overhead introduced by mitigation and controls.

## Preserved mitigation protocol

- The same paths are used in the raw and all mitigated arms.
- Four manually seeded Pauli-twirled variants per factor, including balanced
  terminal X masks decoded in the recorded outcome register. Each variant
  receives four shots, preserving sixteen total per factor and basis.
- Native CZ folding at factors 1, 1.25 and 1.5, with nested selected fold
  subsets and common Pauli frames across factors. Bath resets and intended
  dissipative dynamics are not scaled.
- The same predeclared unweighted linear ZNE intercept, with weights
  17/6, 1/3 and -13/6 for these factors. There is no target fitting or
  selection of extrapolation models after inspecting results.
- Each job contains all-zero, all-one and held-out half-mixture controls
  in Z, each in the raw and terminal-twirled arms. Each preparation/arm
  receives 2,048 shots. Zero and one calibrate the job-local marginal affine
  SPAM response; the mixture is excluded from fitting.
- Runtime's extra twirling and DD are disabled, matching the existing run.
  All data are retained; calibration diagnostics do not select favorable jobs.

There are no additional alternating-bit calibration states or correlated
assignment-matrix mitigation methods in this protocol.

## Currents and readout correction

One Z setting yields all populations. Alternating XY and YX settings yield
all neighbouring-spin correlators and hence I_j/J. The initial preparation
and bath outcomes are sampled by actual circuit execution; they are not
replaced by ideal covariance values.

The original marginal calibration is extended to measured pair parities:
if `E[z_obs,j] = c_j + d_j E[z_true,j]`, use

`E[(z_obs,j-c_j)(z_obs,j+1-c_j+1)] / (d_j*d_j+1)`.

This includes the measured single-site offsets, rather than merely dividing
the current by two calibration contrasts. It assumes the product of the
two local response models. It is **not** a correlated-readout inverse and
does not correct basis-rotation or evolution errors. The held-out mixture
also provides neighbouring Z-parity diagnostics, with expected value zero.
Corrected values are never clipped. Insufficient calibration contrast is
reported, while raw, twirled and uncorrected ZNE results remain available.

## Default workload

| Item | Count |
|---|---:|
| Independent paths | 16,384 |
| Shots per path/basis/arm or noise factor | 16 |
| Measurement bases | Z, XY, YX |
| Matched paths per job (maximum) | 192 |
| Jobs | 86 |
| Jobs per Runtime batch | 8 |
| Runtime batches | 11 |
| Evolution circuits | 638,976 |
| Control circuits | 1,290 |
| Total circuits | 640,266 |
| Evolution shots | 3,145,728 |
| Control shots | 1,056,768 |
| Total shots | 4,202,496 |

A complete path group includes all three bases, raw circuits and all
twirled/folded variants, and is kept within one job. Jobs have their own
controls and calibration. The preparation input uses the recorded
Experiment 2 trajectory stream, seed 29, so the ideal references match
the current scaling study and the separate Aer output check.
The grouping routine balances the 16,384 paths over 86 jobs, so actual jobs
contain 190 or 191 paths under the selected 192-path maximum. The first ten
Runtime batches contain eight jobs each; the last contains six.

The requested S=16 is maintained at each factor. The linear ZNE weights
amplify independent equal-variance shot standard errors by about 3.58
relative to one factor at the same shot count. Thus the unmitigated Aer
sampling check does not certify a 0.004 statistical error after ZNE.
The analysis reports the resulting uncertainties rather than increasing
the chosen shot budget silently.

## Configuration

Edit `config.json` or provide `--config FILE` to `prepare`. Physical
parameters, size, initial diagonal preparation, sampling, path mapping,
compilation and job settings are configurable.

Reference preparation uses the existing fast C++ kernel when available.
On Windows, `sampling.reference_engine="auto"` uses a compiler-free NumPy
implementation with the same C++ MT19937-64 per-path random stream. The
portable implementation is tested against the original kernel, including
paths longer than one generator block. It does not switch to a different
random-number generator.

- `schedule.mode="saved"` reads the validated FinalModel case for the
  configured size/family and requires an exact physics/preparation match.
- `schedule.mode="constant"` uses `steps` and `J_dt`.
- `schedule.mode="warped_cosine"` uses `steps`, `J_tau_min`, `J_tau_max`
  and `q`. The J-scaled steps are converted to physical time with the
  configured J.
- `schedule.mode="manual"` uses an explicit positive `step_dts` list.

Changing physical parameters with a configurable schedule produces fresh
exact digital and stationary references. It does not claim that the new
schedule meets the former acceptance tolerance. The channel remains
first-order, Hamiltonian-first, even/odd, rate-weighted boundary sampling.
XXZ interactions are outside this free-fermion implementation.

The default physical chain is the one used for the current FinalModel
native resource compilation, ordered `[left bath, sites..., right bath]`.
It is configurable and is not asserted to be optimal for today's calibration.
Connectivity and native instructions are checked against the execution target.

## Commands

### Prepared transfer to the hardware machine

The prepared input is
`hardware_runs/model3_full_R16384_S16_P192_20261113`. Its `source.json`,
`paths.npz` and `manifest.json` retain the frozen 16,384 paths and the agreed
192-path job limit, 86 jobs, and eight jobs per Runtime batch. Include these
three files when transferring the repository; generated hardware run folders
are otherwise ignored by Git. The earlier `model3_full_R16384_S16_20261112`
folder records the previous 128-path plan and should not be used for this run.

In your existing Qiskit environment on the machine with Phoenix access:

```sh
python FinalModel/Model3Experiment/run.py execute --run FinalModel/Model3Experiment/hardware_runs/model3_full_R16384_S16_P192_20261113
```

For IBM Runtime batch execution, use:

```sh
python FinalModel/Model3Experiment/run.py execute --run FinalModel/Model3Experiment/hardware_runs/model3_full_R16384_S16_P192_20261113 --execution-mode batch --jobs-per-batch 8
```

Use `--account-file YOUR_ACCOUNT_FILE` if needed. The same command resumes
after an interruption. Both Windows and POSIX execution locks are supported.
The prepared run does not require a C++ compiler or the saved offline target.
Its live target is captured and circuits compiled on the execution machine.
Required Python packages are Qiskit, qiskit-ibm-runtime, NumPy, SciPy and
Matplotlib; the tested environment is Qiskit 2.5.2 / Runtime 0.49.0.

### Create or modify an experiment

Use `.venv-phoenix`, containing the pinned Qiskit 2.5.2 / Runtime 0.49.0
environment. The first command is entirely offline:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel3 \
  .venv-phoenix/bin/python FinalModel/Model3Experiment/run.py prepare
```

It prints the new run directory, freezes the configuration and exact path
choices, records reference errors, and saves a concrete workload plan.

Compile a representative against the saved Phoenix target without any
account or service access:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel3 \
  .venv-phoenix/bin/python FinalModel/Model3Experiment/run.py preview \
  --run RUN_DIRECTORY --paths 1
```

The preview is kept separately and is never used as the live submission.
To execute the prepared run or recover an interrupted collection:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel3 \
  .venv-phoenix/bin/python FinalModel/Model3Experiment/run.py execute \
  --run RUN_DIRECTORY
```

The existing IBM account loader is reused. `--account-file FILE` is optional;
credentials are neither printed nor copied into run artifacts. Execution
captures a live target and compiles native circuits against it. Jobs are
submitted and collected sequentially in ordinary job mode. `--max-jobs N`
limits the number of jobs processed in an invocation, including recovery of
already submitted jobs; another invocation resumes the same manifest.

### IBM Runtime batch execution

`--execution-mode batch` enables actual IBM Runtime `Batch` execution through
`SamplerV2(mode=runtime_batch)`. The agreed default is `batch` mode with
eight jobs submitted together. Ordinary `job` mode remains available.
The equivalent configuration fields in `hardware` are:

```json
{
  "execution_mode": "batch",
  "jobs_per_batch": 8,
  "batch_max_time": 28800
}
```

CLI flags override these submission settings for that invocation. Use
`--execution-mode job` for ordinary sequential execution, and
`--batch-max-time SECONDS` to override the Runtime batch maximum lifetime.
The default is eight hours, subject to the account's service limits.
This lifetime is separate from the existing 300-second per-job execution cap.

`paths_per_job` controls circuit packing within one job. `jobs_per_batch`
controls how many jobs are submitted together into one Runtime batch. The
prepared input uses a 192-path maximum and 86 jobs; with eight jobs per
Runtime batch, a fresh full run uses 11 Runtime batches. All measurement
settings, raw/twirled/folded variants and calibration controls remain grouped
exactly as before. This option does not reduce the circuit or shot counts.
Complete live-job acceptance and payload size have not been validated by
submission. The execution summary prints the frozen packing and resolved
Runtime settings before contacting IBM.

For each group, the code prepares **all** selected compiled caches and verifies
their integrity before opening the Runtime batch. Each job's circuits are
loaded and checked against the live ISA target immediately before submission.
It submits the group without waiting for
individual results, records every job ID, and closes the batch to further
submissions before collecting results. Closing does not cancel queued or
running jobs. The next group is compiled after collection, bounding the local
QPY cache to the configured window instead of precompiling the whole study.
Only one job's circuit objects need to be loaded at a time.

The manifest records Runtime batch IDs, member job indices, maximum lifetime,
and closure state. After an interruption, known jobs are recovered by ID,
and an interrupted batch closure is retried before further submission. Jobs
that fail or expire remain recorded; they are never automatically resubmitted.
Result collection continues through the other submitted jobs in the selected
group before reporting unresolved results. No further group is submitted
while those results remain unresolved.
`--max-jobs` also limits batch mode, so the last Runtime batch may be smaller
than `jobs_per_batch`. No Runtime batch is created for result-only recovery.

### Recovering an IBM Runtime job failure

IBM error 1319 means that a job was out of sync and in an unexpected state
([IBM error registry](https://quantum.cloud.ibm.com/docs/en/errors)).
It does not by itself identify a problem with the submitted circuit.
The runner reports the affected job index, IBM job ID and remote status
in the terminal and `manifest.json`, while saving the results it can
retrieve from the other submitted jobs.

After updating the code, use this PowerShell command to collect existing
jobs without submitting, compiling, or replacing any job:

```powershell
$run = ".\FinalModel\Model3Experiment\hardware_runs\model3_full_R16384_S16_P192_20261113"
python .\FinalModel\Model3Experiment\run.py execute --run $run --collect-only
```

This command can wait for queued/running jobs. It skips results already
collected and leaves unsubmitted groups untouched. If a retrieval fails,
it tries the remaining submitted jobs before returning a nonzero exit code.
Run it again to retry retrieval. A retrieval error does not trigger a new
hardware execution.

If a job remains in `ERROR` or `CANCELLED`, explicitly authorize replacing
that job by its reported integer index. For example, if the report identifies
job 17:

```powershell
python .\FinalModel\Model3Experiment\run.py execute `
  --run $run --execution-mode batch --jobs-per-batch 8 `
  --retry-failed-job 17
```

Use the actual reported index; the last successful output line alone does
not establish which job failed. Repeat `--retry-failed-job INDEX` for several
failed jobs. The command rechecks each job's remote status before changing
the manifest and refuses to replace completed, active, unknown-status, or
already collected jobs. It archives the previous ID and failure details
under `previous_attempts`, then resumes the same frozen circuit groups and
remaining preparation work. Existing counts and all recorded previous IDs
are preserved. Replacement attempts can incur additional usage beyond the
original 86-job plan. After an interruption, resume with the ordinary execute
command; do not repeat the retry flag for a replacement already submitted.

If IBM continues returning 1319, provide the affected job ID to IBM support.
Do not delete or regenerate the run directory or edit job IDs manually.
`--collect-only` and `--retry-failed-job` are mutually exclusive.

Job IDs are saved before result collection. A known job is recovered instead
of resubmitted. A submission interrupted before its ID is saved stops for
explicit reconciliation on IBM; it is never retried automatically. An
exclusive execution lock prevents two controllers submitting the same batch.

### Resume the existing run after the gzip/QPY loading fix

Stop the old runner with Ctrl+C and wait for the prompt before updating its
Python files. Keep the existing run directory, including the locally updated
`manifest.json`, `source.json`, circuit records, caches and any collected
counts. Do not run `prepare` again or replace the Windows manifest with the
original planned manifest from another machine.

Pull the code update and repeat the same command:

```powershell
git pull
python .\FinalModel\Model3Experiment\run.py execute `
  --run .\FinalModel\Model3Experiment\hardware_runs\model3_full_R16384_S16_P192_20261113 `
  --execution-mode batch `
  --jobs-per-batch 8
```

The patch expands each compressed QPY cache **once** into an automatically
cleaned temporary disk file. For indexed QPY formats it reads one circuit
payload at a time and passes that unchanged payload in a one-program QPY
container to the public decoder, avoiding whole-file remainder copies. Older
formats use the regular seekable disk file directly. It never expands the
whole cache into a Python byte buffer. The system temporary drive
needs space for one uncompressed job at a time, which can be several GB.
New cache writing also uses a seekable disk file before streaming compression,
avoiding QPY's whole-job in-memory staging for gzip output.
Precompiled jobs are checked by their sealed metadata and exact QPY-byte hash
without loading circuit objects during the precompilation pass. Objects are
loaded once immediately before that job's submission, and the current live
target is still checked. Original recorded metrics are authenticated by the
cache hash instead of recalculating every circuit depth on each load.

`cache_fix_compatibility.json` identifies the exact original and replacement
hashes for `circuits.py` and `run.py`, including the preceding cache-only
runner revision. The runner accepts only that reviewed
replacement pair, including proven LF/CRLF checkout conversions; all other
source checks remain strict. It records the execution revision in the mutable
manifest while leaving the original frozen source and compiled-record hashes
unchanged. Do not manually edit any hash in `source.json`.

Known job IDs are collected before any new submission. In the reported state,
jobs 0 and 1 are recovered from IBM; the remaining 84 jobs retain their original
paths, readout variants and controls. Existing Runtime batches are closed to
further submissions during recovery, and new work uses new Runtime batches.
If another job ID was saved before interruption, it is recovered as well.

Progress messages distinguish cache verification, one-time decompression,
QPY loading, submission and collection. Submission calls still occur in a
serial client loop; this patch removes the confirmed loading bottleneck and
does not introduce eight concurrent uploads. It makes no new claim about RU
consumption per job.

## Analysis and provenance

After collection:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel3 \
  .venv-phoenix/bin/python FinalModel/Model3Experiment/run.py analyze \
  --run RUN_DIRECTORY
```

The results retain all six methods: raw, raw plus SPAM correction, twirled,
twirled plus SPAM correction, ZNE, and ZNE plus SPAM correction. They report
populations and currents against both the ideal of the executed paths and
the physical stationary target. Exact digital-channel references are also
retained. Corrected methods are marked unavailable if calibration cannot
be inverted stably; the remaining results are preserved.

As before, 2,000 paired trajectory bootstrap resamples are stratified by
observed job, with joint control-bitstring resampling shared across factors.
Evolution shots are not resampled again on top of measured path means.
Intervals are pointwise and descriptive, conditional on observed jobs and
variants; they exclude drift and calibration-model bias.

Raw counts, source/target/code hashes, physical mappings, actual CZ factors,
native metrics and QPY hashes, random seeds and job IDs remain in the run
directory. Native QPY is retained until the corresponding counts and metadata
are saved successfully. By default, the collected QPY cache is then removed
to bound disk use; deterministic construction inputs and compiler/target
provenance are preserved. Set `hardware.retain_native_qpy=true` to keep every
native circuit as well. The representative saved-target preview is retained.
The full experiment would otherwise require roughly tens of GB of native
QPY, based on the actual 72-step preview, exceeding this workspace's free space.
The main
manuscript and previous Model3/FinalModel results are not modified.

Runtime references: [Sampler options](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options),
[IBM Runtime batch execution](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/run-jobs-batch).

## Local checks

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/finalmodel3-tests \
  .venv-phoenix/bin/python -m unittest tests.test_finalmodel_model3experiment -v
```

The tests include exact small-system density-matrix comparison of every
raw/twirled/folded basis distribution, resets, pair correction under asymmetric
local readout response, matched budgets, current signs, account-free planning,
and mocked Runtime interruption/recovery. Batch checks verify precompilation
before opening, all submissions before result collection, job limits,
interrupted closure, and recovery without duplicate submission. They do not
submit hardware jobs.
