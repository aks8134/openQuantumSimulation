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
| Matched paths per job | 128 |
| Jobs | 128 |
| Evolution circuits | 638,976 |
| Control circuits | 1,920 |
| Total circuits | 640,896 |
| Evolution shots | 3,145,728 |
| Control shots | 1,572,864 |
| Total shots | 4,718,592 |

A complete path group includes all three bases, raw circuits and all
twirled/folded variants, and is kept within one job. Jobs have their own
controls and calibration. The preparation input uses the recorded
Experiment 2 trajectory stream, seed 29, so the ideal references match
the current scaling study and the separate Aer output check.

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

The provided `model3_phoenix_ready.zip` contains the runnable Python modules,
the default configuration and the frozen 16,384-path input. It contains no
credentials and no submitted hardware jobs. Extract it into a fresh folder;
the paths shown in `RUN_MODEL3.txt` are relative to that folder.

In your existing Qiskit environment on the machine with Phoenix access:

```sh
python FinalModel/Model3Experiment/run.py execute --run FinalModel/Model3Experiment/hardware_runs/model3_full_R16384_S16_20261112
```

For IBM Runtime batch execution, use:

```sh
python FinalModel/Model3Experiment/run.py execute --run FinalModel/Model3Experiment/hardware_runs/model3_full_R16384_S16_20261112 --execution-mode batch --jobs-per-batch 4
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
`SamplerV2(mode=runtime_batch)`. The default remains ordinary `job` mode.
The equivalent configuration fields in `hardware` are:

```json
{
  "execution_mode": "job",
  "jobs_per_batch": 4,
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
prepared input retains 128 paths per job and 128 jobs; with four jobs per
Runtime batch, a fresh full run uses 32 Runtime batches. All measurement
settings, raw/twirled/folded variants and calibration controls remain grouped
exactly as before. This option does not reduce the circuit or shot counts.
The discussed 192-path packing is still a separate configuration choice and
has not been validated for complete live-job acceptance or payload size.

For each group, the code compiles and validates **all** selected jobs before
opening the Runtime batch. It then submits those jobs without waiting for
individual results, records every job ID, and closes the batch to further
submissions before collecting results. Closing does not cancel queued or
running jobs. The next group is compiled after collection, bounding the local
QPY cache to the configured window instead of precompiling the whole study.
Only one job's circuit objects need to be loaded at a time.

The manifest records Runtime batch IDs, member job indices, maximum lifetime,
and closure state. After an interruption, known jobs are recovered by ID,
and an interrupted batch closure is retried before further submission. Jobs
that fail or expire remain recorded; they are never automatically resubmitted.
`--max-jobs` also limits batch mode, so the last Runtime batch may be smaller
than `jobs_per_batch`. No Runtime batch is created for result-only recovery.

Job IDs are saved before result collection. A known job is recovered instead
of resubmitted. A submission interrupted before its ID is saved stops for
explicit reconciliation on IBM; it is never retried automatically. An
exclusive execution lock prevents two controllers submitting the same batch.

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
