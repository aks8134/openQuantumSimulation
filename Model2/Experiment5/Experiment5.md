# Experiment5: Model2 cosine44 on IBM hardware

This experiment freezes the **44-step warped cosine at Gamma/J=2**, selected
and validated in [Experiment4](../Experiment4/ParameterStudy.md). It builds
raw hardware circuits, retains matched finite-path ideal predictions, and
measures populations and excitation currents. **The default invocation is
an offline plan.** Only `--execute` submits jobs. No hardware execution has
been performed as part of implementing this experiment.

**Aer follow-up:** the 1024x8 default pilot fails the original precision
checks in noiseless Aer (maximum error 0.02563). For a precision study with
the fixed seed-29 ensemble, use the 16384x32 command below. That larger
recommendation is supported by an ideal-shot probability calculation,
not a completed full large Aer replay. See [the Aer evidence and
statistical scope](AerValidation.md).

## Frozen model and schedule

| Parameter | Value |
|---|---|
| System / bath qubits | 10 / 2 |
| Hamiltonian | `H=(J/4) sum(XX+YY)`, Delta=0 |
| J / Gamma | 1 / 2, total Gamma at each boundary |
| Left / right reservoir populations | 0.25 / 0.25 |
| Initial preparation | `(I/2)^tensor10`, prepared by H then ignored Z measurement |
| Stationary populations / excitation currents | 0.25 at every site / zero on every bond |
| Preparation steps | 44 |
| Order | selected boundary collision, even bonds, odd bonds |
| Jump probabilities, left pump/loss then right pump/loss | 0.125, 0.375, 0.125, 0.375 |
| Initial / final dt | 4.207281841881579 / 3.760760459292396 |
| Warped exponent q | 16.424973388132297 |
| Total simulated preparation time | 183.96811308446507 |

```
dt_k = h_min + (h_max-h_min)/2 * [1+cos(pi*(k/43)^q)]
k = 0,...,43
```

The portable [input specification](inputs/cosine44.json) contains all 44
full-precision steps, the original source hash and the three passing
R=16384 replay summaries. A canonical digest locks this exact specification;
reformatting JSON or Windows line endings does not change its meaning.
The runner does not refit the schedule or require Experiment4's ignored
local JSON files. Model parameters are fixed rather than exposed as
hardware-run overrides.

Both jump and Hamiltonian factors use `XXPlusYYGate`: system parameter
`4*(1/4)*J*dt = dt` and bath parameter `2*sqrt(2*Gamma*dt)`.
Pumping first excites the selected zero bath; loss uses a zero bath. The
active bath is reset after every collision, for 44 bath resets per path.
Negative bath cosines are retained. The simulation times set rotation
angles, not delays or hardware microseconds. Large steps prepare a target
output; they do not claim accurate continuous-time transients.

## First hardware pilot

From the repository root, with the project's Python environment activated,
first inspect the fully offline plan:

```powershell
python Model2/Experiment5/run_hardware.py
```

Run the pilot on the same backend used for the Model3 work:

```powershell
python Model2/Experiment5/run_hardware.py --execute --backend ibm_phoenix
```

The same commands work in Bash with `.venv/bin/python` in place of `python`.
The account loader uses `IBMRuntime/apikey.json` if present, otherwise the
saved Qiskit account. Supply `--account-file PATH` to choose a different
account file. Credentials are never copied into result artifacts.

The explicit form of the pilot command is:

```powershell
python Model2/Experiment5/run_hardware.py --execute --backend ibm_phoenix --trajectory-count 1024 --seed 29 --shots 8 --bases z xy yx --batch-size 512 --control-shots 2048
```

| Pilot workload | Count |
|---|---:|
| Fixed trajectories, seed 29 | 1024 |
| Shots per trajectory per basis | 8 |
| Evolution circuits, three bases | 3072 |
| Controls, three per job | 21 |
| Runtime jobs | 7 |
| Total circuits | 3093 |
| Total shots including controls | **67584** |

`z` measures all populations. Alternating `xy` and `yx` cover both Pauli
correlators on every bond, with
`I_j=(J/4)(<Y_j X_(j+1)>-<X_j Y_(j+1)>)`. `--bases z` is supported as an
explicit population-only alternative; the analysis does not invent currents.
The `preparation` classical register is ignored; only the final `outcome`
register is decoded.

**This is a pilot, not a transfer of the original 0.02 validation.** The
first 1024 seed-29 paths have ideal maximum target error **0.0201363873**,
sampled-minus-mean distance **0.00680954294**, and maximum descriptive two SE
**0.0111872496**. They do not pass the original 0.004 sampling allocation.
All these values are printed and saved before any backend access. The
pilot tests hardware agreement with its **own exact saved-path ideal**,
alongside the physical stationary target and infinite-path mean. More
shots per path cannot fix finite-path bias. The subset and seed are fixed;
there is no seed selection or cherry-picking hardware trajectories.

For the full seed-29 path ensemble with the shot count recommended by the
ideal precision calculation, explicitly use:

```powershell
python Model2/Experiment5/run_hardware.py --execute --backend ibm_phoenix --trajectory-count 16384 --seed 29 --shots 32 --bases z xy yx --batch-size 512 --control-shots 2048
```

That configuration is much larger: **97 jobs, 49443 circuits, 2168832 shots**
including controls. Its classical path ensemble passes the saved empirical
checks. The conditional ideal-shot bound covers the final-shot uncertainty
jointly across all populations and currents; device bias, drift and shot
correlations remain outside that calculation. No hardware accuracy
certificate is inherited. Each path's RNG stream is independent and has
the same prefix when the path count increases. The smaller pilot defaults
are retained; this larger workload requires an explicit command.

## Mapping, controls and live compilation

The runner chooses a fresh CZ-connected 12-qubit chain from the requested
backend's calibration, using the existing tested path-ranking helper. Chain
order is left bath, ten system sites, right bath. Default screens are
T1>=80 microseconds, T2>=60 microseconds, CZ error<=0.005 and readout
error<=0.02. No historical qubit is excluded automatically. `--avoid-qubits`
and the screen flags are explicit overrides. If no qualifying chain exists,
the runner fails before submission and does not silently relax the limits.
`--physical-path` supplies exactly twelve ordered qubits, still checked
against the requested calibration screens and connectivity.

Every path's three bases remain in one job. A seeded shuffle sets path
order. Each job includes all-zero, all-one and held-out half-mixture Z
controls on the same system qubits, requested near its beginning, middle
and end. Actual dispatch timing is not guaranteed. The analysis compares
the half-mixture with `(zero+one)/2` inside that job and retains individual
profiles. No SPAM inversion, clipping or automatic correction is applied.
Runtime DD and gate/measurement twirling are disabled; initialization is
enabled. This is the raw first hardware evaluation of the chosen schedule.

The evolution body is compiled once per path for the **live** target.
Readout rotations are appended at its verified final mapping; only the
required native RZ/SX/readout operations are appended. The fixed path,
all 44 resets, native instruction support, final measurement bits and
circuit identities are checked before submission. Local fake-target
circuits are never submitted to a different backend. Calibration scores
rank chains; they do not predict observable error or certify accuracy.

To inspect live compilation without submitting jobs:

```powershell
python Model2/Experiment5/run_hardware.py --compile-only --backend ibm_phoenix
```

This **does access the account and live backend**, unlike the default
offline plan. It prints the saved manifest path. All circuits are compiled
and hashed before the first submission. A representative local compilation
is documented separately in [OfflinePreview.md](OfflinePreview.md).

## Resume, analysis and transfer

For an interrupted process, first inventory both Model2 hardware experiments
and collect already finished jobs on the original machine:

```powershell
python Model2/audit_hardware_runs.py --collect-existing --analyze
```

This creates a manifest backup before recovered counts are saved, performs
no submissions, and analyzes complete collected jobs even if the full run is
unfinished. Read `Model2/hardware_audit/Audit.md` for exact saved IDs, queried
states and resume commands. See [HardwareRecovery.md](../HardwareRecovery.md)
for incomplete-cohort interpretation and failed/ambiguous submissions.

The runner prints its unique run directory under
`Model2/Experiment5/hardware_runs/cosine44_<UTC timestamp>/`. After an
interruption, substitute the printed manifest path:

```powershell
python Model2/Experiment5/run_hardware.py --resume Model2/Experiment5/hardware_runs/cosine44_<timestamp>/manifest.json --execute
```

### Recover a completed compilation after a hashing MemoryError

The original runner hashed `isa_circuits.qpy` with `Path.read_bytes()` after
writing every circuit. On large ensembles this allocated the whole archive
in RAM alongside the compiled circuits. That failure occurs **before any job
submission**. The QPY write has already completed at that point.

After updating `run_hardware.py` and adding `qpy_storage.py`, preserve the old
run directory and first recover its archive offline:

```powershell
python Model2/Experiment5/run_hardware.py --resume "Model2/Experiment5/hardware_runs/cosine44_<timestamp>/manifest.json" --recover-compiled
```

Recovery loads one saved job-sized range at a time, verifies circuit identities,
44-step/reset/CZ resources, final readout and the saved physical chain, and
rebuilds the missing manifest metadata. It hashes the archive in 1 MiB chunks
and records byte ranges for later bounded loading. It retains the original QPY
unchanged, performs no transpilation, and loads no IBM account. A truncated or
invalid archive fails without changing the manifest or submitting jobs.
Keep the Qiskit environment that wrote the archive; recovery supports both
sequential and offset-table QPY formats.

After successful recovery, resume the same saved plan with the earlier
`--resume ... --execute` command. The seed, 16,384 paths, shots, backend and
selected chain come from that manifest; do not pass new run settings.
If the crash preceded the compilation-time backend snapshot, recovery records
that snapshot as unavailable rather than inventing historical calibration.
Execution still records the current backend snapshot before submission.

Fresh compilations now checkpoint an independently hashed QPY per Runtime job
(at most `--batch-size` circuits). A one-job run retains the old
`isa_circuits.qpy` filename. Resume verifies and reuses complete batches rather
than recompiling them, and execution loads only the requested job's circuits.
All jobs are still compiled and checked before the first hardware submission.
Hashing and resume never read the entire large archive into a byte string.

### Inspect or retry an IBM compiler failure

IBM error 6056 is an internal compiler error; the registry recommends retrying
or contacting support. A maintenance warning is relevant backend context, not
proof of the specific cause. The runner stops after the failing job; it does
not submit the remaining jobs automatically after that exception.

Inspect a saved job and the backend without submitting anything:

```powershell
python Model2/Experiment5/run_hardware.py --resume $runManifest --inspect-job dbkkc08nceqs73am38fg
```

This reads the saved manifest, queries job status/error/usage and backend
status, and saves `last_job_diagnostic.json`. It does not load the QPY archive,
change the manifest, or create a new job. It uses the existing account loader.

When Phoenix is operational, an explicitly requested retry is:

```powershell
python Model2/Experiment5/run_hardware.py --resume $runManifest --execute --retry-failed-job dbkkc08nceqs73am38fg
```

Retry is allowed only for the current saved job, with server status `ERROR`,
error code 6056, and no collected results. It refuses active, successful,
cancelled, unknown, or unrelated failures. The old job ID, error and available
usage metrics are retained under the batch's `failed_attempts`; the exact
circuits, path, shots and source are reused. After the retried batch succeeds,
execution continues through the original remaining batches.

New submissions check backend readiness before compilation/loading and again
before each job. The backend must be operational and report status `active`.
Local preparation and readiness failures do not mark a batch as an ambiguous
submission. A network failure *during* submission still retains the existing
ambiguity guard. A plain resume retrieves the known failed job and will not
silently replace it with a new one.

The [IBM error registry](https://quantum.cloud.ibm.com/docs/en/errors) defines
6056. If it recurs on an operational backend, retain both job IDs and report
them to IBM support; the code does not establish that shots or job size caused
the failure. Retrying does not guarantee a server-side compiler fix.

The manifest preserves exact paths, shots, basis groups, native QPY hash,
mapping and job IDs. Resume retrieves submitted jobs and submits only
batches without a recorded ID. If submission was interrupted while the
job ID was unknown, it refuses to submit that batch again; locate the job
in IBM Quantum and record its ID before resuming. Resume accepts only the
saved plan, action and optional account file, preventing accidental shot
or backend changes. A resume without an action only inspects the saved
plan offline. Fully collected results can be reanalyzed without an account.

```powershell
python Model2/Experiment5/analyze.py Model2/Experiment5/hardware_runs/cosine44_<timestamp>
```

Analysis reconstructs raw populations, parity-correct currents, conditional
shot SE and across-path SE, plus same-job preparation diagnostics. The two
SE quantities have different meanings and are not added. They are
descriptive, not simultaneous confidence guarantees or a drift model.
Current values are absent when only Z was measured.

Each complete run contains `Plan.md`, `Analysis.md`, `observables.csv`,
`observables.png`, raw count-bearing `manifest.json`, `hardware_results.json`,
`source.json`, the frozen `spec.json`, native QPY files (one per job for new
multi-job runs, or the indexed original `isa_circuits.qpy` for recovered runs), physical
path/calibration snapshots, and **`results_bundle.zip`**. The bundle includes
the submitted QPY as well as the raw results for later offline auditing.
Compilation and execution snapshots are retained separately. Raw JSON/QPY
and bundles stay local according to the repository's artifact policy.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_exp5_mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_model2_experiment5.py' -v
```

Tests cover full-density Model2 equivalence at the actual large angles for
all four jump types, exact mixture preparation, matching the full replay's
RNG hash, smaller-ensemble disclosure, per-job controls and basis grouping,
native mapping/QPY/reset checks, current parity and normalization, offline
planning, immutable-plan hashes, mixed-shot PUBs, job-ID persistence,
interrupted collection, ambiguous submission, offline reanalysis, and the
self-contained result bundle. All hardware submission tests use mocks.

The memory/recovery and compiler-retry follow-up passed **43 local tests**, covering the Model2
hardware runner and shared Runtime/control regressions. Recovery tests cover
QPY versions 13, 15, 16 and 17, forbidden unbounded reads, an unchanged legacy
archive, truncated-file rejection, partial compilation checkpoints, mocked
multi-job submission, and a 156-qubit FakeKingston round trip against the
standard QPY loader. No test submits live hardware jobs. The two-path offline snapshot
compilation has **880 CZ gates and 44 resets per evolved circuit**; its
native depths are 720–727 across the saved readout settings. These local
checks do not predict live-device accuracy.

Retry tests additionally cover read-only inspection during maintenance,
preserved failed-job/error/usage history, refusal to retry active or completed
jobs, and backend status changing before submission. A preliminary budget for
the raw 97-job configuration is in [BudgetEstimate.md](BudgetEstimate.md).

## Offline instruction-limit audit

To count the complete compiled Phoenix archive without submitting, recompiling,
or changing the saved plan:

```powershell
python Model2/Experiment5/instruction_audit.py --manifest $runManifest
```

The command verifies saved source/QPY hashes and counts every native program
with bounded memory. It writes `instruction_audit/InstructionAudit.md`, JSON
and per-qubit/partition CSVs beside the existing manifest. This can take time
for 49,443 programs. It counts IBM's published instruction weights per qubit,
charges CZ to both operands, conservatively includes implicit initialization,
and checks shot executions separately. Unsupported operations fail rather
than disappearing from the count.

The report gives a mathematical job-count lower bound and a concrete partition
of whole trajectory groups, with a complete control set for each proposed job.
If both counts agree, that is the minimum for these vectors under the stated
heuristic. A separate 80%-capacity calculation gives memory headroom. These
are proposed counts, not changes to the current 97-job run or proof of IBM
compiler acceptance.

The [local footprint projection](../Experiment6/instruction_audit/InstructionAudit.md)
uses 32 saved FakeKingston trajectories. It predicts four raw jobs for 16,384
paths under the published 26.8-million-instructions-per-qubit rule, or five
jobs with an 80% memory cap. It does not replace the exact Phoenix archive
audit. Six focused tests cover instruction weights, shot-versus-memory
accounting, complete path groups/controls, lower-bound certification, and
read-only audit of a real compiled archive.
