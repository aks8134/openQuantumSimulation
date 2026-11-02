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

The runner prints its unique run directory under
`Model2/Experiment5/hardware_runs/cosine44_<UTC timestamp>/`. After an
interruption, substitute the printed manifest path:

```powershell
python Model2/Experiment5/run_hardware.py --resume Model2/Experiment5/hardware_runs/cosine44_<timestamp>/manifest.json --execute
```

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
`source.json`, the frozen `spec.json`, native `isa_circuits.qpy`, physical
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

The 12 new experiment tests and 21 existing circuit/Runtime regression
tests passed together: **33 tests passed**. The two-path offline snapshot
compilation has **880 CZ gates and 44 resets per evolved circuit**; its
native depths are 720–727 across the saved readout settings. These local
checks do not predict live-device accuracy.
