# Experiment5: circuits and FakeKingston depth for a saved adaptive schedule

This experiment reads Experiment4 replay data and builds the corresponding
sampled XX-plus-boundary-pump/loss circuits. It compiles against the installed
local `FakeKingston` snapshot and measures circuit depth and native gate counts.
It does not execute circuits, start new covariance simulations, access an IBM
account, or submit hardware jobs.

The default source is the successful tolerance-0.02, R=8192 run:

`Model3/Experiment4/runs/budget_e8ec77f803de/replay_R8192.json`.

It contains N=10, J=gamma=1, half_mixed preparation, and 490 steps:
20 at dt=0.5, 20 at dt=0.25, and 450 at dt=0.0625. Source physics, timesteps,
and saved jump choices are read directly; they are not resampled. Other
Experiment4 replay files with even-odd XX evolution can also be supplied.

## Run

From the repository root:

```bash
MPLCONFIGDIR=/tmp/model3_exp5_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/circuit_depth.py
```

The default compiles the complete schedule for saved trajectories 0,1,2,3.
These four are a depth sample, not all 8192 trajectories. To inspect growth
through the three schedule stages:

```bash
MPLCONFIGDIR=/tmp/model3_exp5_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/circuit_depth.py \
  --trajectory-indices 0 1 2 3 --prefix-steps 20 40 490
```

Useful options:

- `--source PATH`: another saved Experiment4 replay JSON.
- `--trajectory-indices ...`: zero-based saved paths to compile.
- `--prefix-steps ...`: requested evolution lengths, including optional 0 for preparation alone.
- `--ancillas boundary|shared`: two local endpoint ancillas (default), or one shared ancilla.
- `--layout chain|automatic`: deterministic connected path (default), or compiler-selected layout.
- `--physical-path ...`: override the connected physical path; in chain order, left bath, sites, right bath.
- `--optimization-level 0|1|2|3` and `--seed-transpiler`: defaults 3 and 11.
- `--no-plots` and `--output-directory PATH`.

## Circuit construction

### Exact mixed preparation

The saved initial state is a diagonal product mixture, not a pure state with
the same local populations. To prepare occupation probability p at a site,
apply RY(2 asin(sqrt(p))), then measure that site in Z. Leave the resulting
system bit in place and ignore the classical preparation record. For p=0.5
use H before measurement. Pure p=0 and p=1 use the supplied zero state and X,
respectively. These operations are parallel across sites, and both the
rotation and measurement costs are included in compiled depth.

This produces exactly the intended initial density matrix after averaging
preparation outcomes. No extra circuit ancilla, postselection, conditional
gate, or reset of a prepared system site is needed. A real experiment still
needs enough shots to average preparation and measurement randomness; this
compilation experiment does not assess that uncertainty. All qubits are
assumed to begin in the device's standard zero state; a job-level initial
reset is not explicitly prepended.

### Hamiltonian and selected jump

The Hamiltonian is `prefactor*J*sum(XX+YY)`. Each step applies the even bonds,
then odd bonds, using Qiskit's `XXPlusYYGate(4*prefactor*J*dt)` on every bond.
The saved factor order determines whether this comes before or after the
selected boundary channel.

Jump labels 0,1,2,3 mean left pump, left loss, right pump, right loss. The
sampled channel has angle `theta=sqrt(gamma*fraction*dt/probability)`, with
fractions `(p_left,1-p_left,p_right,1-p_right)`. Apply
`XXPlusYYGate(2*theta)` between that system endpoint and its bath qubit.
For pumping, first X the zero bath qubit; for loss, leave it zero. Reset the
chosen bath qubit afterwards, including the last step. This retains the
original signed-cosine channel, rather than replacing it by exponential damping.

Two independent endpoint baths implement the same reduced system channel as
one reset shared bath. Only one is used in each sampled step. The default
layout embeds bath-left, the 10 system sites, and bath-right on a physical
CZ-connected path, so every interaction is initially local. One shared bath
saves a qubit but requires connecting it to both ends and may cause routing.
The selected physical path is reproducible, not claimed optimal for calibration
quality or circuit depth. No global barriers inflate the dependency depth.

### Final observables

Z measurement provides all site populations. Two additional settings measure
alternating XYXY... and YXYX..., covering both correlators on every bond:

`I_j = prefactor*J*(<Y_j X_(j+1)> - <X_j Y_(j+1)>)`.

Compile each evolution body once. Readout rotations and measurements are then
appended using the final logical-to-physical mapping, without reoptimizing or
rerouting the body. Native H-equivalent readout is RZ(pi/2), SX, RZ(pi/2);
S-dagger followed by H for Y readout reduces to SX, RZ(pi/2), up to global phase.
All appended operations are checked against the physical target. Their costs
are included in full-circuit metrics. The two preparation/readout classical
registers remain distinct.

## Outputs and interpretation

Each configuration writes `Model3/Experiment5/runs/depth_<hash>/` containing:

- `Depth.md`: depth and gate-count table.
- `depth.json`: source SHA-256, versions, backend, path, mapping, compilation
  options, per-path/per-prefix metrics, timing, and source accuracy assessment.
- `circuits/*_logical.qpy`: the logical evolution body, including preparation.
- `circuits/*_native.qpy`: four native circuits, in order: body, Z, XY, YX.
- `figures/depth.png`, `figures/one_step.png`, and `one_step.txt`.

Load a saved circuit using:

```python
from qiskit import qpy
with open(".../circuits/trajectory_0_steps_490_native.qpy", "rb") as handle:
    body, populations, alternating_xy, alternating_yx = qpy.load(handle)
```

Depth is Qiskit's unweighted dependency-layer count, including resets and
measurements. Two-qubit depth filters to two-qubit gates. These numbers are
not a scheduled execution duration: reset and measurement can cost much more
time than single-qubit gates. Full target-width allocation can be 156 qubits
while only 12 are active; both are reported. Counts describe the compiled
paths and transpiler seed, not a best possible implementation or all R paths.

R=8192 multiplies repetition workload, not one circuit's depth. Measuring all
the observables uses three settings per trajectory, plus shots. Validating
an entire steady-time window would require additional evolution prefixes.
The existing ideal tolerance result does not predict hardware accuracy.

## Validation

`tests/test_model3_experiment5.py` checks the actual small-system density-matrix
channel against the covariance implementation for all four jump labels, both
orders and policies, shared and separate bath qubits, and negative dilation
cosines. It checks exact mixed preparation, current sign and readout rotations,
physical layout and target compatibility, source preservation, and QPY reload.

Gate normalization follows the official
[XXPlusYYGate matrix](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.XXPlusYYGate).
The backend is the locally installed
[FakeKingston snapshot](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-kingston).

## Executed result

The stage-prefix command above was executed for trajectories 0,1,2,3 and
20,40,490 steps. Results are in
[`runs/depth_ae673d9d9a95/Depth.md`](runs/depth_ae673d9d9a95/Depth.md).
Qiskit 2.5.2 and qiskit-ibm-runtime 0.49.0 compiled against the local
156-qubit target, using physical path `[0,1,2,3,4,5,6,7,8,9,10,11]`.
There are 12 active qubits: 10 system sites and 2 endpoint baths.

| Full 490-step circuit metric | Observed over the four paths |
|---|---:|
| Native Z-readout depth | 7850-7852 |
| Native depth across Z/XY/YX readouts | 7850-7855 |
| Two-qubit depth | 1960 |
| CZ gates | 9800 |
| Bath resets | 490 |
| Preparation measurements | 10 |
| Final readout measurements | 10 |

Each step has 9 system exchange blocks and 1 sampled bath exchange block.
For this compilation each exchange costs 2 CZ gates, giving
`490*(9+1)*2 = 9800` CZ gates. The 490 Trotter steps therefore do not mean
490 native circuit layers. The saved ideal accuracy result remains separate
from this offline resource calculation.

Seven unit tests passed. The final native QPY file was reloaded and its four
circuits' depths/reset counts matched the saved metrics. Source hashes and
the earlier Experiment4 analysis inputs remained unchanged.

Experiment5 also accepts `Model3/Experiment6` replay JSON files with the same
even-odd channel schema. Supply the new file explicitly using `--source` to
compile a budget-controller schedule. Its future mean-only validation work is
excluded from the executable preparation circuit.

## Executed shorter forward-validated preparation

The 117-step Experiment4 preparation was subsequently compiled using
`--source Model3/Experiment4/runs/budget_419029a6f9bc/replay_R8192.json`
and `--trajectory-indices 0 1 2 3 --prefix-steps 8 64 117`, with the same
FakeKingston settings as the 490-step comparison. Across the four full paths
and Z/XY/YX readouts, native depth is 1882-1887, two-qubit depth 468, and
CZ count 2340. There are 14715-14761 unitary gates including RZ, plus 117
resets and 20 measurements per final circuit.

The 200-step classical mean continuation used by forward validation is
excluded from the preparation circuit. Sampling is checked at the output.
See [analysis/Forward117DepthComparison.md](analysis/Forward117DepthComparison.md)
for the definition of gate counts, full comparison, saved circuits, and
reproducible command. This command uses an explicit source; the original
script default still points to the earlier 490-step study.

## Run the cosine schedule on IBM hardware (PowerShell)

The separate `run_hardware.py` runner executes saved Model3 trajectories.
It rebuilds the logical circuits and recompiles them for the requested live
backend; it does not submit the FakeKingston QPY files to another processor.
The original `circuit_depth.py` remains an offline compiler.

From the repository root in Windows PowerShell, with the updated runner and
tracked compact cosine input available in your Windows checkout:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_hardware.py `
  --source ".\Model3\Experiment5\inputs\cosine72_seed29_paths0-3.json" `
  --backend ibm_phoenix `
  --account-file ".\IBMRuntime\apikey.json" `
  --trajectory-indices 0 1 2 3 `
  --shots 8192 `
  --batch-size 12 `
  --optimization-level 3 `
  --seed-transpiler 11
```

PowerShell uses a backtick for continuation; do not put spaces after it.
If Python is already active in your environment, `python` can replace
`.\.venv\Scripts\python.exe`. A saved Qiskit account is supported by omitting
`--account-file` when `IBMRuntime/apikey.json` is absent. An explicit account
file uses the existing project loader, accepting `apikey`/`api_key`, optional
`crn`/`instance`, and optional `channel` (default `ibm_quantum_platform`).

This submits **12 circuits**: four saved random jump paths, each with Z,
alternating XY, and alternating YX readout, with **8192 shots per circuit**.
There are 72 preparation steps, two boundary ancillas, and 12 active logical
qubits. The 175 classical forward-check steps are excluded. Four paths are
a hardware trial and do not reproduce the ideal R=8192 trajectory average;
shots and trajectory count are separate quantities. No DD, gate twirling,
measurement twirling, or mitigation is enabled by this runner.

To select the first N saved paths, use `--trajectory-count N` instead of
`--trajectory-indices`. The two options are mutually exclusive. Omitting both
retains the default indices `0 1 2 3`; the count must be positive and cannot
exceed the number of paths in the input. For 8192 trajectories with one shot
per measurement setting and 512 circuits per job:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_hardware.py `
  --source ".\Model3\Experiment7\runs\comparison_d3fd93cb452e\cosine\replay_seed29_R8192.json" `
  --backend ibm_phoenix `
  --account-file ".\IBMRuntime\apikey.json" `
  --trajectory-count 8192 `
  --shots 1 `
  --batch-size 512 `
  --optimization-level 3 `
  --seed-transpiler 11
```

The full replay must be copied to the Windows machine; it is excluded from
Git. The default compact input contains only four paths. This command requests
24,576 circuits in 48 sequential jobs. Compilation still builds all circuits
before submission, so it can require substantial time and memory. The separate
`analyze_hardware.py` currently requires at least two shots per circuit;
one-shot results require an analysis that estimates uncertainty across paths.

The compact input is approximately 100 KiB and preserves the exact first four
paths and all 72 timesteps from the original replay. It is the runner's new
default, so `--source` can also be omitted. Full replay JSON and generated
NPZ/QPY files remain local under the [artifact policy](../../ARTIFACTS.md).
To select other saved paths, export another compact input from a full local
replay with `export_hardware_input.py`. Original R=8192 validation is recorded
as provenance; the four-path input does not claim that sampling validation
for itself.

The runner resolves `ibm_phoenix` through your account with fractional gates
disabled, chooses a connected chain from its live target, and includes the
original mixture-preparation measurements and all bath resets. Actual depth
and gate counts are recorded after this new compilation. Live availability
and access are checked when you run the command, not established by local
tests. SamplerV2 is used because these circuits contain mid-circuit
measurements and resets; see IBM's
[dynamic circuit guide](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits)
and [SamplerV2 API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2).

Results are saved under
`Model3/Experiment5/hardware_runs/ibm_phoenix_<timestamp>/`:

- `manifest.json`: source/ISA hashes, actual compiled metrics and layout,
  circuit-to-trajectory/basis mapping, per-batch job IDs and raw final counts.
- `source.json` and `isa_circuits.qpy`: the exact saved source and live-target
  circuits, retained for reproducibility and recovery.
- `hardware_results.json`: final populations/currents, across-path standard
  errors including shot fluctuations, and differences from the physical
  steady state, ideal mean channel, and ideal average of the same saved paths.

Preparation measurement records are ignored. Final counts are read only
from the `outcome` register, preserving the logical site order. Currents use
`prefactor*J*(<YX>-<XY>)` on each bond, with alternating readout parity handled
explicitly. The classical ideal tolerance is not asserted for noisy results.

For compilation against the live backend without submission, append
`--compile-only`. To recover an interrupted hardware run without resubmitting
jobs that already have saved IDs:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_hardware.py `
  --resume ".\Model3\Experiment5\hardware_runs\ibm_phoenix_TIMESTAMP\manifest.json" `
  --account-file ".\IBMRuntime\apikey.json"
```

The runner saves each job ID before waiting for results, skips collected
batches, retrieves submitted jobs, and submits only batches with no prior
submission. If a submission was interrupted before its ID could be saved,
resume stops and asks you to find that job in IBM Quantum and record its ID,
avoiding an automatic duplicate submission. Batches are collected sequentially.
Compilation-only manifests can also be resumed to submit their saved circuits.

Local tests use a generic backend and mocked Runtime jobs. They cover live
target selection, native compilation, two baths and preparation measurements,
bit order and current signs, output-register isolation, job persistence,
recovery, ambiguity handling, batching, shot validation, and integrity checks.
No IBM hardware job was submitted during implementation.

## Population-only Phoenix diagnostic with controls

### Two-stage workflow with calibration-selected paths

`run_population_diagnostic.py` separates a short controls experiment from a
later full evolution experiment. Neither stage automatically launches the other.
The commands below are for Windows PowerShell from the repository root.

First select the path and execute **only three controls**, 2048 shots each:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_population_diagnostic.py controls --backend ibm_phoenix --account-file .\IBMRuntime\apikey.json --control-shots 2048 --avoid-qubits 7
```

This refreshes the live backend configuration and target before searching
connected 12-qubit paths. Default hard filters are T1 >= 80 µs, T2 >= 60 µs,
readout error <= 2%, and reported CZ error <= 0.5%; missing calibration data
are excluded. Qubit 7 is excluded by default. Qualifying paths are ranked by
worst CZ error, a CZ-error sum weighted by actual saved-path gate use, worst
and total system readout error, then minimum T2/T1. Paths have a bath at each
end and the ten system qubits between them. The score is a heuristic and does
not predict an observable error or certify the 0.02 tolerance. The path search
reports whether its two-million-expansion limit was reached; thresholds are
never silently relaxed. Filter/search options are listed by `--help`.

Controls-only skips building the 72-step bodies. It executes all-zero,
all-one, and initial-mixture circuits, **3 circuits / 6144 shots**, and saves:

- A normal run folder with source, actual compiled QPY, counts, and calibration.
- `selected_path.json` with selection criteria, metrics and search diagnostics.
- `ControlReport.md` and `control_assessment` in `hardware_results.json`.
- `results_bundle.zip`, which includes the actual compiled QPY alongside results.
- `hardware_runs/latest_population_controls.json`, a pointer to that control run.

The diagnostic requires family-adjusted Wilson upper bounds on zero/one
baseline errors to be <= 3%, and checks the mixture against the affine
prediction from zero/one populations with nominal family-adjusted normal
tests. The nominal family alpha is 0.05. Preparation errors, time dependence,
and non-binomial device behavior remain limitations; a pass does not validate
reservoir reset performance or full-evolution accuracy.

After the report says `Controls diagnostic passed: True`, execute the full
population experiment with **128 trajectories / 64 shots each**, 8192 shots:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_population_diagnostic.py full --account-file .\IBMRuntime\apikey.json --trajectory-count 128 --shots 64
```

The full stage reads the pointer, verifies the original source and controls
QPY hashes, recomputes the control diagnostic from manifest counts, and
refreshes calibration. It reuses that exact physical path and saved source;
it does not select a replacement path. Failed controls, changed source/QPY,
backend mismatch, or a path that no longer meets its original quality filters
stop the run before submission. The original 72-step cosine schedule and
boundary choices are unchanged. The two stages are separate one-job requests
at the default batch size 256.

Use `--selection-file PATH` on both commands to keep multiple control/path
experiments separate. `--compile-only` saves circuits without submission and
does not replace the control-run pointer. For a real control run, the pointer
is saved before submission; incomplete results block the full stage. An
interrupted run can be recovered with the existing
`run_hardware.py --resume MANIFEST` command. Once collection completes, the
full command uses that same pointer and recomputes the diagnostic; its cached
pass flag is never trusted. There is no need to resubmit the controls.
Copy `results_bundle.zip` back with each run so compiled circuits are retained.

The workflow was checked only with injected local backends, mocked Sampler
jobs, and the previously saved calibration snapshot. No IBM account lookup,
live backend refresh, calibration experiment or hardware job was performed
during implementation. The actual path is selected when you run the first
command.

### Earlier combined experiment

Copy the updated `run_hardware.py` and compact
`inputs/cosine72_seed29_paths0-127.json` into the Windows checkout. The new input
is about 608 KiB and preserves the exact first 128 saved trajectories, including
the original four. It includes matching ideal covariance references and does
not require the large, Git-ignored replay file.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_hardware.py `
  --source .\Model3\Experiment5\inputs\cosine72_seed29_paths0-127.json `
  --backend ibm_phoenix `
  --account-file .\IBMRuntime\apikey.json `
  --trajectory-count 128 `
  --bases z `
  --shots 64 `
  --population-controls `
  --control-shots 512 `
  --batch-size 256 `
  --physical-path 0 1 2 3 4 5 6 7 8 9 19 29 `
  --optimization-level 3 `
  --seed-transpiler 11
```

This plans **131 circuits in one job**: 128 evolved Z-measurement circuits at
64 shots each plus three controls at 512 shots each, **9728 shots total**.
The physical path is the one used in the earlier Phoenix runs; live target
compatibility is checked during compilation. Control circuits prepare all
zeros at the beginning, all ones near the midpoint, and the original initial
mixture at the end. These are submission-order positions; device execution
timing is recorded when available. The mixture control retains its initial
measurements but has no Trotter steps. Controls are checked against their
known populations and excluded from the trajectory average. This is a
preparation/readout diagnostic rather than a full correlated-readout correction.

SamplerV2's per-PUB shot override lets trajectory and control circuits share
a job with different shots. Collection checks each result against its own
requested shot count. See the
[SamplerV2 interface](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2).
The defaults still measure all three bases; `--bases z` selects population-only
results and `error_scope: populations_only`. Unmeasured currents are omitted.
The matching ideal references use only the selected 128 trajectories; the
original full-ensemble sampling validation is not claimed for this subset.

The run folder keeps `source.json`, `isa_circuits.qpy`, `manifest.json`, and
`hardware_results.json`, plus a new `backend_snapshot.json` with available
compilation-time qubit/gate properties. Per-job timing metadata is saved when
available. Keep these files together when copying the results back.
`--compile-only` and `--resume` work with mixed-shot control runs as well.

### Analyze the collected population/control run

`analyze_populations.py` independently reconstructs Z counts and their
within-shot covariance, checks control/trajectory separation and each PUB's
requested shots, and compares populations against the matching saved ideal
paths and analytically reconstructed stationary target. It also tests the
initial-mixture control against the local response predicted by zero/one
controls, and summarizes the selected-qubit calibration snapshot. It does
not connect to IBM or evolve any new trajectories.

```bash
MPLCONFIGDIR=/tmp/model3-pop-analysis-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/analyze_populations.py \
  --run-directory Model3/Experiment5/hardware_runs/ibm_phoenix_20261023T135506_304864Z \
  --compare-analysis \
    Model3/Experiment5/analysis/hardware/ibm_phoenix_20261020T190507_427277Z/analysis.json \
    Model3/Experiment5/analysis/hardware/ibm_phoenix_20261021T102532_376654Z/analysis.json
```

All 131 circuits and 9728 shots of job `dbdmeh7i0auc73crdct0` were collected.
The maximum hardware population error is **0.09106** against physical and
**0.10188** against its matched ideal, while the 128 ideal paths have maximum
physical population error **0.01900**. The initial-mixture control gives
site 7 (physical qubit 7) population **0.38086** instead of 0.5; zero/one
controls predict **0.47656**, a discrepancy of **4.33 nominal SE**. That
qubit has reported T1 **16.67 µs**, T2 **19.61 µs**; adjacent CZ edge 7–8
has reported error **1.2366%**. Together these are concrete layout concerns,
not a unique causal diagnosis. The original ISA QPY is still absent from
the pulled run; the calibration snapshot is available.

The analyzer writes population and control/calibration PNG/PDF figures,
per-path residuals, a comparison with prior runs, CSVs, and a Markdown report
to a separate analysis folder. Primary results remain raw: an exploratory
local affine adjustment is reported only as a response-model diagnostic,
with its assumptions and propagated uncertainty. Original files are
hash-checked and preserved.

See the [128-path population report](analysis/hardware/ibm_phoenix_20261023T135506_304864Z/Analysis.md),
[population plot](analysis/hardware/ibm_phoenix_20261023T135506_304864Z/populations.png),
and [controls and calibration](analysis/hardware/ibm_phoenix_20261023T135506_304864Z/controls_and_calibration.png).

## Analyze saved hardware results and plot output observables

The analysis command reads collected counts and source references without
connecting to IBM, evolving new trajectories, or changing the run files:

```bash
MPLCONFIGDIR=/tmp/model3_hw_analysis OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/analyze_hardware.py \
  --run-directory Model3/Experiment5/hardware_runs/ibm_phoenix_20261020T190507_427277Z
```

Use `--output-directory` to change its default output directory under
`Model3/Experiment5/analysis/hardware/<run-name>/`. It writes `Analysis.md`,
`analysis.json`, a three-panel final population/magnetization/current plot in
PNG/PDF, a per-trajectory current plot, and per-observable CSV tables.
These are spatial profiles at the measured output; no hardware time series
was collected. Error bars show twice the nominal shot SE conditional on the
selected paths. Within-setting site/bond correlations are retained, and
across-path uncertainties are reported separately to avoid double-counting.

For job `dbbrmpc24mrc73ciaq4g` (4 paths, 12 circuits, 8192 shots each), the
maximum local hardware error is **0.25103**, exceeding the 0.02 target.
The same four ideal paths have error 0.11525 from the physical target; their
sampling discrepancy from the ideal mean channel is 0.10305. Hardware differs
from those same ideal paths by 0.25761. These maximum norms do not add as
scalar contributions. The measured spatial mean current is 0.09745 versus
0.27880 for the same ideal paths and the physical stationary current 0.32.
Nominal shot errors are much smaller than these discrepancies.

See [the saved-run report](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/Analysis.md)
and [the observable plot](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/observables.png).
The source hash was reproduced exactly after restoring Windows CRLF line
endings. The ISA QPY file was not copied with this run, so its stored hash
could not be checked here. The analysis does not infer permanent stationarity,
attribute a specific hardware noise mechanism, or treat a measured population
slope as evidence of diffusive transport. Six statistical reconstruction
tests and the twenty existing hardware/circuit tests passed.

## Analyze saved one-shot trajectories

For one shot per trajectory and measurement setting, use the separate
`analyze_one_shot.py` entry point. It estimates uncertainty across trajectories;
within-circuit shot variance cannot be estimated from a single outcome.
It reconstructs every raw count, checks the matching ideal covariance references,
and adds a job-cluster sensitivity check using complete trajectory triples within
each job. No Runtime connection, trajectory evolution, or mitigation is used.
The job-cluster diagnostic requires at least two jobs with two complete paths each.

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\analyze_one_shot.py `
  --run-directory ".\Model3\Experiment5\hardware_runs\ibm_phoenix_20261021T102532_376654Z" `
  --compare-run ".\Model3\Experiment5\hardware_runs\ibm_phoenix_20261020T190507_427277Z"
```

The optional `--compare-run` reads the previous multi-shot run for a descriptive
comparison. Outputs go to `analysis/hardware/<run-name>/`: `Analysis.md`,
`analysis.json`, observable PNG/PDF, a per-job current plot, and CSV tables.
`--output-directory` selects another analysis destination; original run files
are hash-checked and preserved.

The collected 8192-path, one-shot run has 48 jobs and 24,576 circuit executions.
Its maximum physical error is 0.2792285, while the same ideal paths have error
0.01710975. Hardware mean current is 0.1040446 versus the matched ideal 0.3095208.
The nominal pooled SE of the mean current is 0.00257455; clustering by job gives
0.00324952 on the complete-path subset. This result does not meet the 0.02 target.
See [the one-shot analysis](analysis/hardware/ibm_phoenix_20261021T102532_376654Z/Analysis.md)
and [actual observable plot](analysis/hardware/ibm_phoenix_20261021T102532_376654Z/observables.png).

## Subsample the earlier population measurements

`subsample_populations.py` studies shot allocation using only the earlier saved
Z-basis counts. It samples whole bitstrings without replacement within each of
the four fixed trajectories, preserving correlations between sites. The default
uses 10,000 replicates at each of eleven shot levels from 1 to 8192 shots/path.
It writes actual population profiles, resampling quantiles, CSV tables, and an
exact finite-pool covariance check to a separate `population_subsampling/` folder.
Original results are hash-checked and preserved; no hardware connection or
trajectory evolution is performed.

```bash
MPLCONFIGDIR=/tmp/model3-subsampling-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/subsample_populations.py \
  --run-directory Model3/Experiment5/hardware_runs/ibm_phoenix_20261020T190507_427277Z \
  --compare-analysis Model3/Experiment5/analysis/hardware/ibm_phoenix_20261021T102532_376654Z/analysis.json
```

At 2048 shots/path, the four old paths use 8192 Z measurements, matching the
newer experiment's population-measurement count. Population MAE versus the
physical target has median **0.04201**, central 95% resampling range
**0.03918–0.04485**, versus **0.04187** for the full old dataset and **0.04900**
for the newer hardware run. These conditional subsamples do not reproduce a
different trajectory ensemble or execution occasion. One shot on four fixed
paths also does not emulate one shot on 8192 distinct paths.

See the [population subsampling report](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/population_subsampling/Analysis.md)
and [actual population profiles](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/population_subsampling/population_subsamples.png).

## Validate populations with noiseless Aer

`check_aer_populations.py` reconstructs the same saved logical paths and checks
their populations in Aer without connecting to IBM. An exact ten-qubit
density-matrix calculation traces out each freshly prepared bath after its
collision; ignored preparation outcomes are averaged exactly. This transformation
is checked against explicit system-plus-bath density matrices in the tests.

A separate simulation retains the full twelve-qubit circuit, initial preparation
measurements, bath resets, and terminal Z measurements. It compiles to an offline
CZ line with the saved transpiler seed/optimization level and samples 512 shots
per path by default. In addition to raw counts, simulator-only conditional
expectation snapshots give a lower-variance population estimator with its own
sampling SE. These snapshots are not available from hardware measurements.

```bash
MPLCONFIGDIR=/tmp/model3-aer-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
.venv/bin/python Model3/Experiment5/check_aer_populations.py \
  --run-directory Model3/Experiment5/hardware_runs/ibm_phoenix_20261020T190507_427277Z
```

Outputs go to `analysis/hardware/<run-name>/aer_population_check/`: the report,
population plots for the ensemble and each path, a CSV, per-path counts and
diagnostics, and compressed regenerated QPY circuits. Original source/results
are hash-checked and preserved. The original Phoenix ISA QPY files were not
pulled, so the offline regenerated compilation does not certify the historical
Phoenix circuits or reproduce their execution-time calibration.

See the [Aer population report](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/aer_population_check/Analysis.md)
and [population comparison](analysis/hardware/ibm_phoenix_20261020T190507_427277Z/aer_population_check/populations.png).

For the saved four-path Phoenix run, all 40 exact density populations agree
with their saved ideal references within **2.50e-15**. The full twelve-qubit
CZ simulations used 512 shots/path; the largest conditional-estimator
discrepancy was **2.06 nominal SE** across sites/paths. The raw-count
populations also agree within sampling uncertainty (largest marginal
standardized discrepancy **2.50 SE**). The ensemble maximum hardware-versus-
ideal population difference remains **0.08229**. This supports consistency
of the regenerated circuit and mathematical reference; it does not establish
which device error caused the hardware discrepancy. All 26 relevant circuit,
Aer-helper, and hardware-runner tests passed. The compressed QPY archive was
reloaded successfully and all original input hashes were unchanged.

## Diagnose population errors using circuit prefixes

`run_prefix_diagnostic.py` reuses the collected full run's **physical chain,
original cosine timesteps, preparation and saved branch choices**. Its default
reference is `hardware_runs/ibm_phoenix_20261024T042722_287349Z`. It refreshes the
backend calibration and checks the same path against its original quality
thresholds; it does not select a different path. A failing check stops before
submission. No change to the pump/loss gate construction is made.

The default is **32 trajectories × 128 shots**, Z measurements only, at
prefixes **0, 12, 24, 48 and 72**: **160 circuits, 20,480 shots, one batch** at
the default batch size of 256. These are the first 32 trajectories from the
reference run, reused at every prefix. Step zero repeats the initial-mixture
preparation, including its ignored measurements, with no evolution. It does
not add all-zero/all-one or bath-reset controls. The 32 zero-step circuits
have identical logical preparation and independent measurement samples.

All prefix/trajectory pairs are shuffled with a saved seed (`--order-seed 29`)
before submission, so prefix length is not deliberately ordered with job
position. This does not provide a calibrated execution timeline.

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_prefix_diagnostic.py --account-file .\IBMRuntime\apikey.json
```

Use `--reference-run` to choose a different collected full run, and optionally
`--trajectory-count`, `--trajectory-indices`, `--prefix-steps`, `--shots`, or
`--batch-size` to change the diagnostic. The trajectory selection must be a
subset of the reference hardware run. Prefixes must fit the original saved
schedule. In particular, a 48-step prefix retains the **first 48 original
cosine steps**; it does not rescale the cosine to 48 steps.

Add `--compile-only` to query the target and save native circuits without
submitting jobs. This still connects to IBM to resolve its live target.
An interrupted run can be recovered with its saved circuits, shots and job IDs:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_prefix_diagnostic.py --resume ".\Model3\Experiment5\hardware_runs\<prefix-run-folder>\manifest.json" --account-file .\IBMRuntime\apikey.json
```

Resume collects jobs that already have saved IDs and submits only batches
without an ID. An interruption between submission and saving an ID stops
automatic resubmission and requires recovering the ID first. A fully collected
resume can regenerate analysis without accessing the account. Run settings
come from the manifest on resume; other CLI selection/shot options do not
replace them.

Each run is saved under `hardware_runs/ibm_phoenix_prefix_<timestamp>/`, with
`source.json`, `ideal_prefixes.json`, reference provenance, `manifest.json`,
`isa_circuits.qpy`, calibration snapshot and selected-path data. Once collected,
`hardware_results.json` and `results_bundle.zip` include all prefix results;
the bundle retains the original compiled QPY plus the analysis plots/report.
Bring back the bundle after execution.

The analyzer recomputes the exact finite-step **ideal population at each
prefix for those same trajectories** using the covariance channel, including
the actual collision angles. It checks the full-prefix covariance against
the original saved final covariance. No Aer sampling is needed. Intermediate
prefixes are compared with their own transient predictions, not treated as
steady states. The smaller trajectory subset is not independently certified
to satisfy the original 0.02 sampling budget.

Analysis is written to `analysis/hardware/<prefix-run-folder>/`: actual
population profiles at each prefix (PNG/PDF), population progression at the
first/middle/last sites and
hardware-minus-ideal metrics (PNG/PDF), CSV, JSON and `Analysis.md`.
Error bars describe shot uncertainty conditional on these paths; across-path
and paired-residual covariance estimates are retained separately. No automatic
mitigation or attribution to a specific device error is performed.

To regenerate the analysis offline after pulling results:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\analyze_prefix_diagnostic.py --run-directory ".\Model3\Experiment5\hardware_runs\<prefix-run-folder>"
```

This reads saved counts and references only. It can verify the original QPY
inside `results_bundle.zip` if that file was not extracted separately.

## Fresh preparation/readout evaluation on the same chain

`run_preparation_diagnostic.py` follows up the population offsets seen before
any evolution in `ibm_phoenix_prefix_20261024T162033_596311Z`. It reuses that
run's physical path and measures **all ten system sites**, highlighting physical
qubits **56 and 76** (sites 5 and 7). It refreshes the backend target and saves
its calibration snapshot; it does not rerank the chain. Since these circuits
contain no exchange gates, an old full-evolution quality threshold is not used
to prevent measuring the same qubits again.

The default evaluates exactly three controls:

- All zero: deterministic zero preparation followed by final Z readout.
- All one: X preparation followed by final Z readout.
- Original initial mixture: the half-mixture preparation, including its
  unconditioned initial measurements, followed by final Z readout.

Each uses **4096 shots**, giving **3 circuits, 12,288 shots and one batch**.
There are no Trotter steps, two-qubit gates, bath collisions or bath resets.
This evaluates preparation/readout consistency; it is not a reset diagnostic.

Run on hardware from Windows PowerShell at the repository root:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_preparation_diagnostic.py --account-file .\IBMRuntime\apikey.json
```

`--shots` changes the shots per control. `--reference-run` can select another
collected population/prefix run, with `--focus-qubits` selecting physical
system qubits to highlight on that chain. All sites are still screened.
`--compile-only` saves circuits without submission, while querying the target.

The run directory is `hardware_runs/ibm_phoenix_<timestamp>/`. It retains the
source, manifest with reference provenance and physical mapping, QPY circuits,
calibration, raw counts, `hardware_results.json` and `ControlReport.md`.
The detailed report and PNG/PDF/CSV/JSON analysis go to
`analysis/hardware/<run-name>/preparation/`. The updated `results_bundle.zip`
includes those reports/figures plus the original compiled QPY and raw results.
Bring back this bundle after execution. No evolution run is automatically
launched, whether the controls pass or fail.

For each site's intended population p, the stationary local-response model
predicts `(1-p)*zero + p*one`. The mixture test includes the uncertainty in
all three controls. For p=0.5 this is simply `(zero + one)/2`. Baseline Wilson
upper bounds use the existing 0.03 screen, while mixture residuals use the
existing nominal family-adjusted test. These diagnostics do not certify the
0.02 evolved-state accuracy target or separate preparation, readout, timing,
crosstalk and device drift. No automatic mitigation is applied.

Recover interrupted hardware collection with the original manifest:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_preparation_diagnostic.py --resume ".\Model3\Experiment5\hardware_runs\<run-name>\manifest.json" --account-file .\IBMRuntime\apikey.json
```

Saved job IDs are collected without resubmitting those circuits. A submission
interrupted before saving its ID requires recovering that ID first. Offline
analysis, after pulling/extracting the JSON files and retaining the bundle:

```bash
.venv/bin/python Model3/Experiment5/analyze_preparation_diagnostic.py \
  --run-directory Model3/Experiment5/hardware_runs/<run-name>
```

## Isolate boundary collisions and reset sequences

`run_boundary_diagnostic.py` follows up the preparation/readout controls using
two-qubit boundary circuits. Its default reference is the collected control
run `ibm_phoenix_20261025T150619_925833Z`. The physical pairs are retained:
left system/bath **49/39**, right system/bath **95/105**. No system-chain bonds
are evolved. Each circuit measures **both** its active system and bath qubits.

For each side the default evaluates ten circuits:

- System 0 and system 1 baselines with bath 0: two circuits.
- Pump from system 0 and loss from system 1, at 1, 12 and 72 collisions:
  six circuits, resetting the bath after every collision.
- Two 72-reset controls with the same bath preparation as the pump/loss
  cases, with the exchange gate omitted. These leave the system ideally in
  its initial state and test effects associated with bath pulses, resets
  and waiting together.

This gives **20 circuits × 2048 shots = 40,960 shots**, one batch. The original
cosine sequence supplies the collision angles. These are repeated
**conditional** pump/loss branches, not stochastic reservoir sampling and not
a new steady-state schedule. For gate half-angle theta_k, ideal excitation
survival under loss is `product(cos(theta_k)**2)`; ideal pumping from zero gives
one minus that product. The ideal bath population after reset is zero.

Run from Windows PowerShell at the repository root:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_boundary_diagnostic.py --account-file .\IBMRuntime\apikey.json
```

`--sides right` selects only the right pair, `--steps 1 12` shortens the tested
sequences, and `--shots` changes the shots per circuit. Reset-only controls
use the largest requested count. `--compile-only` resolves the target and
saves native circuits without submitting. Native circuits are validated by
the installed Runtime ISA validator, including the synchronization barriers
that make the system wait for bath resets before readout.

The reset-only and collision circuits differ in duration, so their difference
is not an isolated CZ error estimate. Compare measured bath populations with
the fresh baseline bath readout before attributing any apparent excitation to
reset failure. Isolated-pair readout context and the added synchronization
also differ from the original full-chain circuit. No automatic readout
correction or causal noise attribution is applied.

Results go to `hardware_runs/ibm_phoenix_boundary_<timestamp>/`, retaining
source, shuffled case order, manifest, raw counts, original native QPY and
calibration snapshot. `results_bundle.zip` includes those files plus PNG/PDF,
CSV, JSON and a Markdown analysis report. Bring back the complete bundle.
Analysis files are also written under `analysis/hardware/<run-name>/boundary/`.

Resume interrupted hardware collection using the original circuits and job IDs:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_boundary_diagnostic.py --resume ".\Model3\Experiment5\hardware_runs\<boundary-run-name>\manifest.json" --account-file .\IBMRuntime\apikey.json
```

Offline analysis after pulling results:

```bash
.venv/bin/python Model3/Experiment5/run_boundary_diagnostic.py \
  --analyze-only Model3/Experiment5/hardware_runs/<boundary-run-name>
```
