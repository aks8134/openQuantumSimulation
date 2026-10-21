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
