# Compile the existing study schedules for ibm_phoenix

[phoenix_resources.py](phoenix_resources.py) recompiles the **complete saved
preparation circuits** against a full captured backend target. It never
submits hardware jobs. The source schedules are not optimized again.

The included [portable input](inputs/phoenix_schedules.json) contains all
117 accepted schedules: three models, 13 sizes through N=100, and fixed,
searched-constant and warped-family methods. It preserves J=gamma=1,
H=(J/2) sum(XX+YY), preparations and every timestep. Constant arrays are
losslessly encoded by their value/count. This file travels with the code;
the original ignored study checkpoints are not needed on the machine doing
compilation.

## 1. Preview the workload without an IBM connection

From the repository root:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py plan
```

The default is 234 complete preparation paths with Z readout, using seeds
29 and 30, matching this study's resource-path definitions. Output is
`FinalModel/Experiment1/phoenix_runs/plan.csv`.

The largest source circuit, Model2 N=100 fixed, contains **5,128,000 exchange
blocks**. Full optimized transpilation can require substantial RAM and
time. Start with a smaller selection, then use a sufficiently large machine
for the longest baselines. The script does not replace a large circuit with
summed local-block counts or extrapolated depth.

## 2. Read and freeze the full Phoenix target once

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py capture-target --account-file IBMRuntime/apikey.json
```

This is the only command that contacts IBM. It requests
`service.backend("ibm_phoenix", use_fractional_gates=False)` and saves the
complete `backend.target`, its native instruction properties, timing
constraints, capture timestamp and backend compiler plugins. It creates:

```
FinalModel/Experiment1/phoenix_targets/ibm_phoenix/target.pkl
FinalModel/Experiment1/phoenix_targets/ibm_phoenix/target.json
```

Account files support the existing `apikey`/`api_key`, `crn`/`instance` and
`channel` fields. Credentials are not copied into the target cache or run
outputs. Omit `--account-file` to use the repository's account file if
present, otherwise the saved Qiskit account. `--account-name` selects a
named saved account; `--instance` can specify the instance.

The directory must be empty when capturing. For another calibration use
`--target-dir PATH_TO_NEW_CACHE`, then pass the same option to compilation.
Use your own generated cache: it contains a Python/Qiskit pickle. Capture
and compile in the same Python, Qiskit, Runtime and NumPy versions. The
script checks cache hashes and these versions before deserializing it.
Tested environment: Python 3.14.7, Qiskit 2.5.2, Runtime 0.49.0, NumPy 2.5.2.
Plot generation also uses Matplotlib.

A newly captured target represents its recorded capture time. It does not
recreate an earlier execution's calibration merely because the backend name
is the same. Historical reproduction requires the corresponding complete
target and compiler environment.

## 3. Compile a small selection, then extend the sweep

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py compile --sizes 2 4 10 --families warped_cosine --timing
```

After checking those results, compile the full study:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py compile --timing
```

Both commands reuse the frozen target and make no IBM connection. Completed
matching paths are reused automatically. Each path runs in a separate local
process so compiler memory is released afterwards. One path is compiled at
a time; all selected measurement bases reuse its globally compiled body.
An interrupted or failed path is retried on the next invocation. Filters
(`--models`, `--sizes`, `--families`) can be expanded within the same run;
changes to compiler settings, target, input, seeds or script create a new
run identity.

Default compiler settings match the existing hardware workflow:
optimization level 3, transpiler seed 11, captured backend translation and
scheduling plugins, and no unitary approximation. The entire preparation
body is passed to Qiskit once. Native readout is then appended at its final
mapped system locations, as in the hardware scripts. DD, ZNE and twirling
are not applied by this base compilation study.

## Physical layouts

The default `--layout chain` chooses a real CZ-connected path of N+2 physical
qubits and uses the same path across methods and seeds for a given N.
Its order is `[left bath, site0, ..., site(N-1), right bath]`.
Routing is disabled for a valid fixed chain, and the final mapping is
required to match the requested one. Selection uses connectivity, not the
historical experiment's calibration-quality thresholds.

To reproduce the earlier N=10 Model2 path explicitly:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py compile --sizes 10 --physical-path 106 116 115 114 113 112 111 101 91 81 80 90 --timing
```

The path must still be supported by the captured target. For multiple
sizes, `--layout-map paths.json` accepts a UTF-8 JSON object such as
`{"10": [106,116,115,114,113,112,111,101,91,81,80,90]}`; it must include every
requested N. `--avoid-qubits` excludes qubits during fixed-chain selection.

Long-path search is bounded by `--path-search-limit` (500000 expansions)
and `--path-search-seconds` (30 seconds). A search failure is recorded as
`layout_unavailable`; it is not a proof that the device has no suitable
path. Supply a path, increase the search limits, or explicitly choose
automatic placement and routing:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py compile --layout automatic --timing
```

Automatic layouts may differ across methods. Initial/final mappings and
observed routing-pass SWAP counts are saved. Native SWAP decomposition and
subsequent cancellation are reflected in the final native operation counts;
the routing-pass SWAP count is a diagnostic, not an independent additive
gate-cost formula.

## Readout, timing and saved artifacts

The default Z basis matches the earlier study's resource figures. To
include the hardware workflow's population/current readout settings:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py compile --bases z xy yx --timing
```

Counts are per preparation path and measurement basis. They include initial
product-mixture preparation measurements, retained bath resets and final
readout. They have no measurement-shot multiplier. Two compilation paths
do not establish finite-trajectory accuracy.

The output `phoenix_runs/phoenix_<hash>/` contains:

- `Results.md`, `resources.csv` and native scaling PNG/PDF figures.
- A sealed run manifest linking input, target, package versions and source.
- Per-path sealed checkpoints with complete operation counts, CZ and other
  two-qubit counts, depth, two-qubit depth, layouts, routing diagnostics and
  log files.
- With `--save-qpy`, complete native circuit files and their hashes. This
  can require substantial disk space; circuits are otherwise discarded
  after extracting metrics.

All native instructions are checked against the actual captured target.
Allocated backend width and active physical qubits are reported separately
from the N+2 source logical qubits. No `2NK` gate-count formula is imposed.

`--timing` uses Runtime's dynamic ALAP analysis and captured instruction
durations. It reports the sum of local quantum-block durations. Controller
and reset block-transition overhead is unknown, so this is a labelled
duration model, not measured end-to-end runtime. Timing failures remain
`unavailable` with a reason while valid gate/depth metrics remain usable.

`--timeout-seconds SECONDS` can stop an individual compiler worker; the
default zero means unlimited. `--max-source-exchanges COUNT` applies an
explicit workload cap if needed. Such paths remain capped/failed in the
tables and are never assigned approximate compiled resources. A run with
failed or capped paths returns a nonzero exit code after saving its report.

Regenerate reports without backend or target access:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py report FinalModel/Experiment1/phoenix_runs/phoenix_HASH
```

The target cache and generated Phoenix output folders remain local under
the experiment's ignore rules. To regenerate the portable schedule input
from the original local checkpoints:

```bash
.venv/bin/python FinalModel/Experiment1/phoenix_resources.py export-inputs
```

## Windows PowerShell

Use the same arguments, replacing `.venv/bin/python` with
`.\.venv\Scripts\python.exe`. For example:

```powershell
.\.venv\Scripts\python.exe .\FinalModel\Experiment1\phoenix_resources.py capture-target --account-file .\IBMRuntime\apikey.json
.\.venv\Scripts\python.exe .\FinalModel\Experiment1\phoenix_resources.py compile --sizes 2 4 10 --families warped_cosine --timing
```

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phoenix_compile_mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_phoenix_resources.py' -v
```

Tests use an explicitly synthetic target, never an IBM connection. They
check portable schedules, cache integrity, physical layout constraints,
agreement with the existing hardware circuit construction, actual native
transpilation/QPY metrics, dynamic timing, offline execution, resume and
explicit resource caps.
