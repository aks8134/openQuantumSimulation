# Experiment 9: combined mitigation on Experiment 8

Experiment 9 keeps the complete Experiment 8 calculation:

- the Experiment 7 randomized single-boundary dilation;
- explicit `RZ`/`RX`/`RZZ` synthesis at the compiler boundary;
- the three measurement bases `Z`, `X`, and alternating `XY`;
- native-qem local-folding zero-noise extrapolation (ZNE); and
- the exact vacuum-plus-single-excitation reference when requested.

It adds three independently switchable native [`qem`](../../qem/README.md)
mitigation layers: Pauli twirling, duration-aware dynamical decoupling (DD),
and calibrated block readout mitigation. ZNE also comes from `qem`. Run in
the main environment; no Mitiq, Cirq, worker process or second venv is used:

```bash
source .venv/bin/activate
```

## Local noisy FakeKingston / FakeMiami baseline

`--backend fake_kingston` compiles against the installed `FakeKingston` device
target, then samples locally with `AerSimulator.from_backend(FakeKingston())`.
This uses its bundled calibration-snapshot noise model (including gate and
readout errors), basis and connectivity. It is **not** Phoenix's noise profile
or a prediction of current hardware behavior. No credentials, network access,
snapshot refresh or real QPU submission are required. `--aer-method` selects
the local simulation method; the default is `automatic`.

`--backend fake_miami` uses the same pipeline with `FakeMiami`'s own installed
target and calibration-snapshot noise via `AerSimulator.from_backend(FakeMiami())`.
Backend resolution and reset-free sampling are supported, and filenames are
backend-specific. **The currently installed Miami snapshot omits `reset`: the
Experiment 9 boundary-dilation circuits therefore fail compilation on this
target.** No ideal reset instruction or reset-error calibration is silently
invented. Supporting this particular experiment on Miami requires an explicitly
approved local-simulation reset model first; the command below remains for
FakeKingston only.

For N=10 with all mitigation disabled:

```bash
uv run python Model1/Experiment9/mitigated_randomized_lie_trotter.py \
  --n-qubits 10 \
  --backend fake_kingston \
  --trajectories 4 \
  --fold-repetitions 1 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --optimization-level 3 \
  --optimized-classical-reference \
  --no-zne \
  --no-pauli-twirling \
  --no-dynamical-decoupling \
  --no-readout-mitigation \
  --batch-size 6
```

This builds 72 circuits (6 times x 4 trajectories x 3 bases), using 2048
shots per circuit. Existing batching/checkpointing and figures are retained;
filenames contain `fake_kingston` and the mitigation flags. Physical indices
are preserved through sampling so calibration errors stay attached to the
correct device wires. Aer truncates unused wires internally; routing may use
additional active qubits and noisy simulation can be slower than ideal Aer.
The separate classical reference remains the ideal Lindblad solution.

Do not supply the synthetic Aer DD timings for this baseline. If subsequently
enabling readout mitigation, supply a matching calibration file: noisy fake
backends do **not** receive the noiseless Aer identity correction. `fake_fez`
retains its existing layout-only behavior.

Use `--no-layout` to skip the topology/mapping figure, its embedded circuit
panels, and the extra final-time transpilation used for that figure. Observable,
transpilation-metrics, ZNE-scaling, separate one-step circuit and exchange-
decomposition figures remain enabled. The option does not affect samples,
mitigation or checkpoint compatibility, cannot be combined with `--layout-only`,
and does not delete any existing layout image. Result metadata marks the layout
disabled; the final summary does not claim it was generated. Independently use
`--no-circuit-plots` if you also want to skip the separate circuit diagrams.

## Order of operations

For every Experiment 7 base circuit, the executable is constructed in the
following order:

1. Explicitly synthesize each logical `XXPlusYY` block as in Experiment 7.
2. Perform the one optimizing backend transpilation and choose the physical
   layout.
3. Apply `qem.fold` at each requested ZNE scale.
4. Generate a native Pauli-twirled variant through `qem.twirl`.
5. ALAP-schedule the circuit and insert timed DD via `qem.apply_dd`.
6. Sample the native circuit without further lowering or optimization.
7. Correct measured parity expectations through `qem.mitigate_parity`.
8. Average trajectories/fold realizations and fit via `qem.extrapolate`.

Gate representations and physical layouts remain intact; there is no second
compilation that could cancel folds or expand them through a conversion.
Unsupported target gates, inverses, or angle parameters produce explicit
errors. See Experiment 8 for the exact `SX^dagger = SX^3` native fallback and
requested-versus-realized scaling caveats.

## Pauli twirling

`qem` twirls Clifford entanglers \(U\in\{CZ,CX\}\) by selecting a random
two-qubit Pauli
\(P\) is inserted before the gate and

\[
Q = U P U^\dagger
\]

is inserted after it. Since Clifford conjugation maps Paulis to Paulis,

\[
QUP = U
\]

up to a global phase. The ideal circuit is unchanged. The random realization
is reproducible from `--seed-twirling`; ensemble averaging occurs jointly over
the already existing qDRIFT trajectories and fold repetitions, so Pauli
twirling does not introduce another circuit-count dimension.

Paulis are synthesized directly using target-supported X/RX/SX and virtual
RZ gates. `RXX`, `RYY`, and `RZZ` are also twirled: when a Pauli anticommutes
with the rotation generator, the angle is negated to preserve the ideal gate.
Other two-qubit gates (including ECR) are currently left untwirled and are
explicitly reported. Unsupported negative-angle rotations are rejected rather
than automatically lowered. The result records twirled and untwirled types.

## Dynamical decoupling

After folding and twirling, `qem.apply_dd` uses Qiskit's duration-aware ALAP
scheduling and padding passes. Sequences are `xx`, `yy`, `xy4`, and `xy8`;
`xyxy` remains an alias for `xy4`. Only active physical wires are selected.
Hardware pulse/acquisition alignment constraints are honored. Idle intervals
after initialization/reset are skipped by default.

Plain Aer has no calibrated gate durations: use `--no-dynamical-decoupling`
or explicitly provide `--dd-durations timings.json`. The file must contain
`dt` in seconds and `instructions` accepted by Qiskit `InstructionDurations`,
for example `{"dt": 1e-9, "instructions": [["x", null, 16, "dt"],
["measure", null, 32, "dt"]]}`. Supply durations for **all** compiled gate
types, including rotation/virtual/reset instructions as applicable. A timing
file does not itself add an Aer noise model. Hardware uses target timings;
`--dd-durations` is Aer-only. Missing timing is an error, not silent DD skipping.

A **synthetic** timing file is supplied at [`aer_timings.json`](aer_timings.json).
Use `--dd-durations Model1/Experiment9/aer_timings.json` to test scheduling and
DD insertion on plain Aer. Its assumed values are: `dt = 1 ns`, 40 ns physical
single-qubit gates (20 ns SX/SXdg), 300 ns two-qubit gates, zero-duration
RZ/P/U1/phase, and 1000 ns measurement/reset. Other placeholder durations and
limitations are explicitly recorded in the JSON. They are not IBM calibrations,
and there is no added noise or decoherence. Do not interpret this as a physical
DD-effectiveness test. Timing data is already included in result/checkpoint
compatibility metadata; no existing results are changed by supplying this file.

With both alignments equal to 1 dt, `qem.apply_dd` omits the unnecessary
`ConstrainedReschedule` pass, avoiding its opaque-`unitary` rejection on Aer.
ALAP/ASAP scheduling and DD padding still run with the supplied durations;
there is no gate lowering. Stricter hardware alignments retain the alignment
pass. Offline regression checks cover N=10 single-step left/right and
ten-substep circuits with folding, twirling and DD at scales 1, 1.5 and 2.
No sampling or hardware execution is used for these checks. The timing file
does not guarantee support for every possible instruction or timing model.

`--dd-spacing-fractions` accepts fractions of the idle interval before,
between and after pulses: pulse-count + 1 nonnegative values summing to one.
The default is balanced spacing. The old integer `--dd-spacing` is obsolete;
only its default `-1` is accepted for command compatibility.

## Readout mitigation

Readout mitigation never constructs a dense \(2^N\times2^N\) matrix. It solves
small assignment-block adjoint systems for the measured parity and evaluates
their weights against the sampled counts. Blocks may distinguish the two
assignment directions and include within-block correlations; independence is
assumed between blocks. Blocks are limited to eight qubits.

Hardware requires `--readout-calibration calibration.json` (or disable readout
mitigation). No backend average measurement error is treated as a measured
assignment matrix. Calibration must cover every measured physical wire in
every compiled variant. A file has this schema:

```json
{
  "backend": "ibm_kingston",
  "source": "description/date of the measured calibration",
  "blocks": [
    {"physical_qubits": [85], "assignment": [[0.97, 0.06], [0.03, 0.94]]},
    {"physical_qubits": [86], "assignment": [[0.98, 0.04], [0.02, 0.96]]}
  ]
}
```

This is a schema example, **not a valid calibration for a real run**. Entries
are `A[observed, prepared]`; each column sums to one. Block qubits are in
least-significant-first order. Overlapping or partially measured blocks are
rejected. `qem.calibration_circuits` and `qem.fit_readout_model` can build these
matrices, but this experiment does not automatically execute calibration jobs.

Actual terminal measurement destinations select calibration blocks, accounting
for routing SWAPs rather than assuming the initial layout is the final mapping.
The matrices/source are saved per variant for checkpoint recovery. Correction
is deterministic, without clipping or resampling. Shot variance uses corrected
parity weights; calibration uncertainty is not included. For the existing
noiseless Aer path only, absent calibration means explicitly labelled identity
matrices—it is not evidence of measured readout mitigation performance.

## Running

Example Aer run:

```bash
python Model1/Experiment9/mitigated_randomized_lie_trotter.py \
  --n-qubits 7 \
  --backend aer \
  --trajectories 4 \
  --fold-repetitions 2 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --zne-scale-factors 1 1.5 2 \
  --zne-inference linear \
  --no-dynamical-decoupling \
  --optimization-level 3 \
  --optimized-classical-reference
```

For hardware, select an accessible IBM backend, and provide a measured
`--readout-calibration FILE` or `--no-readout-mitigation`. Omit
`--no-dynamical-decoupling` to enable target-timed DD.
As in Experiments 4--8, a non-Aer execution submits real QPU jobs.

## Independent off switches

- `--no-zne`: submit only the unfurled scale-one circuit. It requires
  `--fold-repetitions 1`.
- `--no-pauli-twirling`: do not add random Pauli frames.
- `--no-dynamical-decoupling`: do not schedule or insert DD pulses.
- `--no-readout-mitigation`: reconstruct observables from raw counts.

DD is configured with `--dd-sequence xx|yy|xy4|xy8` and optional
`--dd-spacing-fractions`; the default is balanced `xx`.

For an Experiment 7-like unmitigated baseline while retaining Experiment 9's
three-basis code path:

```bash
python Model1/Experiment9/mitigated_randomized_lie_trotter.py \
  --n-qubits 7 \
  --backend aer \
  --trajectories 4 \
  --fold-repetitions 1 \
  --shots 8192 \
  --times 0 0.2 0.4 0.6 0.8 1.0 \
  --trotter-delta-t 0.1 \
  --no-zne \
  --no-pauli-twirling \
  --no-dynamical-decoupling \
  --no-readout-mitigation
```

All ordinary Experiment 8 controls remain available, including arbitrary
`--times`, `--batch-size`, checkpoint/resume, circuit-plot suppression,
layout-only compilation, optimization level, and both classical-reference
modes. Filenames explicitly encode Pauli twirling (`pauli_on` / `pauli_off`),
DD (`dd_off` or, for example, `dd_xx`), and readout mitigation
(`readout_on` / `readout_off`). ZNE is encoded
as `zne` / `no_zne`. This prevents unlike runs from silently overwriting one
another. The generic `fully_mitigated` label is omitted because individual
techniques can be disabled.

Existing archives are not renamed or modified. New names start with `qem_`.
Folding/mitigation version 3 prevents combining old Mitiq runs with these new
algorithms, even if an old path is explicitly supplied. Matching new runs keep
the existing append-times/replace-matching-times and checkpoint behavior.
Calibration matrices and timing settings are part of compatibility checks.

## Circuit count and shots

### Execution environment

Run this script with `.venv/bin/python` on Linux or
`.venv\Scripts\python.exe` on Windows. All mitigation now runs through `qem`
in this same process. The previous worker environment can remain on disk, but
Experiment 8/9 no longer imports it or depends on it.

Pauli twirling and DD transform an already existing submitted variant and do
not multiply the circuit count. Readout mitigation uses a supplied calibration
file and adds no calibration circuits. Therefore the production count is
unchanged from Experiment 8:

\[
N_\mathrm{circuits}
= N_t\,N_\mathrm{basis}\,R\,N_\lambda\,F,
\qquad N_\mathrm{basis}=3,
\]

where \(R\) is the number of qDRIFT trajectories, \(N_\lambda\) the number of
ZNE scales, and \(F\) the number of fold repetitions. `--shots` remains the
total shots per saved-time/basis/scale, divided evenly over \(R F\).

## Within-batch progress

Progress is enabled by default. Install the UI dependency in the main venv:

```bash
uv pip install --python .venv/bin/python -r Model1/Experiment9/requirements-progress.txt
```

On Windows, use `--python .venv\Scripts\python.exe` instead. This adds only
[`tqdm`](https://tqdm.github.io/docs/tqdm/); no mitigation worker is involved.

Each batch shows exact counts for preparing compiler inputs and native
mitigation variants. Backend resolution, bulk transpilation and sampling show
an elapsed-time indicator refreshed once per second. Aer and IBM Sampler return
one complete batch result, so these waiting stages deliberately show **no
per-circuit completion percentage or ETA**. Preparation percentages are not
simulation percentages. IBM queue/execution status is not polled by this UI.

The implementation does not split batches, add jobs, change shots/seeds or
alter checkpoint intervals. Resuming shows progress for the remaining work.
Use `--no-progress` to disable all bars and the refresh thread (and avoid
requiring the UI dependency). Other experiments are unchanged.

## Outputs

Experiment 9 writes its own `figures/` and `results/` paths. The JSON includes
the Experiment 8 raw counts, scale observables, extrapolation weights,
uncertainties, folding diagnostics, exact reference (when requested), and new
per-circuit twirling, DD, layout, and readout-calibration metadata. Existing
Experiment 7 and Experiment 8 results are not modified.
