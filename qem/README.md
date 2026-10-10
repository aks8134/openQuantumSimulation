# Native functional QEM

`qem` implements our own gate-folding, extrapolation, gate-aware Pauli
twirling, and calibrated readout-correction algorithms. Scheduled DD uses
Qiskit's scheduler and `PadDynamicalDecoupling` pass. Nothing here imports
Mitiq/Cirq, starts a worker, accesses credentials, or submits provider jobs.
Existing experiments and `mitiq_qem` are not migrated by this library.

The design follows the repository's functional/OCaml-style conventions:
immutable plans and results, explicit seeds, transformations returning new
circuits, and `Ok`/`Err` plus `bind`/`pipe_result` for composition. Configuration
errors raise `ValueError`/`TypeError`; execution/numerical failures return
`Err(Failure(...))`. Circuit payloads are Qiskit objects and remain mutable by
Qiskit's nature: treat returned circuits as owned values. Inputs are not modified.

Importing `qem` loads only the standard library. NumPy and Qiskit are loaded
when the relevant algorithm is called, from the active environment (normally
the repository's `.venv`). No additional environment is needed. This removes
Mitiq startup, but does not eliminate filesystem latency when importing Qiskit.

## Folding without conversion or automatic lowering

```python
from qiskit import QuantumCircuit
from qem import FoldPlan, Ok, fold

circuit = QuantumCircuit(2)
circuit.rx(0.3, 0)
circuit.rzz(0.2, 0, 1)
circuit.reset(1)

match fold(circuit, FoldPlan(scale_factor=2, seed=7)):
    case Ok(result):
        folded = result.circuit
        print(result.before.operations, result.after.operations)
        print(result.selection.achieved_scale)
        print(result.represented_weight_scale)
    case failure:
        print(failure)
```

The original two eligible gates stay intact. One randomly selected gate
receives an inverse/forward pair, so this example grows from 3 operations to
5, not into a different basis. Resets, measurements, barriers and delays are
preserved, and gates never move across them. By default only `rz` is excluded
as a virtual gate; other exclusions and eligible `gate_names` are explicit.
Control-flow blocks are currently rejected rather than partially mitigated.

For a gate selected k times the chronological replacement is
`G (G† G)^k`. Integer fold counts are selected to approximate the requested
scale. `selection.achieved_scale` reports the rounded weighted selection,
while `represented_weight_scale` includes any extra instructions needed by
a custom inverse factory. Weights default to one per eligible gate and can
be set with `weights=(("rzz", 5.0), ...)`. These are declared cost/noise proxies,
not measured physical noise factors or guaranteed hardware pulse counts.

Pass `target=backend.target` to validate support on the unchanged physical
qubits, including parameters. A native gate set need not support every
inverse. Unsupported output returns `Err`: it is **not silently lowered**.
An explicit `inverse_factory(gate)` may return a same-width tuple of native
gates implementing the exact inverse, including phase. The caller must verify
that custom synthesis. Standard native inverse rotations remain rotations.

Keep post-folding optimization disabled in the caller: optimizing away the
inserted identity pairs would remove the intended noise scaling. A target
support check alone is not a certificate of hardware timing or runtime limits.

## Extrapolation and uncertainty

```python
from qem import ZNEPlan, extrapolate

estimate = extrapolate(
    ZNEPlan((1.0, 1.5, 2.0), inference="linear"),
    (0.82, 0.77, 0.72),
    standard_errors=(0.01, 0.01, 0.01),
)
```

Supported inference is `linear`, `polynomial` (explicit `degree`),
`richardson` (degree number-of-scales minus one), and `none` at scale one.
Use consistently defined **achieved** scales when folding is quantized.
Repeated realized scales must be aggregated before constructing a ZNEPlan.
The least-squares intercept uses SVD-based solves, not normal-equation inversion.
Ill-conditioned fits return `Err` using the plan's configurable limit.

Reported weights give `estimate = sum(w_i * y_i)`. Supply either pointwise
standard errors (independent estimates) or a full covariance matrix; propagated
variance is `w.T @ covariance @ w`. These uncertainties do not change the
unweighted fit. Without supplied uncertainty, `standard_error` is `None`.
No estimate is silently clipped to observable bounds.

## Gate-aware Pauli twirling

```python
from qem import TwirlingPlan, twirl

randomized = twirl(
    circuit,
    TwirlingPlan(seed=19, gate_names=("rzz",)),
    # target=backend.target,  # optional native-support validation
)
```

CX/CZ use conjugate Pauli frames with exact global-phase compensation.
RX/RY/RZ and RXX/RYY/RZZ use rotation-aware randomization:
`P U(s*theta) P = U(theta)`, with angle sign flipped when P anticommutes with
the generator. Arbitrary controlled rotations are **not** treated as Clifford
gates. Each result records choices, its realized seed, metrics and unselected
two-qubit gate names. Default selected gates are CX, CZ and RZZ; opt into other
supported rotation types explicitly.

Native Pauli synthesis uses supported X (or RX(pi)/SX pairs) and RZ frames
where needed, with phase compensation. It does not call a gate-lowering pass.
This does add physical Pauli pulses, which can have a performance/noise cost;
twirling alone is not guaranteed to improve an observable. Ensemble execution
and averaging remain the caller's responsibility: use distinct recorded seeds.

## Scheduled dynamical decoupling

```python
from qem import DDPlan, apply_dd

decoupled = apply_dd(
    circuit,  # already compiled and physically placed for this target
    DDPlan(sequence="xy4", skip_reset_qubits=True),
    target=backend.target,
)
```

Sequences are `xx`, `yy`, `xy4` and `xy8`; ALAP and ASAP scheduling, selected
physical qubits and custom inter-pulse spacing are configurable. Qiskit
scheduling, constrained rescheduling and padding use measured instruction
durations and target pulse/acquisition alignment. Y frame components, when
synthesized with X/RZ, have zero internal spacing so the frame change stays
adjacent to its physical pulse. Default DD skips windows immediately following
reset/initialization. Windows too short to fit a sequence are left unmodified.
When both pulse and acquisition alignments are 1 dt, constrained rescheduling
is omitted because every scheduled start is already aligned. Scheduling and DD
padding still run; gates are not lowered. Stricter hardware alignments retain
the constrained-rescheduling pass.

Explicit `InstructionDurations` can be supplied instead of a target for local
tests; otherwise missing timing data is an error, not a silent no-op. Ideal Aer
does not by itself supply calibrated hardware idle durations. DD must run
after folding/twirling because they invalidate the original schedule. Inspect
`inserted_gate_count`: a valid schedule may have no usable idle windows.
No optimizing transpilation should be performed afterward without checking
that the mitigation has survived.

The timing implementation deliberately uses documented Qiskit behavior:
[PadDynamicalDecoupling](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.PadDynamicalDecoupling)
and [ALAPScheduleAnalysis](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.ALAPScheduleAnalysis).
It is not an independent pulse scheduler or a wrapper around Mitiq's DDD.

## Calibrated, scalable readout correction

```python
from qem import Ok, calibration_circuits, fit_readout_model, mitigate_parity

batch_result = calibration_circuits(
    ((0,), (1,)),
    physical_qubits=(4, 7),
    num_qubits=8,
)
if isinstance(batch_result, Ok):
    batch = batch_result.value
    # Submit batch.circuits yourself, with fixed physical placement.
    # Then supply counts in the same circuit order:
    model_result = fit_readout_model(batch, calibration_counts)
    if isinstance(model_result, Ok):
        corrected = mitigate_parity(observable_counts, model_result.value, (0, 1))
```

`physical_qubits[b]` identifies the physical qubit measured into classical bit
b. Model bits refer to **classical outcome positions**, not logical circuit
labels. The caller must use the correct measurement mapping from its compiled
circuit and preserve calibration placement during compilation/execution.
The library neither guesses that mapping nor accesses IBM credentials.

Matrices use `A[observed, prepared]` (column stochastic). Independent one-bit
blocks need two parallel calibration circuits; blocks require
`2**max_block_width` circuits. Disjoint blocks assume no inter-block readout
correlations; within-block correlations are retained. Calibration is specific
to the backend, qubits, timing and measurement setup and can become stale.
Backend scalar readout-error summaries are not substituted for measured matrices.

`mitigate_marginal` solves for a small corrected probability vector. Blocks and
the union needed for a marginal are capped at 8 bits to avoid accidental
exponential allocations. `mitigate_parity` solves `A_b.T w_b = f_b`, then
averages products of local weights over observed outcomes. It supports large
systems without materializing their full distribution. This applies to parity
in whichever basis was actually measured; it cannot recover unmeasured bases.

`ReadoutPlan` selects linear solve or pseudoinverse, cutoff and conditioning
limits. Negative quasi-probabilities and out-of-bound expectation estimates are
retained. Optional simplex projection is explicit and may introduce bias.
Parity correction cannot apply that projection. Reported shot uncertainty
treats the calibration matrices as fixed: calibration uncertainty and drift
are not included. Ill-conditioned calibrations normally return `Err`.

## Composition and verification

```python
from qem import FoldPlan, TwirlingPlan, fold, pipe_result, twirl

result = pipe_result(
    fold(circuit, FoldPlan(2.0, seed=3)),
    lambda folded: twirl(folded.circuit, TwirlingPlan(seed=5)),
)
```

Keep intermediate values when retaining every stage's diagnostics. Compose
folding → twirling → scheduled DD; apply readout correction to returned counts
and then extrapolate scale-dependent corrected observable estimates. Allocate
shots/seeds and account for shared-data covariance in the experiment driver.
All techniques remain independently usable; there are no implicit enabled stages.

Tests use small matrices, native circuit identities, synthetic calibration data
and local scheduling. They do not submit hardware jobs or run existing experiments:

```bash
.venv/bin/python -m unittest tests.test_qem -v
```
