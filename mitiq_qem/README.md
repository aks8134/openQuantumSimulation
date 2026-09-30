# Functional Mitiq QEM facade

`mitiq_qem` is a small immutable planning layer around four selected Mitiq
techniques:

- zero-noise extrapolation (ZNE);
- CNOT/CZ Pauli twirling;
- digital dynamical decoupling (DDD);
- readout-error mitigation by confusion-matrix inversion.

The package deliberately does not submit jobs. Circuit construction and
classical inference are separate, so an experiment can checkpoint every
external execution and later combine the results. All plans, batches,
provenance entries, and results are frozen values.

The folder is named `mitiq_qem`, rather than `mitiq`, so it does not shadow the
upstream dependency.

## Installation

The mathematical planning, ZNE inference, and count-level readout correction
can be imported without Mitiq. Circuit transformations load Mitiq lazily:

```bash
uv pip install "mitiq[qiskit]"
```

If the optional dependency is absent, transformation functions raise
`MitiqUnavailable` with the installation command.

## ZNE

```python
from mitiq_qem import (
    combine_zne,
    construct_zne,
    inference,
    scaling,
    zne_plan,
)

plan = zne_plan(
    (1.0, 2.0, 3.0),
    repetitions=4,
    scaling=scaling("random_local"),
    inference=inference("richardson"),
    seed=7,
)

batch = construct_zne(circuit, plan)
# Execute batch.circuits externally, preserving this order.
result = combine_zne(plan, tuple(expectation_values))
print(result.estimate, result.scale_standard_errors)
```

Supported scaling methods are `random_local`, `fold_all`, and `global`.
Supported inference methods are `linear`, `richardson`, and `polynomial`.

For a Qiskit circuit containing resets, local folding is independently applied
to each unitary region and every reset is retained at its original boundary.
Global folding is rejected because a reset has no unitary inverse. Classical
control and mid-circuit measurements are rejected according to Mitiq's
non-adaptive circuit model. Terminal measurements are retained.

## Pauli twirling

```python
from mitiq_qem import (
    combine_pauli_twirling,
    construct_pauli_twirling,
    pauli_twirling_plan,
)

plan = pauli_twirling_plan(variants=16, seed=11)
batch = construct_pauli_twirling(circuit, plan)
result = combine_pauli_twirling(tuple(expectation_values))
```

This follows Mitiq's Pauli-twirling implementation, which twirls CNOT and CZ
gates. `variants` is the number of independently randomized circuits whose
expectation values are averaged.

## Digital dynamical decoupling

```python
from mitiq_qem import combine_ddd, construct_ddd, ddd_plan

plan = ddd_plan("xyxy", spacing=-1, trials=1)
batch = construct_ddd(circuit, plan)
result = combine_ddd(tuple(expectation_values))
```

The built-in rules are `xx`, `yy`, and `xyxy`. A spacing of `-1` asks Mitiq to
choose the largest spacing that fits. DDD is a gate-level approximation: the
backend compiler and scheduler can change or remove the intended idle-window
structure, so the compiled circuit must be inspected.

## Readout mitigation

For independent identical readout errors:

```python
from mitiq_qem import (
    diagonal_expectation,
    mitigate_counts,
    uncorrelated_readout_plan,
)

plan = uncorrelated_readout_plan(
    2,
    p0=0.02,  # P(measure 1 | prepared 0)
    p1=0.03,  # P(measure 0 | prepared 1)
)
result = mitigate_counts(plan, counts)
zz = diagonal_expectation(result, (1.0, -1.0, -1.0, 1.0))
```

`mitigate_counts` returns signed quasi-probabilities; it does not silently clip
negative values or fabricate integer counts. A calibrated inverse matrix can be
provided with `readout_plan(matrix)`. Existing Mitiq `MeasurementResult` values
can be corrected through `mitigate_measurements(plan, result)`.

## Functional composition

Every circuit constructor accepts either one circuit or an existing
`CircuitBatch`. The recommended Mitiq ordering can therefore be expressed as:

```python
scaled = construct_zne(circuit, zne_configuration)
twirled = construct_pauli_twirling(scaled, twirling_configuration)
decoupled = construct_ddd(twirled, ddd_configuration)
circuits_to_execute = decoupled.circuits
```

Each final circuit retains a tuple of `Stage` values describing its scale
factor, twirling variant, DDD trial, and random seeds. After execution, readout
correction is applied first; DDD trials and twirling variants are then averaged;
ZNE inference is applied last.

## Deliberately excluded

PEC, CDR, QSE, LRE, virtual distillation, shadows, and other experimental
methods are outside this package's current scope.
