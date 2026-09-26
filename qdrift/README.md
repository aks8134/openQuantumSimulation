# Functional qDRIFT library

This package represents a Hamiltonian decomposition and a sampled qDRIFT
program as immutable values. The mathematical sampler is independent of
Qiskit; `to_qiskit` is an interpreter at the package boundary.

For

\[
H=\sum_i h_iH_i,
\qquad
\lambda=\sum_i |h_i|\lVert H_i\rVert,
\]

qDRIFT selects term \(i\) with probability
\(p_i=|h_i|\lVert H_i\rVert/\lambda\). With sample complexity \(N_s\), each
selection applies

\[
\exp\!\left[-i\,\operatorname{sgn}(h_i)
\frac{\lambda t}{N_s\lVert H_i\rVert}H_i\right].
\]

`sample_count` is therefore the number of sampled exponentials in one random
realization. It is not the number of independent circuits used to estimate an
ensemble average.

## Example

The following constructs \(H=0.7X_0X_1-0.2Z_1\) without mutating an existing
Hamiltonian value:

```python
from qdrift import add_term, empty_hamiltonian, qdrift, to_qiskit

empty = empty_hamiltonian()
with_xx = add_term(empty, 0.7, "XX", targets=(0, 1), label="XX")
model = add_term(with_xx, -0.2, "Z", targets=(1,), label="Z1")

program = qdrift(
    model,
    evolution_time=1.0,
    sample_count=200,
    seed=7,
)
circuit = to_qiskit(program, num_qubits=2)
```

Pauli operators have norm one, so the default `operator_norm=1.0` is correct.
For a custom Hermitian operator, pass its spectral norm explicitly:

```python
model = add_term(
    empty_hamiltonian(),
    h_i,
    H_i,
    operator_norm=norm_H_i,
    targets=(0, 1),
    label="custom term",
)
```

NumPy matrices, Qiskit `Operator`, `Pauli`, and `SparsePauliOp` values are
lowered automatically. Other representations can be supported without
changing the sampler by passing a custom `evolution_factory` to `to_qiskit`.

The retained term metadata—coefficient, operator, norm, targets, and label—is
intended to support a later state- and observable-aware decomposition optimizer.

