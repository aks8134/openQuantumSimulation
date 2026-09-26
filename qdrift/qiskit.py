"""Qiskit interpreter for immutable :mod:`qdrift.core` programs."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from math import log2
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import HamiltonianGate, PauliEvolutionGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

from .core import HamiltonianTerm, QDriftProgram, QDriftStep


EvolutionFactory = Callable[[HamiltonianTerm[Any], float], Instruction]


def _matrix_qubit_count(operator: Any) -> int | None:
    try:
        matrix = np.asarray(operator, dtype=complex)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return None
    width = int(round(log2(matrix.shape[0])))
    return width if 2**width == matrix.shape[0] else None


def operator_qubit_count(operator: Any) -> int:
    """Infer the number of qubits on which a supported operator acts."""

    if isinstance(operator, str):
        if not operator or any(map(lambda symbol: symbol not in "IXYZ", operator.upper())):
            raise ValueError("a string operator must be a non-empty Pauli label")
        return len(operator)
    declared = getattr(operator, "num_qubits", None)
    if isinstance(declared, Integral) and not isinstance(declared, bool):
        return int(declared)
    matrix_width = _matrix_qubit_count(operator.data if isinstance(operator, Operator) else operator)
    if matrix_width is None:
        raise TypeError(
            "cannot infer operator width; use a Pauli label, Pauli/SparsePauliOp, "
            "Operator, square power-of-two matrix, or a custom evolution_factory"
        )
    return matrix_width


def default_evolution_factory(
    selected: HamiltonianTerm[Any],
    operator_time: float,
) -> Instruction:
    """Lower a standard Pauli or matrix operator to an evolution gate."""

    operator = selected.operator
    if isinstance(operator, str):
        pauli_operator = SparsePauliOp.from_list(((operator.upper(), 1.0),))
        return PauliEvolutionGate(pauli_operator, time=operator_time, label=selected.label)
    if isinstance(operator, Pauli):
        return PauliEvolutionGate(operator, time=operator_time, label=selected.label)
    if isinstance(operator, SparsePauliOp):
        return PauliEvolutionGate(operator, time=operator_time, label=selected.label)
    matrix = operator.data if isinstance(operator, Operator) else np.asarray(operator, dtype=complex)
    return HamiltonianGate(matrix, time=operator_time, label=selected.label)


def _term_width(selected: HamiltonianTerm[Any]) -> int:
    if selected.targets is not None:
        return len(selected.targets)
    return operator_qubit_count(selected.operator)


def _required_qubits(program: QDriftProgram[Any]) -> int:
    def extent(selected: HamiltonianTerm[Any]) -> int:
        return (
            max(selected.targets) + 1
            if selected.targets is not None
            else operator_qubit_count(selected.operator)
        )

    return max(map(lambda indexed: extent(indexed[1]), program.hamiltonian.active_terms), default=0)


def _required_target_extent(program: QDriftProgram[Any]) -> int:
    explicit = tuple(
        filter(
            lambda selected: selected.targets is not None,
            map(lambda indexed: indexed[1], program.hamiltonian.active_terms),
        )
    )
    return max(map(lambda selected: max(selected.targets) + 1, explicit), default=0)


def _circuit_qubit_count(
    program: QDriftProgram[Any],
    requested: int | None,
) -> int:
    if requested is None:
        return _required_qubits(program)
    if isinstance(requested, bool) or not isinstance(requested, Integral):
        raise TypeError("num_qubits must be an integer or None")
    converted = int(requested)
    if converted <= 0:
        raise ValueError("num_qubits must be positive")
    required_targets = _required_target_extent(program)
    if converted < required_targets:
        raise ValueError(
            f"num_qubits={converted} is smaller than the required {required_targets}"
        )
    return converted


def _step_targets(
    step: QDriftStep[Any],
    num_qubits: int,
    instruction_width: int,
) -> tuple[int, ...]:
    selected = step.term
    targets = selected.targets
    if targets is None:
        if instruction_width != num_qubits:
            raise ValueError(
                f"term {step.term_index} produces a {instruction_width}-qubit "
                f"instruction but the circuit has {num_qubits}; provide explicit targets"
            )
        return tuple(range(num_qubits))
    if len(targets) != instruction_width:
        raise ValueError(
            f"term {step.term_index} has {len(targets)} targets but its operator acts "
            f"on {instruction_width} qubits"
        )
    return targets


def _step_circuit(
    step: QDriftStep[Any],
    num_qubits: int,
    evolution_factory: EvolutionFactory,
) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    gate = evolution_factory(step.term, step.operator_time)
    if step.term.targets is not None and gate.num_qubits != _term_width(step.term):
        raise ValueError(
            f"evolution_factory returned a {gate.num_qubits}-qubit instruction for "
            f"a {_term_width(step.term)}-qubit term"
        )
    circuit.append(gate, _step_targets(step, num_qubits, gate.num_qubits))
    return circuit


def to_qiskit(
    program: QDriftProgram[Any],
    *,
    num_qubits: int | None = None,
    evolution_factory: EvolutionFactory = default_evolution_factory,
) -> QuantumCircuit:
    """Interpret a sampled qDRIFT program as a Qiskit circuit.

    Supply ``evolution_factory`` to support a custom operator representation.
    It receives the complete Hamiltonian term and the signed evolution time and
    must return a Qiskit ``Instruction``.
    """

    width = _circuit_qubit_count(program, num_qubits)
    if width <= 0:
        raise ValueError("the qDRIFT program does not define any circuit qubits")
    circuit = reduce(
        lambda accumulated, step: accumulated.compose(
            _step_circuit(step, width, evolution_factory)
        ),
        program.steps,
        QuantumCircuit(width, name="qdrift"),
    )
    circuit.metadata = {
        "algorithm": "qDRIFT",
        "evolution_time": program.evolution_time,
        "lambda_norm": program.lambda_norm,
        "sample_count": program.sample_count,
        "seed": program.seed,
        "sampled_term_indices": program.sampled_term_indices,
    }
    return circuit
