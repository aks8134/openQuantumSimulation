"""Calibrated block readout mitigation; no assumed backend error profiles.

Assignment matrices use A[observed, prepared]. A block's bits are in local
least-significant-first order and refer to CLASSICAL outcome positions.
Disjoint blocks are assumed independent; correlations inside a block are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral
from typing import Any

from .core import Failure, Result, attempt, indices, real


@dataclass(frozen=True, slots=True)
class ReadoutBlock:
    bits: tuple[int, ...]
    assignment: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        bits = indices(self.bits, "bits")
        if len(bits) > 8:
            raise ValueError("a readout block is limited to 8 bits; use smaller blocks")
        matrix = tuple(tuple(real(value, "assignment probability") for value in row)
                       for row in self.assignment)
        dimension = 1 << len(bits)
        if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
            raise ValueError("assignment matrix must have shape 2**len(bits) squared")
        if any(value < 0 or value > 1 for row in matrix for value in row):
            raise ValueError("assignment entries must be probabilities")
        if any(abs(sum(row[column] for row in matrix) - 1) > 1e-8 for column in range(dimension)):
            raise ValueError("each assignment column must sum to one")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "assignment", matrix)


@dataclass(frozen=True, slots=True)
class ReadoutModel:
    blocks: tuple[ReadoutBlock, ...]
    source: str = "explicit user-supplied assignment matrices"

    def __post_init__(self):
        blocks = tuple(self.blocks)
        if not blocks or any(not isinstance(block, ReadoutBlock) for block in blocks):
            raise ValueError("blocks must contain ReadoutBlock values")
        bits = tuple(bit for block in blocks for bit in block.bits)
        if len(set(bits)) != len(bits):
            raise ValueError("readout blocks must be disjoint")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be an explicit nonempty description")
        object.__setattr__(self, "blocks", blocks)

    @property
    def bits(self):
        return tuple(sorted(bit for block in self.blocks for bit in block.bits))


@dataclass(frozen=True, slots=True)
class ReadoutPlan:
    method: str = "solve"
    rcond: float = 1e-10
    max_condition: float | None = 1e8
    project_probabilities: bool = False

    def __post_init__(self):
        if self.method not in ("solve", "pinv"):
            raise ValueError("method must be solve or pinv")
        cutoff = real(self.rcond, "rcond")
        if not 0 < cutoff < 1:
            raise ValueError("rcond must be between zero and one")
        object.__setattr__(self, "rcond", cutoff)
        if self.max_condition is not None:
            limit = real(self.max_condition, "max_condition")
            if limit <= 1:
                raise ValueError("max_condition must exceed one")
            object.__setattr__(self, "max_condition", limit)
        if not isinstance(self.project_probabilities, bool):
            raise TypeError("project_probabilities must be boolean")


@dataclass(frozen=True, slots=True)
class CalibrationBatch:
    circuits: tuple[Any, ...]
    blocks: tuple[tuple[int, ...], ...]
    physical_qubits: tuple[int, ...]
    prepared_states: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class ReadoutEstimate:
    bits: tuple[int, ...]
    probabilities: tuple[float, ...]
    shots: int
    condition_number: float
    projected: bool


@dataclass(frozen=True, slots=True)
class ReadoutExpectation:
    estimate: float
    standard_error: float | None
    shots: int
    condition_numbers: tuple[float, ...]


def _outcome(key):
    if isinstance(key, bool):
        raise TypeError("outcomes cannot be booleans")
    if isinstance(key, Integral):
        value = int(key)
    elif isinstance(key, str):
        text = "".join(key.split())
        if text.lower().startswith("0x"):
            value = int(text, 16)
        else:
            value = int(text, 2)
    else:
        raise TypeError("outcomes must be binary/hex strings or integers")
    if value < 0:
        raise ValueError("outcomes must be nonnegative")
    return value


def _counts(counts):
    entries = []
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("counts must be nonnegative integers, not quasi-probabilities")
        entries.append((_outcome(key), int(value)))
    if sum(count for _, count in entries) == 0:
        raise ValueError("counts must contain at least one shot")
    return tuple(entries)


def _pack(outcome, bits):
    return sum(((outcome >> bit) & 1) << position for position, bit in enumerate(bits))


def calibration_circuits(blocks, *, physical_qubits=None, num_qubits=None) -> Result[CalibrationBatch, Failure]:
    """Return parallel block calibration circuits, without execution.

    blocks partition classical bits 0..m-1. physical_qubits[b] identifies the
    physical qubit measured into bit b. Compile with that placement fixed before
    hardware execution. For independent one-bit blocks only two circuits are
    needed; general blocks require 2**max_block_width circuits.
    """
    def build():
        from qiskit import QuantumCircuit
        from .core import integer
        normalized = tuple(indices(block, "block bits") for block in blocks)
        if not normalized or any(len(block) > 8 for block in normalized):
            raise ValueError("provide nonempty calibration blocks of at most 8 bits")
        flattened = tuple(bit for block in normalized for bit in block)
        if sorted(flattened) != list(range(len(flattened))):
            raise ValueError("calibration blocks must partition classical bits 0..m-1")
        physical = (tuple(range(len(flattened))) if physical_qubits is None
                    else indices(physical_qubits, "physical_qubits"))
        if len(physical) != len(flattened):
            raise ValueError("one physical qubit is required per classical bit")
        width = max(physical) + 1 if num_qubits is None else integer(num_qubits, "num_qubits", minimum=1)
        if width <= max(physical):
            raise ValueError("physical qubit is outside num_qubits")
        circuits, states = [], []
        for index in range(1 << max(map(len, normalized))):
            circuit = QuantumCircuit(width, len(physical), name=f"readout_calibration_{index}")
            prepared = tuple(index % (1 << len(block)) for block in normalized)
            for block, value in zip(normalized, prepared):
                for position, bit in enumerate(block):
                    if (value >> position) & 1:
                        circuit.x(physical[bit])
            for bit, qubit in enumerate(physical):
                circuit.measure(qubit, bit)
            circuits.append(circuit)
            states.append(prepared)
        return CalibrationBatch(tuple(circuits), normalized, physical, tuple(states))
    return attempt("readout-calibration-circuits", build)


def fit_readout_model(batch: CalibrationBatch, counts) -> Result[ReadoutModel, Failure]:
    """Estimate columns from measured calibration counts, aggregating repeats."""
    def fit():
        data = tuple(_counts(value) for value in counts)
        if len(data) != len(batch.prepared_states) or len(data) != len(batch.circuits):
            raise ValueError("one counts dictionary is required per calibration circuit")
        blocks = []
        for block_index, bits in enumerate(batch.blocks):
            dimension = 1 << len(bits)
            matrix = [[0] * dimension for _ in range(dimension)]
            for entries, prepared in zip(data, batch.prepared_states):
                column = prepared[block_index]
                if not 0 <= column < dimension:
                    raise ValueError("invalid prepared calibration state")
                for outcome, count in entries:
                    if outcome >= (1 << len(batch.physical_qubits)):
                        raise ValueError("calibration outcome exceeds its classical width")
                    matrix[_pack(outcome, bits)][column] += count
            totals = tuple(sum(row[column] for row in matrix) for column in range(dimension))
            if any(total == 0 for total in totals):
                raise ValueError("every block preparation needs measured counts")
            probabilities = tuple(tuple(value / totals[column] for column, value in enumerate(row)) for row in matrix)
            blocks.append(ReadoutBlock(bits, probabilities))
        return ReadoutModel(tuple(blocks), source="measured parallel block calibration")
    return attempt("readout-calibration-fit", fit)


def _solve(matrix, vector, plan):
    import numpy as np
    condition = float(np.linalg.cond(matrix))
    if plan.max_condition is not None and condition > plan.max_condition:
        raise ValueError("assignment matrix is ill-conditioned; recalibrate or explicitly change the plan")
    result = (np.linalg.solve(matrix, vector) if plan.method == "solve"
              else np.linalg.pinv(matrix, rcond=plan.rcond) @ vector)
    return result, condition


def _simplex(vector):
    import numpy as np
    sorted_values = np.sort(vector)[::-1]
    threshold = (np.cumsum(sorted_values) - 1) / np.arange(1, len(vector) + 1)
    rho = np.flatnonzero(sorted_values > threshold)[-1]
    return np.maximum(vector - threshold[rho], 0)


def mitigate_marginal(counts, model: ReadoutModel, bits, plan: ReadoutPlan = ReadoutPlan()) -> Result[ReadoutEstimate, Failure]:
    """Correct a small marginal; leave negative quasi-probabilities by default.

    The union of touched blocks is limited to 8 bits. For large parity
    observables use mitigate_parity, which never builds the full distribution.
    Optional simplex projection is nonlinear and can introduce bias.
    """
    def mitigate():
        import numpy as np
        selected = indices(bits, "bits")
        if set(selected) - set(model.bits):
            raise ValueError("every selected bit needs a calibration block")
        blocks = tuple(block for block in model.blocks if set(block.bits) & set(selected))
        union = tuple(bit for block in blocks for bit in block.bits)
        if len(union) > 8:
            raise ValueError("marginal touches more than 8 bits; use mitigate_parity")
        entries = _counts(counts)
        shots = sum(value for _, value in entries)
        observed = np.zeros(1 << len(union))
        for outcome, count in entries:
            observed[_pack(outcome, union)] += count / shots
        assignment = np.ones((1, 1))
        for block in blocks:
            assignment = np.kron(np.asarray(block.assignment), assignment)
        corrected, condition = _solve(assignment, observed, plan)
        if plan.project_probabilities:
            corrected = _simplex(corrected)
        marginal = np.zeros(1 << len(selected))
        local_bits = tuple(union.index(bit) for bit in selected)
        for outcome, probability in enumerate(corrected):
            marginal[_pack(outcome, local_bits)] += probability
        return ReadoutEstimate(selected, tuple(map(float, marginal)), shots, condition, plan.project_probabilities)
    return attempt("readout-marginal", mitigate)


def mitigate_parity(counts, model: ReadoutModel, bits, plan: ReadoutPlan = ReadoutPlan()) -> Result[ReadoutExpectation, Failure]:
    """Apply blockwise adjoint correction to a measured Pauli parity.

    Solve A_b.T w_b = f_b, then average products w_b(observed_b).
    Memory is O(sum_b 4**block_width), not O(2**total_qubits). The shot standard
    error treats calibration matrices as fixed and excludes calibration error.
    Bounds are not clipped; simplex projection is deliberately unsupported here.
    """
    def mitigate():
        import numpy as np
        selected = indices(bits, "bits", allow_empty=True)
        if plan.project_probabilities:
            raise ValueError("simplex projection is not defined for adjoint parity correction")
        if set(selected) - set(model.bits):
            raise ValueError("every selected bit needs a calibration block")
        entries = _counts(counts)
        shots = sum(value for _, value in entries)
        functions, conditions = [], []
        for block in model.blocks:
            relevant = tuple(position for position, bit in enumerate(block.bits) if bit in selected)
            if not relevant:
                continue
            ideal = np.asarray([(-1.0) ** sum((outcome >> position) & 1 for position in relevant)
                                for outcome in range(1 << len(block.bits))])
            corrected, condition = _solve(np.asarray(block.assignment).T, ideal, plan)
            functions.append((block.bits, corrected))
            conditions.append(condition)
        mean, second = 0.0, 0.0
        for outcome, count in entries:
            value = 1.0
            for block_bits, weights in functions:
                value *= float(weights[_pack(outcome, block_bits)])
            mean += count * value / shots
            second += count * value * value / shots
        if not isfinite(mean) or not isfinite(second):
            raise ValueError("readout weights overflowed; calibration is too unstable for this observable")
        error = sqrt(max(0.0, second - mean * mean) / (shots - 1)) if shots > 1 else None
        return ReadoutExpectation(mean, error, shots, tuple(conditions))
    return attempt("readout-parity", mitigate)
