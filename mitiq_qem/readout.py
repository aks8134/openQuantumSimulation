"""Immutable readout-confusion inversion and Mitiq interoperation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import isfinite, log2
from numbers import Real
from typing import Mapping

import numpy as np

from .core import finite_real, positive_int


@dataclass(frozen=True, slots=True)
class ReadoutPlan:
    num_qubits: int
    inverse_confusion_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ReadoutProbability:
    bitstring: str
    raw: float
    mitigated: float


@dataclass(frozen=True, slots=True)
class ReadoutResult:
    num_qubits: int
    shots: float
    probabilities: tuple[ReadoutProbability, ...]

    @property
    def raw_probabilities(self) -> tuple[float, ...]:
        return tuple(map(lambda item: item.raw, self.probabilities))

    @property
    def mitigated_probabilities(self) -> tuple[float, ...]:
        return tuple(map(lambda item: item.mitigated, self.probabilities))


def readout_plan(inverse_confusion_matrix) -> ReadoutPlan:
    matrix = np.asarray(inverse_confusion_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse confusion matrix must be square")
    width = int(round(log2(matrix.shape[0]))) if matrix.shape[0] else -1
    if width <= 0 or 2**width != matrix.shape[0]:
        raise ValueError("matrix dimension must be a positive power of two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("inverse confusion matrix must be finite")
    return ReadoutPlan(
        width,
        tuple(map(lambda row: tuple(map(float, row)), matrix)),
    )


def _single_qubit_confusion(p0: float, p1: float) -> np.ndarray:
    return np.asarray(((1.0 - p0, p1), (p0, 1.0 - p1)), dtype=float)


def uncorrelated_readout_plan(
    num_qubits: int,
    *,
    p0: Real,
    p1: Real,
) -> ReadoutPlan:
    """Build the inverse matrix for identical independent readout errors.

    ``p0`` is P(measure 1 | prepared 0), and ``p1`` is
    P(measure 0 | prepared 1).
    """

    width = positive_int(num_qubits, "num_qubits")
    zero_error = finite_real(p0, "p0")
    one_error = finite_real(p1, "p1")
    if not 0.0 <= zero_error < 1.0 or not 0.0 <= one_error < 1.0:
        raise ValueError("p0 and p1 must lie in [0, 1)")
    if zero_error + one_error >= 1.0:
        raise ValueError("readout confusion matrix is singular or orientation-reversing")
    inverse = np.linalg.inv(_single_qubit_confusion(zero_error, one_error))
    complete = reduce(
        lambda accumulated, _: np.kron(accumulated, inverse),
        range(width),
        np.asarray(((1.0,),)),
    )
    return readout_plan(complete)


def _bitstrings(width: int) -> tuple[str, ...]:
    return tuple(map(lambda value: format(value, f"0{width}b"), range(2**width)))


def mitigate_counts(
    plan: ReadoutPlan,
    counts: Mapping[str, Real],
) -> ReadoutResult:
    """Apply confusion inversion and retain signed quasi-probabilities."""

    bitstrings = _bitstrings(plan.num_qubits)
    invalid = tuple(
        filter(
            lambda key: len(key) != plan.num_qubits or set(key) - {"0", "1"},
            counts,
        )
    )
    if invalid:
        raise ValueError(f"invalid count bitstrings: {invalid}")
    values = tuple(
        map(lambda key: finite_real(counts.get(key, 0.0), f"count {key}"), bitstrings)
    )
    if any(map(lambda value: value < 0.0, values)):
        raise ValueError("counts must be non-negative")
    shots = float(sum(values))
    if not isfinite(shots) or shots <= 0.0:
        raise ValueError("counts must contain at least one shot")
    raw = np.asarray(values, dtype=float) / shots
    inverse = np.asarray(plan.inverse_confusion_matrix, dtype=float)
    mitigated = inverse @ raw
    probabilities = tuple(
        map(
            lambda index: ReadoutProbability(
                bitstrings[index],
                float(raw[index]),
                float(mitigated[index]),
            ),
            range(len(bitstrings)),
        )
    )
    return ReadoutResult(plan.num_qubits, shots, probabilities)


def diagonal_expectation(
    result: ReadoutResult,
    eigenvalues: tuple[Real, ...],
) -> float:
    expected = 2**result.num_qubits
    if len(eigenvalues) != expected:
        raise ValueError(f"expected {expected} diagonal eigenvalues")
    converted = np.asarray(
        tuple(map(lambda value: finite_real(value, "eigenvalue"), eigenvalues)),
        dtype=float,
    )
    return float(converted @ np.asarray(result.mitigated_probabilities))


def mitigate_measurements(plan: ReadoutPlan, measurement_result):
    """Apply Mitiq REM to an upstream ``MeasurementResult`` value."""

    from .mitiq import mitigate_measurement_result

    return mitigate_measurement_result(measurement_result, plan)
