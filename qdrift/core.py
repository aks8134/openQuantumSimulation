"""Immutable, operator-agnostic qDRIFT sampling.

The core module deliberately knows nothing about Qiskit.  An operator may be
any Python value; an interpreter is responsible for turning the sampled
exponentials into executable gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Generic, TypeVar

import numpy as np


OperatorT = TypeVar("OperatorT")


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _sample_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("sample_count must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError("sample_count must be positive")
    return converted


def _targets(value: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    converted = tuple(value)
    if not converted:
        raise ValueError("targets must not be empty")
    if any(map(lambda target: isinstance(target, bool) or not isinstance(target, Integral), converted)):
        raise TypeError("every target must be an integer")
    normalized = tuple(map(int, converted))
    if any(map(lambda target: target < 0, normalized)):
        raise ValueError("targets must be non-negative")
    if len(frozenset(normalized)) != len(normalized):
        raise ValueError("targets must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class HamiltonianTerm(Generic[OperatorT]):
    """One term ``coefficient * operator`` in a Hamiltonian decomposition.

    ``operator_norm`` is the spectral norm of ``operator``.  It defaults to
    one, which is correct for Pauli strings and other Hermitian unitaries.
    ``targets`` gives the circuit qubits on which a local operator acts.
    """

    coefficient: float
    operator: OperatorT
    operator_norm: float = 1.0
    targets: tuple[int, ...] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        coefficient = _finite_real(self.coefficient, "coefficient")
        operator_norm = _finite_real(self.operator_norm, "operator_norm")
        if operator_norm <= 0.0:
            raise ValueError("operator_norm must be positive")
        if self.label is not None and not self.label.strip():
            raise ValueError("label must be non-empty when supplied")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "operator_norm", operator_norm)
        object.__setattr__(self, "targets", _targets(self.targets))

    @property
    def sampling_weight(self) -> float:
        """Return ``|h_i| ||H_i||`` for this term."""

        return abs(self.coefficient) * self.operator_norm


@dataclass(frozen=True, slots=True)
class Hamiltonian(Generic[OperatorT]):
    """An immutable decomposition of ``H = sum_i h_i H_i``."""

    terms: tuple[HamiltonianTerm[OperatorT], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not all(map(lambda item: isinstance(item, HamiltonianTerm), self.terms)):
            raise TypeError("terms must contain only HamiltonianTerm values")

    @property
    def lambda_norm(self) -> float:
        """Return ``lambda = sum_i |h_i| ||H_i||``."""

        return float(sum(map(lambda item: item.sampling_weight, self.terms)))

    @property
    def active_terms(self) -> tuple[tuple[int, HamiltonianTerm[OperatorT]], ...]:
        """Return indexed nonzero terms, retaining original term indices."""

        return tuple(
            filter(
                lambda indexed: indexed[1].sampling_weight > 0.0,
                enumerate(self.terms),
            )
        )


@dataclass(frozen=True, slots=True)
class QDriftStep(Generic[OperatorT]):
    """One randomly selected exponential in a qDRIFT realization."""

    position: int
    term_index: int
    term: HamiltonianTerm[OperatorT]
    probability: float
    operator_time: float


@dataclass(frozen=True, slots=True)
class QDriftProgram(Generic[OperatorT]):
    """A completely sampled, immutable qDRIFT realization."""

    hamiltonian: Hamiltonian[OperatorT]
    evolution_time: float
    sample_count: int
    seed: int | None
    lambda_norm: float
    probabilities: tuple[float, ...]
    steps: tuple[QDriftStep[OperatorT], ...]

    @property
    def sampled_term_indices(self) -> tuple[int, ...]:
        return tuple(map(lambda step: step.term_index, self.steps))


def term(
    coefficient: Real,
    operator: OperatorT,
    *,
    operator_norm: Real = 1.0,
    targets: tuple[int, ...] | None = None,
    label: str | None = None,
) -> HamiltonianTerm[OperatorT]:
    """Construct one immutable Hamiltonian term."""

    return HamiltonianTerm(
        coefficient=coefficient,
        operator=operator,
        operator_norm=operator_norm,
        targets=targets,
        label=label,
    )


def hamiltonian(
    terms: tuple[HamiltonianTerm[OperatorT], ...] = (),
) -> Hamiltonian[OperatorT]:
    """Construct an immutable Hamiltonian from a tuple of terms."""

    return Hamiltonian(tuple(terms))


def empty_hamiltonian() -> Hamiltonian[OperatorT]:
    """Return an empty Hamiltonian suitable for functional construction."""

    return Hamiltonian()


def add_term(
    model: Hamiltonian[OperatorT],
    coefficient: Real,
    operator: OperatorT,
    *,
    operator_norm: Real = 1.0,
    targets: tuple[int, ...] | None = None,
    label: str | None = None,
) -> Hamiltonian[OperatorT]:
    """Return a new Hamiltonian with ``coefficient * operator`` appended."""

    return Hamiltonian(
        (
            *model.terms,
            term(
                coefficient,
                operator,
                operator_norm=operator_norm,
                targets=targets,
                label=label,
            ),
        )
    )


def sampling_probabilities(model: Hamiltonian[OperatorT]) -> tuple[float, ...]:
    """Return one probability per input term, including zeros."""

    lambda_norm = model.lambda_norm
    if not model.active_terms or lambda_norm <= 0.0:
        raise ValueError("qDRIFT requires at least one nonzero Hamiltonian term")
    return tuple(map(lambda item: item.sampling_weight / lambda_norm, model.terms))


def _sampled_step(
    position_and_active_index: tuple[int, int],
    active: tuple[tuple[int, HamiltonianTerm[OperatorT]], ...],
    probabilities: tuple[float, ...],
    base_time: float,
) -> QDriftStep[OperatorT]:
    position, active_index = position_and_active_index
    term_index, selected = active[active_index]
    sign = 1.0 if selected.coefficient > 0.0 else -1.0
    return QDriftStep(
        position=position,
        term_index=term_index,
        term=selected,
        probability=probabilities[term_index],
        operator_time=sign * base_time / selected.operator_norm,
    )


def qdrift(
    model: Hamiltonian[OperatorT],
    *,
    evolution_time: Real,
    sample_count: int,
    seed: int | None = None,
) -> QDriftProgram[OperatorT]:
    """Sample one qDRIFT realization.

    ``sample_count`` is the number of random exponentials in this realization,
    not the number of separately averaged circuits.  Passing a seed makes the
    returned immutable program reproducible.
    """

    time = _finite_real(evolution_time, "evolution_time")
    if time < 0.0:
        raise ValueError("evolution_time must be non-negative")
    count = _sample_count(sample_count)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")

    active = model.active_terms
    probabilities = sampling_probabilities(model)
    active_probabilities = tuple(map(lambda indexed: probabilities[indexed[0]], active))
    lambda_norm = model.lambda_norm
    random = np.random.default_rng(None if seed is None else int(seed))
    active_indices = random.choice(
        len(active),
        size=count,
        replace=True,
        p=np.asarray(active_probabilities, dtype=float),
    )
    base_time = lambda_norm * time / count
    steps = tuple(
        map(
            lambda item: _sampled_step(item, active, probabilities, base_time),
            enumerate(map(int, active_indices)),
        )
    )
    return QDriftProgram(
        hamiltonian=model,
        evolution_time=time,
        sample_count=count,
        seed=None if seed is None else int(seed),
        lambda_norm=lambda_norm,
        probabilities=probabilities,
        steps=steps,
    )
