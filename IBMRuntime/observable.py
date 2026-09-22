"""Immutable Pauli-observable algebra for Estimator workloads."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from .functional import concat_map, map_tuple
from .validation import issue_if


PauliAxis: TypeAlias = Literal["X", "Y", "Z"]
PAULI_AXES: tuple[str, ...] = ("X", "Y", "Z")


@dataclass(frozen=True, slots=True)
class PauliFactor:
    axis: PauliAxis
    qubit: int


@dataclass(frozen=True, slots=True)
class PauliTerm:
    coefficient: float
    factors: tuple[PauliFactor, ...] = ()


@dataclass(frozen=True, slots=True)
class Observable:
    name: str
    terms: tuple[PauliTerm, ...]


def pauli(axis: PauliAxis, qubit: int) -> PauliFactor:
    return PauliFactor(axis=axis, qubit=qubit)


def pauli_term(coefficient: float, *factors: PauliFactor) -> PauliTerm:
    return PauliTerm(coefficient=float(coefficient), factors=tuple(factors))


def observable(name: str, *terms: PauliTerm) -> Observable:
    return Observable(name=name, terms=tuple(terms))


def _term_errors(
    term: PauliTerm,
    term_index: int,
    qubit_count: int,
) -> tuple[str, ...]:
    prefix = f"term {term_index}"
    qubits = map_tuple(lambda factor: factor.qubit, term.factors)
    return (
        *issue_if(
            not isfinite(term.coefficient),
            f"{prefix} has a non-finite coefficient",
        ),
        *issue_if(
            len(frozenset(qubits)) != len(qubits),
            f"{prefix} applies more than one factor to the same qubit",
        ),
        *concat_map(
            lambda indexed_factor: (
                *issue_if(
                    indexed_factor[1].axis not in PAULI_AXES,
                    f"{prefix} factor {indexed_factor[0]} has an invalid axis",
                ),
                *issue_if(
                    not 0 <= indexed_factor[1].qubit < qubit_count,
                    f"{prefix} factor {indexed_factor[0]} refers to invalid "
                    f"qubit {indexed_factor[1].qubit}",
                ),
            ),
            enumerate(term.factors),
        ),
    )


def validation_errors(
    value: Observable,
    qubit_count: int,
) -> tuple[str, ...]:
    return (
        *issue_if(not value.name, "an observable name cannot be empty"),
        *issue_if(not value.terms, "an observable must contain at least one term"),
        *concat_map(
            lambda indexed_term: _term_errors(
                indexed_term[1],
                indexed_term[0],
                qubit_count,
            ),
            enumerate(value.terms),
        ),
    )
