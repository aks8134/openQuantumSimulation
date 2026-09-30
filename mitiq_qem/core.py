"""Immutable values shared by the functional Mitiq facade."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Generic, TypeVar

import numpy as np


CircuitT = TypeVar("CircuitT")
Parameter = str | int | float | bool | None


def finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True, slots=True)
class Stage:
    """One transformation in the provenance of a circuit variant."""

    technique: str
    index: int
    parameters: tuple[tuple[str, Parameter], ...] = ()

    def __post_init__(self) -> None:
        if not self.technique.strip():
            raise ValueError("technique must be non-empty")
        if self.index < 0:
            raise ValueError("stage index must be non-negative")
        object.__setattr__(self, "parameters", tuple(self.parameters))


@dataclass(frozen=True, slots=True)
class CircuitVariant(Generic[CircuitT]):
    """A circuit plus immutable transformation provenance."""

    circuit: CircuitT
    source_index: int = 0
    stages: tuple[Stage, ...] = ()

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        object.__setattr__(self, "stages", tuple(self.stages))


@dataclass(frozen=True, slots=True)
class CircuitBatch(Generic[CircuitT]):
    """An immutable collection of circuits ready for external execution."""

    variants: tuple[CircuitVariant[CircuitT], ...]

    def __post_init__(self) -> None:
        variants = tuple(self.variants)
        if not variants:
            raise ValueError("a circuit batch cannot be empty")
        object.__setattr__(self, "variants", variants)

    @property
    def circuits(self) -> tuple[CircuitT, ...]:
        return tuple(map(lambda variant: variant.circuit, self.variants))

    def __len__(self) -> int:
        return len(self.variants)


@dataclass(frozen=True, slots=True)
class AveragedResult:
    """Mean and sampling uncertainty for an ensemble technique."""

    estimate: float
    standard_error: float
    values: tuple[float, ...]


def circuit_batch(circuit: CircuitT) -> CircuitBatch[CircuitT]:
    """Lift one circuit into a one-element immutable batch."""

    return CircuitBatch((CircuitVariant(circuit),))


def as_batch(
    value: CircuitT | CircuitBatch[CircuitT],
) -> CircuitBatch[CircuitT]:
    return value if isinstance(value, CircuitBatch) else circuit_batch(value)


def with_stage(
    variant: CircuitVariant[CircuitT],
    circuit: CircuitT,
    stage: Stage,
) -> CircuitVariant[CircuitT]:
    return CircuitVariant(
        circuit=circuit,
        source_index=variant.source_index,
        stages=(*variant.stages, stage),
    )


def average(values: tuple[Real, ...]) -> AveragedResult:
    """Average independent estimates and report their standard error."""

    converted = tuple(map(lambda value: finite_real(value, "value"), values))
    if not converted:
        raise ValueError("at least one value is required")
    estimate = float(np.mean(converted))
    standard_error = (
        0.0
        if len(converted) == 1
        else float(np.std(converted, ddof=1) / sqrt(len(converted)))
    )
    return AveragedResult(estimate, standard_error, converted)
