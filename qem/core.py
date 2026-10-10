"""Small, dependency-free functional values shared by native QEM algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Callable, Generic, TypeAlias, TypeVar

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E


Result: TypeAlias = Ok[T] | Err[E]


@dataclass(frozen=True, slots=True)
class Failure:
    technique: str
    message: str


def bind(result: Result[T, E], function: Callable[[T], Result[U, E]]) -> Result[U, E]:
    return function(result.value) if isinstance(result, Ok) else result


def map_result(result: Result[T, E], function: Callable[[T], U]) -> Result[U, E]:
    return Ok(function(result.value)) if isinstance(result, Ok) else result


def pipe(value: T, *functions: Callable[..., Any]) -> Any:
    for function in functions:
        value = function(value)
    return value


def pipe_result(result: Result[T, E], *functions: Callable[..., Any]) -> Any:
    for function in functions:
        result = bind(result, function)
    return result


def attempt(technique: str, action: Callable[[], T]) -> Result[T, Failure]:
    """Contain expected adapter/numerical errors; never hide interrupts or OOM."""
    try:
        return Ok(action())
    except MemoryError:
        raise
    except Exception as error:
        return Err(Failure(technique, f"{type(error).__name__}: {error}"))


def real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integer(value: Integral, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def indices(values, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    result = tuple(integer(value, name) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be distinct")
    return result


def names(values, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty gate names")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be distinct")
    return result


@dataclass(frozen=True, slots=True)
class CircuitMetrics:
    operations: int
    depth: int
    gate_counts: tuple[tuple[str, int], ...]
    nonvirtual_gates: int
    reset_count: int
    measurement_count: int
    qubits: int
