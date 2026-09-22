"""Immutable result values used instead of expected exceptions."""

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar


Value = TypeVar("Value")
Error = TypeVar("Error")


@dataclass(frozen=True, slots=True)
class Ok(Generic[Value]):
    value: Value


@dataclass(frozen=True, slots=True)
class Err(Generic[Error]):
    error: Error


Result: TypeAlias = Ok[Value] | Err[Error]
