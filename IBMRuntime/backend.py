"""Immutable descriptions of execution targets."""

from dataclasses import dataclass
from typing import Literal, TypeAlias


AerMethod: TypeAlias = Literal[
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "stabilizer",
]

AER_METHODS: tuple[str, ...] = (
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "stabilizer",
)


@dataclass(frozen=True, slots=True)
class IBMHardware:
    backend_name: str


@dataclass(frozen=True, slots=True)
class Aer:
    method: AerMethod = "automatic"


Target: TypeAlias = IBMHardware | Aer


def validation_errors(target: Target) -> tuple[str, ...]:
    match target:
        case IBMHardware(backend_name):
            return () if backend_name.strip() else ("an IBM backend name cannot be empty",)
        case Aer(method):
            return () if method in AER_METHODS else (f"unsupported Aer method: {method}",)
        case _:
            return (f"unsupported target: {type(target).__name__}",)
