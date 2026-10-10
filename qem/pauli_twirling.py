"""Immutable plans for native, gate-aware Pauli randomization."""

from dataclasses import dataclass
from typing import Any

from .core import CircuitMetrics, integer, names, real

SUPPORTED_GATES = ("cx", "cz", "rx", "ry", "rz", "rxx", "ryy", "rzz")


@dataclass(frozen=True, slots=True)
class TwirlingPlan:
    seed: int | None = 0
    gate_names: tuple[str, ...] = ("cx", "cz", "rzz")

    def __post_init__(self):
        if self.seed is not None:
            object.__setattr__(self, "seed", integer(self.seed, "seed"))
        normalized = names(self.gate_names, "gate_names")
        if set(normalized) - set(SUPPORTED_GATES):
            raise ValueError(f"supported twirling gates are {SUPPORTED_GATES}")
        object.__setattr__(self, "gate_names", normalized)


@dataclass(frozen=True, slots=True)
class PauliImplementation:
    """Native instructions on one qubit and their compensating global phase."""
    gates: tuple[Any, ...]
    global_phase: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "global_phase", real(self.global_phase, "global_phase"))


@dataclass(frozen=True, slots=True)
class TwirlChoice:
    instruction_index: int
    gate_name: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    angle_sign: int


@dataclass(frozen=True, slots=True)
class TwirledCircuit:
    circuit: Any
    before: CircuitMetrics
    after: CircuitMetrics
    choices: tuple[TwirlChoice, ...]
    seed: int
    untwirled_two_qubit_gates: tuple[tuple[str, int], ...]
