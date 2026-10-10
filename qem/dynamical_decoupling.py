"""Plans for scheduled DD; the Qiskit adapter handles backend timing."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .core import CircuitMetrics, indices, integer, real

DD_SEQUENCES = MappingProxyType({
    "xx": ("X", "X"),
    "yy": ("Y", "Y"),
    "xy4": ("X", "Y", "X", "Y"),
    "xy8": ("X", "Y", "X", "Y", "Y", "X", "Y", "X"),
})


@dataclass(frozen=True, slots=True)
class DDPlan:
    sequence: str = "xx"
    qubits: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None
    skip_reset_qubits: bool = True
    schedule: str = "alap"
    pulse_alignment: int = 1
    acquire_alignment: int = 1

    def __post_init__(self):
        if self.sequence not in DD_SEQUENCES:
            raise ValueError(f"sequence must be one of {tuple(DD_SEQUENCES)}")
        if self.qubits is not None:
            object.__setattr__(self, "qubits", indices(self.qubits, "qubits"))
        if self.spacing is not None:
            spacing = tuple(real(value, "spacing") for value in self.spacing)
            if (len(spacing) != len(DD_SEQUENCES[self.sequence]) + 1
                    or any(value < 0 for value in spacing)
                    or abs(sum(spacing) - 1) > 1e-10):
                raise ValueError("spacing must be nonnegative, sum to one, and match pulses + 1")
            object.__setattr__(self, "spacing", spacing)
        if not isinstance(self.skip_reset_qubits, bool):
            raise TypeError("skip_reset_qubits must be boolean")
        if self.schedule not in ("alap", "asap"):
            raise ValueError("schedule must be alap or asap")
        for field in ("pulse_alignment", "acquire_alignment"):
            object.__setattr__(self, field, integer(getattr(self, field), field, minimum=1))


@dataclass(frozen=True, slots=True)
class DecoupledCircuit:
    circuit: Any
    before: CircuitMetrics
    after: CircuitMetrics
    sequence: str
    physical_qubits: tuple[int, ...]
    inserted_gate_count: int
    pulse_alignment: int
    acquire_alignment: int
