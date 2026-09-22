"""A small immutable circuit language.

These types are Python equivalents of OCaml variants and records. Qiskit is
deliberately not imported into this functional core.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from typing import TypeAlias, TypeVar

from .functional import concat_map, filter_map, fold_left, map_tuple
from .validation import issue_if


@dataclass(frozen=True, slots=True)
class H:
    qubit: int


@dataclass(frozen=True, slots=True)
class X:
    qubit: int


@dataclass(frozen=True, slots=True)
class CX:
    control: int
    target: int


@dataclass(frozen=True, slots=True)
class Measure:
    qubit: int
    bit: int


@dataclass(frozen=True, slots=True)
class RZ:
    angle: float
    qubit: int


@dataclass(frozen=True, slots=True)
class MultiControlledRZ:
    angle: float
    controls: tuple[int, ...]
    target: int
    control_state: int = 0
    label: str | None = None


@dataclass(frozen=True, slots=True)
class XXPlusYY:
    angle: float
    phase: float
    first: int
    second: int
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ControlledXXPlusYY:
    angle: float
    phase: float
    control: int
    first: int
    second: int
    control_state: int = 1
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ClassicallyControlledXXPlusYY:
    angle: float
    phase: float
    bit: int
    condition_value: int
    first: int
    second: int
    label: str | None = None


@dataclass(frozen=True, slots=True)
class MultiControlledXXPlusYY:
    angle: float
    phase: float
    controls: tuple[int, ...]
    first: int
    second: int
    control_state: int = 0
    label: str | None = None


@dataclass(frozen=True, slots=True)
class Reset:
    qubit: int


@dataclass(frozen=True, slots=True)
class SaveDensityMatrix:
    qubits: tuple[int, ...]
    label: str


Operation: TypeAlias = (
    H
    | X
    | CX
    | Measure
    | RZ
    | MultiControlledRZ
    | XXPlusYY
    | ControlledXXPlusYY
    | ClassicallyControlledXXPlusYY
    | MultiControlledXXPlusYY
    | Reset
    | SaveDensityMatrix
)


@dataclass(frozen=True, slots=True)
class Circuit:
    qubit_count: int
    bit_count: int
    operations: tuple[Operation, ...] = ()
    name: str = "circuit"


CircuitTransform: TypeAlias = Callable[[Circuit], Circuit]
T = TypeVar("T")


def empty(qubits: int, bits: int, *, name: str = "circuit") -> Circuit:
    return Circuit(qubit_count=qubits, bit_count=bits, name=name)


def append(circuit: Circuit, operation: Operation) -> Circuit:
    """Return a new circuit containing one additional operation."""

    return replace(circuit, operations=(*circuit.operations, operation))


def h(qubit: int) -> CircuitTransform:
    return lambda circuit: append(circuit, H(qubit))


def x(qubit: int) -> CircuitTransform:
    return lambda circuit: append(circuit, X(qubit))


def cx(control: int, target: int) -> CircuitTransform:
    return lambda circuit: append(circuit, CX(control, target))


def measure(qubit: int, bit: int) -> CircuitTransform:
    return lambda circuit: append(circuit, Measure(qubit, bit))


def rz(angle: float, qubit: int) -> CircuitTransform:
    return lambda circuit: append(circuit, RZ(float(angle), qubit))


def multi_controlled_rz(
    angle: float,
    controls: tuple[int, ...],
    target: int,
    *,
    control_state: int = 0,
    label: str | None = None,
) -> CircuitTransform:
    return lambda circuit: append(
        circuit,
        MultiControlledRZ(
            float(angle),
            tuple(controls),
            target,
            control_state,
            label,
        ),
    )


def xx_plus_yy(
    angle: float,
    phase: float,
    first: int,
    second: int,
    *,
    label: str | None = None,
) -> CircuitTransform:
    return lambda circuit: append(
        circuit,
        XXPlusYY(float(angle), float(phase), first, second, label),
    )


def controlled_xx_plus_yy(
    angle: float,
    phase: float,
    control: int,
    first: int,
    second: int,
    *,
    control_state: int = 1,
    label: str | None = None,
) -> CircuitTransform:
    return lambda circuit: append(
        circuit,
        ControlledXXPlusYY(
            float(angle),
            float(phase),
            control,
            first,
            second,
            control_state,
            label,
        ),
    )


def classically_controlled_xx_plus_yy(
    angle: float,
    phase: float,
    bit: int,
    condition_value: int,
    first: int,
    second: int,
    *,
    label: str | None = None,
) -> CircuitTransform:
    return lambda circuit: append(
        circuit,
        ClassicallyControlledXXPlusYY(
            float(angle),
            float(phase),
            bit,
            condition_value,
            first,
            second,
            label,
        ),
    )


def multi_controlled_xx_plus_yy(
    angle: float,
    phase: float,
    controls: tuple[int, ...],
    first: int,
    second: int,
    *,
    control_state: int = 0,
    label: str | None = None,
) -> CircuitTransform:
    return lambda circuit: append(
        circuit,
        MultiControlledXXPlusYY(
            float(angle),
            float(phase),
            tuple(controls),
            first,
            second,
            control_state,
            label,
        ),
    )


def reset(qubit: int) -> CircuitTransform:
    return lambda circuit: append(circuit, Reset(qubit))


def save_density_matrix(
    qubits: tuple[int, ...],
    *,
    label: str,
) -> CircuitTransform:
    return lambda circuit: append(circuit, SaveDensityMatrix(tuple(qubits), label))


def pipe(value: T, *functions: Callable[[T], T]) -> T:
    """Apply functions left-to-right, like OCaml's ``|>`` operator."""

    return fold_left(lambda result, function: function(result), value, functions)


def _operation_errors(
    circuit: Circuit,
    index: int,
    operation: Operation,
) -> tuple[str, ...]:
    prefix = f"operation {index}"

    match operation:
        case H(qubit) | X(qubit) | Reset(qubit):
            return issue_if(
                not 0 <= qubit < circuit.qubit_count,
                f"{prefix} refers to invalid qubit {qubit}",
            )
        case CX(control, target):
            return (
                *issue_if(
                    not 0 <= control < circuit.qubit_count,
                    f"{prefix} refers to invalid control qubit {control}",
                ),
                *issue_if(
                    not 0 <= target < circuit.qubit_count,
                    f"{prefix} refers to invalid target qubit {target}",
                ),
                *issue_if(
                    control == target,
                    f"{prefix} uses the same control and target",
                ),
            )
        case Measure(qubit, bit):
            return (
                *issue_if(
                    not 0 <= qubit < circuit.qubit_count,
                    f"{prefix} refers to invalid qubit {qubit}",
                ),
                *issue_if(
                    not 0 <= bit < circuit.bit_count,
                    f"{prefix} refers to invalid classical bit {bit}",
                ),
            )
        case RZ(angle, qubit):
            return (
                *issue_if(
                    not isfinite(angle),
                    f"{prefix} has a non-finite rotation angle",
                ),
                *issue_if(
                    not 0 <= qubit < circuit.qubit_count,
                    f"{prefix} refers to invalid qubit {qubit}",
                ),
            )
        case MultiControlledRZ(
            angle,
            controls,
            target,
            control_state,
            _,
        ):
            invalid_qubits = filter_map(
                lambda role_and_qubit: (
                    f"{prefix} refers to invalid {role_and_qubit[0]} "
                    f"qubit {role_and_qubit[1]}"
                    if not 0 <= role_and_qubit[1] < circuit.qubit_count
                    else None
                ),
                (
                    *map_tuple(
                        lambda control: ("control", control),
                        controls,
                    ),
                    ("target", target),
                ),
            )
            return (
                *issue_if(
                    not isfinite(angle),
                    f"{prefix} has a non-finite rotation angle",
                ),
                *issue_if(
                    not controls,
                    f"{prefix} requires at least one control",
                ),
                *invalid_qubits,
                *issue_if(
                    len(frozenset((*controls, target)))
                    != len(controls) + 1,
                    f"{prefix} requires distinct controls and target",
                ),
                *issue_if(
                    not isinstance(control_state, int)
                    or not 0 <= control_state < 2 ** len(controls),
                    f"{prefix} control state is outside its control register",
                ),
            )
        case XXPlusYY(angle, phase, first, second, _):
            return (
                *issue_if(
                    not isfinite(angle) or not isfinite(phase),
                    f"{prefix} has a non-finite gate parameter",
                ),
                *issue_if(
                    not 0 <= first < circuit.qubit_count,
                    f"{prefix} refers to invalid first qubit {first}",
                ),
                *issue_if(
                    not 0 <= second < circuit.qubit_count,
                    f"{prefix} refers to invalid second qubit {second}",
                ),
                *issue_if(
                    first == second,
                    f"{prefix} uses the same qubit twice",
                ),
            )
        case ControlledXXPlusYY(
            angle,
            phase,
            control,
            first,
            second,
            control_state,
            _,
        ):
            invalid_qubits = filter_map(
                lambda role_and_qubit: (
                    f"{prefix} refers to invalid {role_and_qubit[0]} "
                    f"qubit {role_and_qubit[1]}"
                    if not 0 <= role_and_qubit[1] < circuit.qubit_count
                    else None
                ),
                (
                    ("control", control),
                    ("first", first),
                    ("second", second),
                ),
            )
            return (
                *issue_if(
                    not isfinite(angle) or not isfinite(phase),
                    f"{prefix} has a non-finite gate parameter",
                ),
                *invalid_qubits,
                *issue_if(
                    len(frozenset((control, first, second))) != 3,
                    f"{prefix} requires three distinct qubits",
                ),
                *issue_if(
                    control_state not in (0, 1),
                    f"{prefix} control state must be 0 or 1",
                ),
            )
        case ClassicallyControlledXXPlusYY(
            angle,
            phase,
            bit,
            condition_value,
            first,
            second,
            _,
        ):
            return (
                *issue_if(
                    not isfinite(angle) or not isfinite(phase),
                    f"{prefix} has a non-finite gate parameter",
                ),
                *issue_if(
                    not 0 <= bit < circuit.bit_count,
                    f"{prefix} refers to invalid classical bit {bit}",
                ),
                *issue_if(
                    condition_value not in (0, 1),
                    f"{prefix} condition value must be 0 or 1",
                ),
                *issue_if(
                    not 0 <= first < circuit.qubit_count,
                    f"{prefix} refers to invalid first qubit {first}",
                ),
                *issue_if(
                    not 0 <= second < circuit.qubit_count,
                    f"{prefix} refers to invalid second qubit {second}",
                ),
                *issue_if(
                    first == second,
                    f"{prefix} uses the same qubit twice",
                ),
            )
        case MultiControlledXXPlusYY(
            angle,
            phase,
            controls,
            first,
            second,
            control_state,
            _,
        ):
            invalid_qubits = filter_map(
                lambda role_and_qubit: (
                    f"{prefix} refers to invalid {role_and_qubit[0]} "
                    f"qubit {role_and_qubit[1]}"
                    if not 0 <= role_and_qubit[1] < circuit.qubit_count
                    else None
                ),
                (
                    *map_tuple(
                        lambda control: ("control", control),
                        controls,
                    ),
                    ("first", first),
                    ("second", second),
                ),
            )
            return (
                *issue_if(
                    not isfinite(angle) or not isfinite(phase),
                    f"{prefix} has a non-finite gate parameter",
                ),
                *issue_if(
                    not controls,
                    f"{prefix} requires at least one control",
                ),
                *invalid_qubits,
                *issue_if(
                    len(frozenset((*controls, first, second)))
                    != len(controls) + 2,
                    f"{prefix} requires distinct controls and targets",
                ),
                *issue_if(
                    not isinstance(control_state, int)
                    or not 0 <= control_state < 2 ** len(controls),
                    f"{prefix} control state is outside its control register",
                ),
            )
        case SaveDensityMatrix(qubits, label):
            invalid_qubits = filter_map(
                lambda qubit: (
                    f"{prefix} refers to invalid qubit {qubit}"
                    if not 0 <= qubit < circuit.qubit_count
                    else None
                ),
                qubits,
            )
            return (
                *issue_if(
                    not qubits,
                    f"{prefix} must save at least one qubit",
                ),
                *issue_if(
                    len(frozenset(qubits)) != len(qubits),
                    f"{prefix} contains duplicate qubits",
                ),
                *invalid_qubits,
                *issue_if(
                    not label,
                    f"{prefix} has an empty snapshot label",
                ),
            )
        case _:
            return (f"{prefix} has an unsupported operation",)


def validation_errors(circuit: Circuit) -> tuple[str, ...]:
    """Return every structural error without raising or mutating the circuit."""

    return (
        *issue_if(
            circuit.qubit_count <= 0,
            "a circuit must contain at least one qubit",
        ),
        *issue_if(
            circuit.bit_count < 0,
            "a circuit cannot contain a negative number of bits",
        ),
        *issue_if(not circuit.name, "a circuit name cannot be empty"),
        *concat_map(
            lambda indexed_operation: _operation_errors(
                circuit,
                indexed_operation[0],
                indexed_operation[1],
            ),
            enumerate(circuit.operations),
        ),
    )
