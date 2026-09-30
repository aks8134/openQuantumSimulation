"""Qiskit-specific validation and nonunitary-boundary preservation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import takewhile
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class CircuitFeatures:
    resets: int
    measurements: int
    mid_circuit_measurements: int
    control_flow_operations: int
    classically_conditioned_operations: int


def _is_measurement(item: Any) -> bool:
    return item.operation.name == "measure"


def _is_reset(item: Any) -> bool:
    return item.operation.name == "reset"


def _is_control_flow(item: Any) -> bool:
    try:
        from qiskit.circuit import ControlFlowOp

        return isinstance(item.operation, ControlFlowOp)
    except ImportError:
        return False


def _is_conditioned(item: Any) -> bool:
    return getattr(item.operation, "condition", None) is not None


def _terminal_measurement_start(data: tuple[Any, ...]) -> int:
    terminal_count = len(tuple(takewhile(_is_measurement, reversed(data))))
    return len(data) - terminal_count


def circuit_features(circuit: Any) -> CircuitFeatures:
    data = tuple(circuit.data)
    terminal_start = _terminal_measurement_start(data)
    indexed = tuple(enumerate(data))
    return CircuitFeatures(
        resets=sum(map(lambda item: int(_is_reset(item)), data)),
        measurements=sum(map(lambda item: int(_is_measurement(item)), data)),
        mid_circuit_measurements=sum(
            map(
                lambda indexed_item: int(
                    _is_measurement(indexed_item[1])
                    and indexed_item[0] < terminal_start
                ),
                indexed,
            )
        ),
        control_flow_operations=sum(
            map(lambda item: int(_is_control_flow(item)), data)
        ),
        classically_conditioned_operations=sum(
            map(lambda item: int(_is_conditioned(item)), data)
        ),
    )


def validate_mitiq_circuit(circuit: Any) -> CircuitFeatures:
    """Reject adaptive constructs outside Mitiq's supported program model."""

    features = circuit_features(circuit)
    problems = tuple(
        filter(
            lambda value: value is not None,
            (
                (
                    "Mitiq does not support control-flow operations"
                    if features.control_flow_operations
                    else None
                ),
                (
                    "Mitiq does not support classically conditioned operations"
                    if features.classically_conditioned_operations
                    else None
                ),
                (
                    "Mitiq does not support mid-circuit measurements"
                    if features.mid_circuit_measurements
                    else None
                ),
            ),
        )
    )
    if problems:
        raise ValueError("; ".join(problems))
    return features


def _unitary_instruction(item: Any) -> bool:
    try:
        from qiskit.circuit import Gate

        return isinstance(item.operation, Gate)
    except ImportError:
        return False


@dataclass(frozen=True, slots=True)
class _Segment:
    unitary: bool
    instructions: tuple[Any, ...]


def _add_to_segments(
    segments: tuple[_Segment, ...],
    item: Any,
) -> tuple[_Segment, ...]:
    unitary = _unitary_instruction(item)
    return (
        (*segments[:-1], _Segment(unitary, (*segments[-1].instructions, item)))
        if segments and segments[-1].unitary == unitary
        else (*segments, _Segment(unitary, (item,)))
    )


def _append_item(circuit: Any, source: Any, item: Any) -> Any:
    updated = circuit.copy()
    qubits = tuple(
        map(
            lambda bit: updated.qubits[source.find_bit(bit).index],
            item.qubits,
        )
    )
    clbits = tuple(
        map(
            lambda bit: updated.clbits[source.find_bit(bit).index],
            item.clbits,
        )
    )
    updated.append(item.operation, qubits, clbits)
    return updated


def _segment_circuit(source: Any, segment: _Segment) -> Any:
    empty = source.copy_empty_like()
    return reduce(
        lambda accumulated, item: _append_item(accumulated, source, item),
        segment.instructions,
        empty,
    )


def scale_unitary_regions(
    circuit: Any,
    scale_factor: float,
    scale: Callable[[Any, float], Any],
) -> Any:
    """Scale unitary regions while preserving reset/measurement boundaries."""

    validate_mitiq_circuit(circuit)
    segments = reduce(_add_to_segments, tuple(circuit.data), ())

    def transform(segment: _Segment) -> Any:
        region = _segment_circuit(circuit, segment)
        return scale(region, scale_factor) if segment.unitary else region

    transformed = tuple(map(transform, segments))
    result = reduce(
        lambda accumulated, region: accumulated.compose(region, inplace=False),
        transformed,
        circuit.copy_empty_like(),
    )
    result.name = circuit.name
    result.metadata = None if circuit.metadata is None else dict(circuit.metadata)
    return result
