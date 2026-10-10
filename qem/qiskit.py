"""Native Qiskit interpreters. No conversion, worker, submission, or lowering.

Qiskit/NumPy are imported only when a transform is called. Circuits are copied;
their gate representation, registers, metadata and physical layout are retained.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, pi
from random import Random, SystemRandom
from typing import Any

from .core import CircuitMetrics, Err, Failure, Result, attempt
from .dynamical_decoupling import DDPlan, DD_SEQUENCES, DecoupledCircuit
from .pauli_twirling import PauliImplementation, TwirlChoice, TwirledCircuit, TwirlingPlan
from .zne import FoldPlan, FoldSelection, select_folds


@dataclass(frozen=True, slots=True)
class FoldedCircuit:
    circuit: Any
    before: CircuitMetrics
    after: CircuitMetrics
    selection: FoldSelection
    selected_instructions: tuple[tuple[int, int], ...]
    inserted_operations: int
    represented_weight_scale: float


def _validate(circuit):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ControlFlowOp
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("a Qiskit QuantumCircuit is required")
    if any(isinstance(item.operation, ControlFlowOp) for item in circuit.data):
        raise ValueError("control-flow blocks are not supported; transform unitary regions explicitly")


def circuit_metrics(circuit, *, virtual_names=("rz",)) -> CircuitMetrics:
    from qiskit.circuit import Gate
    counts = Counter(item.operation.name for item in circuit.data)
    return CircuitMetrics(
        int(circuit.size()), int(circuit.depth()), tuple(sorted(counts.items())),
        sum(isinstance(item.operation, Gate) and item.operation.name not in virtual_names
            for item in circuit.data),
        counts["reset"], counts["measure"], circuit.num_qubits,
    )


def _empty_copy(circuit):
    result = circuit.copy_empty_like()
    result.metadata = deepcopy(circuit.metadata)
    # copy_empty_like shallow-copies the original scheduling cache. Qiskit has
    # no public setter for op_start_times; modified circuits must be rescheduled.
    result._op_start_times = None
    return result


def _supported(target, gate, qargs):
    return target is None or target.instruction_supported(
        operation_name=gate.name, qargs=tuple(qargs), parameters=gate.params,
    )


def _require_supported(circuit, target):
    if target is None:
        return
    missing = []
    for item in circuit.data:
        if item.operation.name in ("barrier",):
            continue
        qargs = tuple(circuit.find_bit(bit).index for bit in item.qubits)
        if not _supported(target, item.operation, qargs):
            missing.append((item.operation.name, qargs))
    if missing:
        raise ValueError(
            f"target does not support these operations/parameters: {tuple(dict.fromkeys(missing))}; "
            "supply an exact native inverse/Pauli factory or explicitly compile before mitigation. "
            "qem never silently lowers gates."
        )


def fold(circuit, plan: FoldPlan = FoldPlan(), *, target=None, inverse_factory=None) -> Result[FoldedCircuit, Failure]:
    """Append G (G† G)^k in place, never folding reset/measure/delay.

    An inverse_factory maps a Gate to a tuple of same-width native Gates whose
    chronological product is its exact inverse, including global phase. The
    caller owns custom-factory correctness. No transpiler is invoked. Original
    gates remain intact; new inverses can be composite when no target is given.
    """
    def transform():
        from qiskit.circuit import Gate
        _validate(circuit)
        weights = dict(plan.weights)
        eligible = tuple(
            index for index, item in enumerate(circuit.data)
            if isinstance(item.operation, Gate)
            and not item.clbits
            and getattr(item.operation, "condition", None) is None
            and item.operation.name not in plan.excluded_names
            and (plan.gate_names is None or item.operation.name in plan.gate_names)
        )
        selected = select_folds(
            tuple(weights.get(circuit.data[index].operation.name, 1.0) for index in eligible), plan,
        )
        if isinstance(selected, Err):
            raise ValueError(selected.error.message)
        selection = selected.value
        repetitions = dict(zip(eligible, selection.repetitions))
        inverses = {}
        for index, count in repetitions.items():
            if not count:
                continue
            gate = circuit.data[index].operation
            inverse = tuple(inverse_factory(gate)) if inverse_factory else (gate.inverse(),)
            if not inverse or any(not isinstance(value, Gate) or value.num_qubits != gate.num_qubits for value in inverse):
                raise ValueError("inverse_factory must return nonempty, same-width Qiskit Gates")
            inverses[index] = inverse
        inserted = sum(count * (len(inverses[index]) + 1) for index, count in repetitions.items() if count)
        if inserted > plan.max_inserted_operations:
            raise ValueError("native inverse expansion exceeds max_inserted_operations")
        result = _empty_copy(circuit)
        added_weight_ratio = 0.0
        for index, item in enumerate(circuit.data):
            result.append(item.operation, item.qubits, item.clbits)
            for _ in range(repetitions.get(index, 0)):
                for gate in (*inverses[index], item.operation):
                    result.append(gate, item.qubits)
                    if gate.name not in plan.excluded_names:
                        added_weight_ratio += weights.get(gate.name, 1.0) / selection.eligible_weight
        _require_supported(result, target)
        represented = 1 + added_weight_ratio
        if not isfinite(represented):
            raise ValueError("represented weight scale overflowed; rescale the weights")
        return FoldedCircuit(
            result, circuit_metrics(circuit, virtual_names=plan.excluded_names),
            circuit_metrics(result, virtual_names=plan.excluded_names), selection,
            tuple((index, count) for index, count in repetitions.items() if count),
            inserted, represented,
        )
    return attempt("zne-folding", transform)


def native_pauli(pauli: str, qubit: int, target=None) -> PauliImplementation:
    """Construct an exact Pauli using supported X/RX/SX and virtual RZ gates."""
    from qiskit.circuit.library import RXGate, RZGate, SXGate, XGate, YGate, ZGate
    if pauli == "I":
        return PauliImplementation(())
    if pauli not in ("X", "Y", "Z"):
        raise ValueError("Pauli must be I, X, Y, or Z")
    if target is None:
        return PauliImplementation(({"X": XGate, "Y": YGate, "Z": ZGate}[pauli](),))
    if pauli == "Z":
        if _supported(target, RZGate(pi), (qubit,)):
            return PauliImplementation((RZGate(pi),), pi / 2)
        if _supported(target, ZGate(), (qubit,)):
            return PauliImplementation((ZGate(),))
    else:
        if _supported(target, XGate(), (qubit,)):
            x = PauliImplementation((XGate(),))
        elif _supported(target, RXGate(pi), (qubit,)):
            x = PauliImplementation((RXGate(pi),), pi / 2)
        elif _supported(target, SXGate(), (qubit,)):
            x = PauliImplementation((SXGate(), SXGate()))
        else:
            raise ValueError(f"no native X realization on physical qubit {qubit}")
        if pauli == "X":
            return x
        if _supported(target, YGate(), (qubit,)):
            return PauliImplementation((YGate(),))
        if _supported(target, RZGate(pi), (qubit,)):
            # Chronological X, RZ(pi): RZ(pi) X = Y exactly.
            return PauliImplementation((*x.gates, RZGate(pi)), x.global_phase)
    raise ValueError(f"no native {pauli} realization on physical qubit {qubit}")


@lru_cache(maxsize=32)
def _clifford_twirl(name, before):
    import numpy as np
    from qiskit.circuit.library import CXGate, CZGate
    bits = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    labels = {value: key for key, value in bits.items()}
    (x0, z0), (x1, z1) = map(bits.__getitem__, before)
    if name == "cx":
        after = (labels[x0, z0 ^ z1], labels[x1 ^ x0, z1])
    else:
        after = (labels[x0, z0 ^ x1], labels[x1, z1 ^ x0])
    matrices = {
        "I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    }
    original = (CXGate() if name == "cx" else CZGate()).to_matrix()
    pre = np.kron(matrices[before[1]], matrices[before[0]])
    post = np.kron(matrices[after[1]], matrices[after[0]])
    coefficient = np.trace(post @ original @ pre @ original.conj().T) / 4
    phase = -float(np.angle(coefficient))
    return after, phase


def twirl(circuit, plan: TwirlingPlan = TwirlingPlan(), *, target=None, pauli_factory=native_pauli) -> Result[TwirledCircuit, Failure]:
    """Twirl CX/CZ and Pauli rotations, with exact angle/sign compensation.

    Pauli tuples follow gate qarg order, NOT displayed Pauli-string order.
    RXX/RYY/RZZ and RX/RY/RZ use P U(s*theta) P = U(theta); generic controlled
    rotations are not Clifford twirled. Unselected two-qubit gates are reported.
    """
    def transform():
        from qiskit.circuit import Gate
        from qiskit.circuit.library import (
            CXGate, CZGate, RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate,
        )
        _validate(circuit)
        seed = plan.seed if plan.seed is not None else SystemRandom().getrandbits(64)
        rng = Random(seed)
        result = _empty_copy(circuit)
        choices, skipped = [], Counter()
        gate_classes = {
            "cx": CXGate, "cz": CZGate, "rx": RXGate, "ry": RYGate,
            "rz": RZGate, "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate,
        }

        def append_paulis(labels, qargs):
            for label, bit in zip(labels, qargs):
                qubit = circuit.find_bit(bit).index
                implementation = pauli_factory(label, qubit, target)
                for gate in implementation.gates:
                    if not isinstance(gate, Gate) or gate.num_qubits != 1:
                        raise ValueError("Pauli factory must return single-qubit Gates")
                    result.append(gate, (bit,))
                result.global_phase += implementation.global_phase

        for index, item in enumerate(circuit.data):
            operation = item.operation
            if operation.name not in plan.gate_names:
                result.append(operation, item.qubits, item.clbits)
                if isinstance(operation, Gate) and operation.num_qubits == 2:
                    skipped[operation.name] += 1
                continue
            if (not isinstance(operation, Gate) or item.clbits
                    or getattr(operation, "condition", None) is not None):
                raise ValueError("only unconditional unitary gates can be twirled")
            if not isinstance(operation, gate_classes[operation.name]):
                raise ValueError("selected twirling gates must be standard Qiskit gates, not name-aliasing custom gates")
            expected = 2 if operation.name in ("cx", "cz", "rxx", "ryy", "rzz") else 1
            if operation.num_qubits != expected:
                raise ValueError("selected gate width does not match its twirling rule")
            before = tuple(rng.choice("IXYZ") for _ in item.qubits)
            angle_sign = 1
            if operation.name in ("cx", "cz"):
                after, phase = _clifford_twirl(operation.name, before)
                modified = operation
                result.global_phase += phase
            else:
                axis = operation.name[1].upper()
                anticommutations = sum(label not in ("I", axis) for label in before)
                angle_sign = -1 if anticommutations % 2 else 1
                modified = operation.copy()
                modified.params = [angle_sign * operation.params[0]]
                after = before
            append_paulis(before, item.qubits)
            result.append(modified, item.qubits)
            append_paulis(after, item.qubits)
            choices.append(TwirlChoice(index, operation.name, before, after, angle_sign))
        _require_supported(result, target)
        return TwirledCircuit(
            result, circuit_metrics(circuit), circuit_metrics(result), tuple(choices),
            seed, tuple(sorted(skipped.items())),
        )
    return attempt("pauli-twirling", transform)


def apply_dd(circuit, plan: DDPlan = DDPlan(), *, target=None, durations=None, pauli_factory=native_pauli) -> Result[DecoupledCircuit, Failure]:
    """Schedule and insert timed DD via Qiskit, without gate lowering.

    A physical Target or explicit InstructionDurations is mandatory. Target
    timing constraints take precedence. Alignment rescheduling is unnecessary
    when both alignments are 1 dt; scheduling and DD padding still run.
    Y frame components stay adjacent to
    their physical pulse (zero internal spacing). Unsupported timing/control
    flow returns Err rather than silently disabling mitigation.
    """
    def transform():
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            ALAPScheduleAnalysis, ASAPScheduleAnalysis, ConstrainedReschedule,
            PadDynamicalDecoupling,
        )
        _validate(circuit)
        if target is None and durations is None:
            raise ValueError("DD requires target durations or explicit InstructionDurations")
        _require_supported(circuit, target)
        qubits = plan.qubits if plan.qubits is not None else tuple(range(circuit.num_qubits))
        if any(qubit >= circuit.num_qubits for qubit in qubits):
            raise ValueError("DD physical qubit is outside the circuit")
        pulse_alignment = target.pulse_alignment if target is not None else plan.pulse_alignment
        acquire_alignment = target.acquire_alignment if target is not None else plan.acquire_alignment
        logical = DD_SEQUENCES[plan.sequence]
        spacing = plan.spacing or (1 / (2 * len(logical)), *([1 / len(logical)] * (len(logical) - 1)), 1 / (2 * len(logical)))
        gates, flattened_spacing = [], [spacing[0]]
        for position, label in enumerate(logical):
            pieces = pauli_factory(label, qubits[0], target).gates
            if not pieces:
                raise ValueError("DD pulse factory returned no gates")
            for piece_index, gate in enumerate(pieces):
                if gate.num_qubits != 1:
                    raise ValueError("DD pulses must be single-qubit gates")
                for qubit in qubits:
                    if not _supported(target, gate, (qubit,)):
                        raise ValueError("DD pulses are not supported on every selected qubit")
                gates.append(gate)
                flattened_spacing.append(spacing[position + 1] if piece_index == len(pieces) - 1 else 0.0)
        scheduler = ALAPScheduleAnalysis if plan.schedule == "alap" else ASAPScheduleAnalysis
        passes = [scheduler(durations=durations, target=target)]
        # At 1-dt resolution every scheduled start is already aligned. Avoid
        # the unnecessary rescheduler, which rejects some opaque Aer gates.
        # Stricter targets still require the hardware alignment correction.
        if pulse_alignment != 1 or acquire_alignment != 1:
            passes.append(ConstrainedReschedule(
                pulse_alignment=pulse_alignment, acquire_alignment=acquire_alignment,
            ))
        passes.append(
            PadDynamicalDecoupling(
                durations=durations, target=target, dd_sequence=gates,
                qubits=list(qubits), spacing=list(flattened_spacing),
                skip_reset_qubits=plan.skip_reset_qubits,
                pulse_alignment=pulse_alignment,
            )
        )
        manager = PassManager(passes)
        transformed = manager.run(circuit.copy())
        _require_supported(transformed, target)
        before, after = circuit_metrics(circuit), circuit_metrics(transformed)
        # Counts exclude padding delays, so a negative value flags an unexpected rewrite.
        original_gates = dict(before.gate_counts)
        new_gates = dict(after.gate_counts)
        inserted = sum(new_gates.get(name, 0) - original_gates.get(name, 0) for name in set(gate.name for gate in gates))
        return DecoupledCircuit(transformed, before, after, plan.sequence, qubits,
                                inserted, pulse_alignment, acquire_alignment)
    return attempt("dynamical-decoupling", transform)
