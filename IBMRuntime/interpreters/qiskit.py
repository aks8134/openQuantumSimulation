"""The sole boundary containing mutable Qiskit and IBM Runtime objects."""

from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal, TypeVar

from ..backend import (
    Aer,
    FakeIBMBackend,
    IBMHardware,
    Target,
    validation_errors as target_errors,
)
from ..circuit import (
    CX,
    H,
    RZ,
    X,
    Circuit,
    ClassicallyControlledXXPlusYY,
    ControlledXXPlusYY,
    MultiControlledRZ,
    MultiControlledXXPlusYY,
    Measure,
    Operation,
    Reset,
    SaveDensityMatrix,
    XXPlusYY,
    validation_errors as circuit_errors,
)
from ..compile import (
    CompilationMetrics,
    CompilerConfig,
    validation_errors as compiler_errors,
)
from ..execute import (
    DensityMatrix,
    DensityMatrixResult,
    Estimate,
    EstimateResult,
    ExecutionPlan,
    ExecutionFailure,
    ExpectationValue,
    RuntimeFailure,
    Sample,
    SampleResult,
    SimulateDensityMatrices,
    ValidationFailure,
    Workload,
    normalize_counts,
    validate,
)
from ..functional import concat_map, map_tuple
from ..observable import Observable
from ..result import Err, Ok, Result
from ..runtime import RuntimeEnvironment, RuntimeModule


A = TypeVar("A")
Provider = Literal["aer", "fake", "ibm"]


@dataclass(frozen=True, slots=True)
class _ResolvedBackend:
    provider: Provider
    name: str
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _Executable:
    logical_qubit_count: int
    original_gate_count: int
    original_depth: int
    compiled_gate_count: int
    compiled_depth: int
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _SubmittedJob:
    provider: Provider
    backend_name: str
    shots: int
    job_id: str | None
    workload: Workload
    original_gate_count: int
    original_depth: int
    compiled_gate_count: int
    compiled_depth: int
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _SubmittedSampleBatch:
    provider: Provider
    backend_name: str
    shots: int
    job_id: str | None
    executables: tuple[_Executable, ...]
    native: Any = field(repr=False, compare=False)


def _protected(
    stage: Literal["resolve_backend", "compile", "submit", "collect"],
    action: Callable[[], A],
) -> Result[A, RuntimeFailure]:
    try:
        return Ok(action())
    except Exception as exception:
        return Err(
            ExecutionFailure(
                stage=stage,
                message=str(exception) or type(exception).__name__,
                exception_type=type(exception).__name__,
            )
        )


def _resolve_backend(
    target: Target,
    environment: RuntimeEnvironment,
) -> Result[_ResolvedBackend, RuntimeFailure]:
    match target:
        case Aer(method):
            def resolve_aer() -> _ResolvedBackend:
                from qiskit_aer import AerSimulator

                backend = AerSimulator(method=method)
                return _ResolvedBackend(provider="aer", name=backend.name, native=backend)

            return _protected("resolve_backend", resolve_aer)

        case IBMHardware(backend_name):
            if environment.ibm_account is None:
                return Err(
                    ExecutionFailure(
                        stage="resolve_backend",
                        message="IBM execution requires an IBMAccount in RuntimeEnvironment",
                    )
                )

            def resolve_ibm() -> _ResolvedBackend:
                from qiskit_ibm_runtime import QiskitRuntimeService

                account = environment.ibm_account
                service = QiskitRuntimeService(
                    channel=account.channel,
                    token=account.api_key,
                    instance=account.instance,
                )
                backend = service.backend(backend_name)
                return _ResolvedBackend(provider="ibm", name=backend.name, native=backend)

            return _protected("resolve_backend", resolve_ibm)

        case FakeIBMBackend("fake_fez"):
            def resolve_fake_fez() -> _ResolvedBackend:
                from qiskit_ibm_runtime.fake_provider import FakeFez

                backend = FakeFez()
                return _ResolvedBackend(
                    provider="fake",
                    name=backend.name,
                    native=backend,
                )

            return _protected("resolve_backend", resolve_fake_fez)


def _to_qiskit_instruction(
    operation: Operation,
    native_qubits: tuple[Any, ...],
    native_clbits: tuple[Any, ...],
):
    from qiskit.circuit import (
        CircuitInstruction,
        Measure as QiskitMeasure,
        Reset as QiskitReset,
    )
    from qiskit.circuit.library import CXGate, HGate, RZGate, XGate, XXPlusYYGate

    match operation:
        case H(qubit):
            return CircuitInstruction(HGate(), (native_qubits[qubit],), ())
        case X(qubit):
            return CircuitInstruction(XGate(), (native_qubits[qubit],), ())
        case CX(control, target):
            return CircuitInstruction(
                CXGate(),
                (native_qubits[control], native_qubits[target]),
                (),
            )
        case Measure(qubit, bit):
            return CircuitInstruction(
                QiskitMeasure(),
                (native_qubits[qubit],),
                (native_clbits[bit],),
            )
        case RZ(angle, qubit):
            return CircuitInstruction(RZGate(angle), (native_qubits[qubit],), ())
        case MultiControlledRZ(
            angle,
            controls,
            target,
            control_state,
            label,
        ):
            gate = RZGate(angle).control(
                num_ctrl_qubits=len(controls),
                ctrl_state=control_state,
                label=label,
                annotated=False,
            )
            return CircuitInstruction(
                gate,
                (
                    *map_tuple(
                        lambda control: native_qubits[control],
                        controls,
                    ),
                    native_qubits[target],
                ),
                (),
            )
        case XXPlusYY(angle, phase, first, second, label):
            return CircuitInstruction(
                XXPlusYYGate(theta=angle, beta=phase, label=label),
                (native_qubits[first], native_qubits[second]),
                (),
            )
        case ControlledXXPlusYY(
            angle,
            phase,
            control,
            first,
            second,
            control_state,
            label,
        ):
            gate = XXPlusYYGate(theta=angle, beta=phase).control(
                num_ctrl_qubits=1,
                ctrl_state=control_state,
                label=label,
                annotated=False,
            )
            return CircuitInstruction(
                gate,
                (
                    native_qubits[control],
                    native_qubits[first],
                    native_qubits[second],
                ),
                (),
            )
        case ClassicallyControlledXXPlusYY(
            angle,
            phase,
            bit,
            condition_value,
            first,
            second,
            label,
        ):
            from qiskit import QuantumCircuit
            from qiskit.circuit import IfElseOp

            true_body = QuantumCircuit(2, name=label or "conditional_xy")
            true_body.append(
                XXPlusYYGate(theta=angle, beta=phase, label=label),
                (0, 1),
            )
            operation = IfElseOp(
                (native_clbits[bit], condition_value),
                true_body,
                label=(
                    None
                    if label is None
                    else f"{label} if c[{bit}]={condition_value}"
                ),
            )
            return CircuitInstruction(
                operation,
                (native_qubits[first], native_qubits[second]),
                (),
            )
        case MultiControlledXXPlusYY(
            angle,
            phase,
            controls,
            first,
            second,
            control_state,
            label,
        ):
            gate = XXPlusYYGate(theta=angle, beta=phase).control(
                num_ctrl_qubits=len(controls),
                ctrl_state=control_state,
                label=label,
                annotated=False,
            )
            return CircuitInstruction(
                gate,
                (
                    *map_tuple(
                        lambda control: native_qubits[control],
                        controls,
                    ),
                    native_qubits[first],
                    native_qubits[second],
                ),
                (),
            )
        case Reset(qubit):
            return CircuitInstruction(QiskitReset(), (native_qubits[qubit],), ())
        case SaveDensityMatrix(qubit_indices, label):
            from qiskit_aer.library import SaveDensityMatrix as QiskitSaveDensityMatrix

            return CircuitInstruction(
                QiskitSaveDensityMatrix(len(qubit_indices), label=label),
                map_tuple(lambda qubit: native_qubits[qubit], qubit_indices),
                (),
            )
        case _:
            raise TypeError(f"unsupported operation: {type(operation).__name__}")


def _to_qiskit(circuit: Circuit):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    quantum_register = QuantumRegister(circuit.qubit_count, "q")
    classical_register = (
        ClassicalRegister(circuit.bit_count, "c")
        if circuit.bit_count > 0
        else None
    )
    registers = (
        (quantum_register,)
        if classical_register is None
        else (quantum_register, classical_register)
    )
    empty_native = QuantumCircuit(*registers, name=circuit.name)
    qubits = tuple(quantum_register)
    clbits = () if classical_register is None else tuple(classical_register)
    instructions = map_tuple(
        lambda operation: _to_qiskit_instruction(operation, qubits, clbits),
        circuit.operations,
    )
    instruction_sequence = QuantumCircuit.from_instructions(
        instructions,
        qubits=qubits,
        clbits=clbits,
    )
    return empty_native.compose(instruction_sequence, inplace=False)


def draw_circuit(circuit: Circuit, *, fold: int = -1):
    """Return a Matplotlib figure of an immutable circuit before transpilation."""
    return _to_qiskit(circuit).draw(
        output="mpl",
        fold=fold,
        idle_wires=True,
    )


def _compile_circuit(
    backend: _ResolvedBackend,
    circuit: Circuit,
    config: CompilerConfig,
) -> Result[_Executable, RuntimeFailure]:
    def compile_native() -> _Executable:
        from qiskit.transpiler import generate_preset_pass_manager

        original = _to_qiskit(circuit)
        pass_manager = generate_preset_pass_manager(
            target=backend.native.target,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
        )
        compiled = pass_manager.run(original)
        return _Executable(
            logical_qubit_count=circuit.qubit_count,
            original_gate_count=int(original.size()),
            original_depth=int(original.depth()),
            compiled_gate_count=int(compiled.size()),
            compiled_depth=int(compiled.depth()),
            native=compiled,
        )

    return _protected("compile", compile_native)


def _compile_circuits(
    backend: _ResolvedBackend,
    circuits: tuple[Circuit, ...],
    config: CompilerConfig,
) -> Result[tuple[_Executable, ...], RuntimeFailure]:
    """Compile a circuit batch with one shared pass manager."""

    def compile_native() -> tuple[_Executable, ...]:
        from qiskit.transpiler import generate_preset_pass_manager

        required_qubits = max(
            map(lambda circuit: circuit.qubit_count, circuits)
        )
        if required_qubits > backend.native.num_qubits:
            raise ValueError(
                f"backend {backend.name} has {backend.native.num_qubits} "
                f"qubits but the batch requires {required_qubits}"
            )
        if backend.provider in ("fake", "ibm"):
            operations = concat_map(
                lambda circuit: circuit.operations,
                circuits,
            )
            required_operations = frozenset(
                filter(
                    lambda name: name is not None,
                    (
                        "measure",
                        (
                            "reset"
                            if any(
                                map(
                                    lambda operation: isinstance(
                                        operation,
                                        Reset,
                                    ),
                                    operations,
                                )
                            )
                            else None
                        ),
                        (
                            "if_else"
                            if any(
                                map(
                                    lambda operation: isinstance(
                                        operation,
                                        ClassicallyControlledXXPlusYY,
                                    ),
                                    operations,
                                )
                            )
                            else None
                        ),
                    ),
                )
            )
            supported_operations = frozenset(
                backend.native.target.operation_names
            )
            missing_operations = required_operations - supported_operations
            if missing_operations:
                raise ValueError(
                    f"backend {backend.name} does not support required "
                    "dynamic-circuit operations: "
                    f"{', '.join(sorted(missing_operations))}"
                )

        originals = map_tuple(_to_qiskit, circuits)
        pass_manager = generate_preset_pass_manager(
            target=backend.native.target,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
        )
        compiled = tuple(
            pass_manager.run(list(originals), num_processes=1)
        )
        return map_tuple(
            lambda index: _Executable(
                logical_qubit_count=circuits[index].qubit_count,
                original_gate_count=int(originals[index].size()),
                original_depth=int(originals[index].depth()),
                compiled_gate_count=int(compiled[index].size()),
                compiled_depth=int(compiled[index].depth()),
                native=compiled[index],
            ),
            range(len(circuits)),
        )

    return _protected("compile", compile_native)


def _job_id(job: Any) -> str | None:
    candidate = getattr(job, "job_id", None)
    if not callable(candidate):
        return None
    try:
        return str(candidate())
    except Exception:
        return None


def compile_circuit_batch_sync(
    circuits: tuple[Circuit, ...],
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
) -> Result[tuple[CompilationMetrics, ...], RuntimeFailure]:
    """Resolve and transpile circuits without submitting provider work."""

    if not circuits:
        return Err(ValidationFailure(("a compilation batch cannot be empty",)))

    common_problems = (*target_errors(target), *compiler_errors(compiler))
    problems = (
        *common_problems,
        *concat_map(
            lambda indexed_circuit: map_tuple(
                lambda problem: f"circuit {indexed_circuit[0]}: {problem}",
                circuit_errors(indexed_circuit[1]),
            ),
            enumerate(circuits),
        ),
    )
    if problems:
        return Err(ValidationFailure(problems))

    match _resolve_backend(target, environment):
        case Err(error):
            return Err(error)
        case Ok(backend):
            pass

    match _compile_circuits(backend, circuits, compiler):
        case Err(error):
            return Err(error)
        case Ok(executables):
            return Ok(
                map_tuple(
                    lambda executable: CompilationMetrics(
                        original_gate_count=(
                            executable.original_gate_count
                        ),
                        original_depth=executable.original_depth,
                        compiled_gate_count=(
                            executable.compiled_gate_count
                        ),
                        compiled_depth=executable.compiled_depth,
                    ),
                    executables,
                )
            )


def draw_transpiled_circuit_layout_sync(
    circuit: Circuit,
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
    *,
    view: Literal["virtual", "physical"] = "virtual",
    logical_labels: tuple[str, ...] | None = None,
):
    """Compile a circuit and return its backend-layout Matplotlib figure."""
    common_problems = (*target_errors(target), *compiler_errors(compiler))
    problems = (
        *common_problems,
        *circuit_errors(circuit),
        *(
            ()
            if view in ("virtual", "physical")
            else ("layout view must be 'virtual' or 'physical'",)
        ),
        *(
            ()
            if logical_labels is None
            or len(logical_labels) == circuit.qubit_count
            else (
                "logical label count must match the circuit qubit count",
            )
        ),
    )
    if problems:
        return Err(ValidationFailure(problems))

    match _resolve_backend(target, environment):
        case Err(error):
            return Err(error)
        case Ok(backend):
            pass
    match _compile_circuits(backend, (circuit,), compiler):
        case Err(error):
            return Err(error)
        case Ok((executable,)):
            pass

    def draw_layout():
        layout = executable.native.layout
        physical_by_logical = (
            tuple(range(circuit.qubit_count))
            if layout is None
            else tuple(
                layout.initial_index_layout(filter_ancillas=True)
            )
        )
        if len(physical_by_logical) != circuit.qubit_count:
            raise ValueError(
                "transpiler layout does not map every logical qubit"
            )

        def add_mapping_table(figure):
            width, height = figure.get_size_inches()
            figure.set_size_inches(
                max(float(width) + 4.5, 11.0),
                max(
                    float(height),
                    min(14.0, 0.45 * (circuit.qubit_count + 3)),
                ),
            )
            if figure.axes:
                figure.axes[0].set_position((0.02, 0.06, 0.65, 0.88))
            table_axis = figure.add_axes((0.71, 0.08, 0.27, 0.84))
            table_axis.axis("off")
            table_axis.set_title(
                "Logical \u2192 physical mapping",
                fontsize=12,
                pad=12,
            )
            columns = (
                ("Logical", "Physical")
                if logical_labels is None
                else ("Logical", "Role", "Physical")
            )

            def mapping_row(index):
                endpoints = (f"q[{index}]", str(physical_by_logical[index]))
                return (
                    endpoints
                    if logical_labels is None
                    else (
                        endpoints[0],
                        logical_labels[index],
                        endpoints[1],
                    )
                )

            table = table_axis.table(
                cellText=map_tuple(
                    mapping_row,
                    range(circuit.qubit_count),
                ),
                colLabels=columns,
                colColours=("#648fff",) * len(columns),
                colWidths=(
                    (0.42, 0.58)
                    if logical_labels is None
                    else (0.25, 0.50, 0.25)
                ),
                cellLoc="center",
                colLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(
                max(7.0, min(11.0, 100.0 / (circuit.qubit_count + 4)))
            )
            table.scale(
                1.0,
                max(
                    0.72,
                    min(1.5, 12.0 / (circuit.qubit_count + 2)),
                ),
            )
            return figure

        coupling_map = getattr(backend.native, "coupling_map", None)
        if coupling_map is not None:
            import matplotlib.pyplot as plt
            import numpy as np
            import rustworkx as rx
            from matplotlib.collections import LineCollection

            active_physical = frozenset(physical_by_logical)
            logical_by_physical = dict(
                map(
                    lambda indexed: (indexed[1], indexed[0]),
                    enumerate(physical_by_logical),
                )
            )
            physical_count = backend.native.num_qubits
            active_color = "#000000"
            inactive_color = "#648fff"
            graph = coupling_map.graph.to_undirected(multigraph=False)
            coupling_edges = tuple(graph.edge_list())
            positions = rx.spring_layout(
                graph,
                seed=(
                    0
                    if compiler.seed_transpiler is None
                    else compiler.seed_transpiler
                ),
                num_iter=150,
                scale=1.0,
            )
            edge_segments = map_tuple(
                lambda edge: (
                    tuple(map(float, positions[edge[0]])),
                    tuple(map(float, positions[edge[1]])),
                ),
                coupling_edges,
            )
            edge_colors = map_tuple(
                lambda edge: (
                    active_color
                    if edge[0] in active_physical
                    and edge[1] in active_physical
                    else inactive_color
                ),
                coupling_edges,
            )
            inactive_physical = tuple(
                filter(
                    lambda physical: physical not in active_physical,
                    range(physical_count),
                )
            )
            figure, axis = plt.subplots(figsize=(10.0, 8.0))
            axis.add_collection(
                LineCollection(
                    edge_segments,
                    colors=edge_colors,
                    linewidths=map_tuple(
                        lambda color: 2.4 if color == active_color else 1.4,
                        edge_colors,
                    ),
                    zorder=1,
                )
            )
            axis.scatter(
                map_tuple(
                    lambda physical: float(positions[physical][0]),
                    inactive_physical,
                ),
                map_tuple(
                    lambda physical: float(positions[physical][1]),
                    inactive_physical,
                ),
                s=95,
                color=inactive_color,
                edgecolors=inactive_color,
                linewidths=0.8,
                zorder=2,
            )
            axis.scatter(
                map_tuple(
                    lambda physical: float(positions[physical][0]),
                    physical_by_logical,
                ),
                map_tuple(
                    lambda physical: float(positions[physical][1]),
                    physical_by_logical,
                ),
                s=430,
                color=active_color,
                edgecolors=active_color,
                linewidths=1.2,
                zorder=3,
            )

            def draw_active_label(physical):
                label = (
                    str(physical)
                    if view == "physical"
                    else str(logical_by_physical[physical])
                )
                return axis.text(
                    float(positions[physical][0]),
                    float(positions[physical][1]),
                    label,
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontfamily="DejaVu Sans",
                    fontweight="bold",
                    zorder=4,
                )

            map_tuple(draw_active_label, physical_by_logical)
            all_coordinates = np.asarray(
                map_tuple(
                    lambda physical: positions[physical],
                    range(physical_count),
                ),
                dtype=float,
            )
            axis.set_xlim(
                float(np.min(all_coordinates[:, 0])) - 0.08,
                float(np.max(all_coordinates[:, 0])) + 0.08,
            )
            axis.set_ylim(
                float(np.min(all_coordinates[:, 1])) - 0.08,
                float(np.max(all_coordinates[:, 1])) + 0.08,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            return add_mapping_table(figure)

        if backend.provider != "aer":
            raise ValueError(
                f"backend {backend.name} has no coupling map to plot"
            )

        # Aer has no physical coupling graph or calibration-defined qubit
        # coordinates. Its unconstrained mapping is therefore represented
        # explicitly instead of inventing a hardware topology.
        import matplotlib.pyplot as plt
        import numpy as np

        count = circuit.qubit_count
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        x_values = np.cos(angles)
        y_values = np.sin(angles)
        figure, axis = plt.subplots(figsize=(7, 7))
        axis.scatter(
            x_values,
            y_values,
            s=1100,
            color="black",
            edgecolors="#648fff",
            linewidths=3,
            zorder=2,
        )
        def draw_qubit_label(index):
            x_value = x_values[index]
            y_value = y_values[index]
            label = (
                f"v{index}\u2192p{index}"
                if view == "virtual"
                else f"p{index}"
            )
            return axis.text(
                x_value,
                y_value,
                label,
                color="white",
                ha="center",
                va="center",
                fontsize=11,
                zorder=3,
            )

        map_tuple(draw_qubit_label, range(count))
        axis.set_title(
            "Aer transpiled layout\n"
            "unconstrained connectivity; identity placement",
        )
        axis.set_aspect("equal")
        axis.set_xlim(-1.35, 1.35)
        axis.set_ylim(-1.35, 1.35)
        axis.axis("off")
        return add_mapping_table(figure)

    return _protected("compile", draw_layout)


def _to_qiskit_observable(
    value: Observable,
    executable: _Executable,
):
    from qiskit.quantum_info import SparsePauliOp

    terms = map_tuple(
        lambda term: (
            "".join(map(lambda factor: factor.axis, term.factors)),
            list(map(lambda factor: factor.qubit, term.factors)),
            term.coefficient,
        ),
        value.terms,
    )
    logical = SparsePauliOp.from_sparse_list(
        list(terms),
        num_qubits=executable.logical_qubit_count,
    )
    layout = executable.native.layout
    return logical if layout is None else logical.apply_layout(layout)


def _submit(
    backend: _ResolvedBackend,
    executable: _Executable,
    workload: Workload,
) -> Result[_SubmittedJob, RuntimeFailure]:
    def submit_native() -> _SubmittedJob:
        if backend.provider == "fake":
            raise ValueError(
                "fake IBM backends support offline compilation only"
            )
        match workload:
            case Sample(shots, seed_simulator):
                if backend.provider == "aer":
                    from qiskit.primitives import BackendSamplerV2

                    sampler = BackendSamplerV2(
                        backend=backend.native,
                        options={"seed_simulator": seed_simulator},
                    )
                else:
                    from qiskit_ibm_runtime import SamplerV2

                    sampler = SamplerV2(mode=backend.native)
                job = sampler.run([executable.native], shots=shots)
            case SimulateDensityMatrices():
                job = backend.native.run(executable.native)
                shots = 0
            case Estimate(observables, precision, seed_simulator):
                native_observables = map_tuple(
                    lambda value: _to_qiskit_observable(value, executable),
                    observables,
                )
                if backend.provider == "aer":
                    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

                    estimator = AerEstimatorV2(
                        options={
                            "backend_options": {
                                "method": backend.native.options.method,
                            },
                            "run_options": {
                                "seed_simulator": seed_simulator,
                            },
                        },
                    )
                else:
                    from qiskit_ibm_runtime import EstimatorV2

                    estimator = EstimatorV2(mode=backend.native)
                job = estimator.run(
                    [(executable.native, list(native_observables))],
                    precision=precision,
                )
                shots = 0
            case _:
                raise TypeError(f"unsupported workload: {type(workload).__name__}")

        return _SubmittedJob(
            provider=backend.provider,
            backend_name=backend.name,
            shots=shots,
            job_id=_job_id(job),
            workload=workload,
            original_gate_count=executable.original_gate_count,
            original_depth=executable.original_depth,
            compiled_gate_count=executable.compiled_gate_count,
            compiled_depth=executable.compiled_depth,
            native=job,
        )

    return _protected("submit", submit_native)


def _submit_sample_batch(
    backend: _ResolvedBackend,
    executables: tuple[_Executable, ...],
    workload: Sample,
) -> Result[_SubmittedSampleBatch, RuntimeFailure]:
    def submit_native() -> _SubmittedSampleBatch:
        if backend.provider == "fake":
            raise ValueError(
                "fake IBM backends support offline compilation only"
            )
        if backend.provider == "aer":
            from qiskit.primitives import BackendSamplerV2

            sampler = BackendSamplerV2(
                backend=backend.native,
                options={"seed_simulator": workload.seed_simulator},
            )
        else:
            from qiskit_ibm_runtime import SamplerV2

            sampler = SamplerV2(mode=backend.native)

        job = sampler.run(
            list(map(lambda executable: executable.native, executables)),
            shots=workload.shots,
        )
        return _SubmittedSampleBatch(
            provider=backend.provider,
            backend_name=backend.name,
            shots=workload.shots,
            job_id=_job_id(job),
            executables=executables,
            native=job,
        )

    return _protected("submit", submit_native)


def _immutable_matrix(value: Any) -> DensityMatrix:
    import numpy as np

    array = np.asarray(value, dtype=complex)
    return map_tuple(lambda row: map_tuple(complex, row), array)


def _collect(
    job: _SubmittedJob,
) -> Result[SampleResult | DensityMatrixResult | EstimateResult, RuntimeFailure]:
    def collect_native() -> SampleResult | DensityMatrixResult | EstimateResult:
        match job.workload:
            case Sample():
                pub_result = job.native.result()[0]
                counts = pub_result.join_data().get_counts()
                return SampleResult(
                    counts=normalize_counts(counts),
                    shots=job.shots,
                    backend_name=job.backend_name,
                    job_id=job.job_id,
                    original_gate_count=job.original_gate_count,
                    original_depth=job.original_depth,
                    compiled_gate_count=job.compiled_gate_count,
                    compiled_depth=job.compiled_depth,
                )
            case SimulateDensityMatrices(labels):
                result = job.native.result()
                if not result.success:
                    raise RuntimeError(f"Aer simulation failed: {result.status}")
                data = result.data(0)
                snapshots = map_tuple(
                    lambda label: (label, _immutable_matrix(data[label])),
                    labels,
                )
                return DensityMatrixResult(
                    snapshots=snapshots,
                    backend_name=job.backend_name,
                    job_id=job.job_id,
                    original_gate_count=job.original_gate_count,
                    original_depth=job.original_depth,
                    compiled_gate_count=job.compiled_gate_count,
                    compiled_depth=job.compiled_depth,
                )
            case Estimate(observables, precision, _):
                import numpy as np

                pub_result = job.native.result()[0]
                expectation_values = np.asarray(
                    pub_result.data.evs,
                    dtype=float,
                ).reshape(-1)
                standard_errors = np.asarray(
                    pub_result.data.stds,
                    dtype=float,
                ).reshape(-1)
                values = map_tuple(
                    lambda index: ExpectationValue(
                        observable=observables[index].name,
                        value=float(expectation_values[index]),
                        standard_error=float(standard_errors[index]),
                    ),
                    range(len(observables)),
                )
                return EstimateResult(
                    values=values,
                    target_precision=precision,
                    backend_name=job.backend_name,
                    job_id=job.job_id,
                    original_gate_count=job.original_gate_count,
                    original_depth=job.original_depth,
                    compiled_gate_count=job.compiled_gate_count,
                    compiled_depth=job.compiled_depth,
                )
            case _:
                raise TypeError(f"unsupported workload: {type(job.workload).__name__}")

    return _protected("collect", collect_native)


def _collect_sample_batch(
    job: _SubmittedSampleBatch,
) -> Result[tuple[SampleResult, ...], RuntimeFailure]:
    def collect_native() -> tuple[SampleResult, ...]:
        pub_results = tuple(job.native.result())
        if len(pub_results) != len(job.executables):
            raise RuntimeError(
                "Sampler returned a different number of results than circuits"
            )

        return map_tuple(
            lambda index: SampleResult(
                counts=normalize_counts(
                    pub_results[index].join_data().get_counts()
                ),
                shots=job.shots,
                backend_name=job.backend_name,
                job_id=job.job_id,
                original_gate_count=(
                    job.executables[index].original_gate_count
                ),
                original_depth=job.executables[index].original_depth,
                compiled_gate_count=(
                    job.executables[index].compiled_gate_count
                ),
                compiled_depth=job.executables[index].compiled_depth,
            ),
            range(len(job.executables)),
        )

    return _protected("collect", collect_native)


def run_sample_batch_sync(
    circuits: tuple[Circuit, ...],
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    workload: Sample = Sample(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
) -> Result[tuple[SampleResult, ...], RuntimeFailure]:
    """Compile and sample multiple circuits in one provider job."""

    if not circuits:
        return Err(ValidationFailure(("a sample batch cannot be empty",)))

    def circuit_problems(indexed_circuit):
        index, circuit = indexed_circuit
        result = validate(
            ExecutionPlan(
                circuit=circuit,
                target=target,
                compiler=compiler,
                workload=workload,
            )
        )
        return (
            map_tuple(
                lambda problem: f"circuit {index}: {problem}",
                result.error.problems,
            )
            if isinstance(result, Err)
            else ()
        )

    problems = concat_map(
        circuit_problems,
        enumerate(circuits),
    )
    if problems:
        return Err(ValidationFailure(problems))

    match _resolve_backend(target, environment):
        case Err(error):
            return Err(error)
        case Ok(backend):
            pass

    match _compile_circuits(backend, circuits, compiler):
        case Err(error):
            return Err(error)
        case Ok(executables):
            pass

    match _submit_sample_batch(backend, executables, workload):
        case Err(error):
            return Err(error)
        case Ok(job):
            return _collect_sample_batch(job)


def _is_finished(job: _SubmittedJob) -> Result[bool, RuntimeFailure]:
    def check_status() -> bool:
        in_final_state = getattr(job.native, "in_final_state", None)
        if callable(in_final_state):
            return bool(in_final_state())

        done = getattr(job.native, "done", None)
        if callable(done):
            return bool(done())

        raise TypeError("the provider job has no completion-status operation")

    return _protected("collect", check_status)


def _resolve_effect(effect: ModuleType, target: Target, environment: RuntimeEnvironment):
    if isinstance(target, IBMHardware):
        # Import on the calling thread; the blocking HTTP lookup itself can then
        # safely run in the asynchronous effect's worker thread.
        from qiskit_ibm_runtime import QiskitRuntimeService

        del QiskitRuntimeService
        return effect.blocking(lambda: _resolve_backend(target, environment))
    return effect.defer(lambda: _resolve_backend(target, environment))


def _submit_effect(
    effect: ModuleType,
    backend: _ResolvedBackend,
    executable: _Executable,
    workload: Workload,
):
    if backend.provider == "ibm":
        # As above, keep module loading on the calling thread and move only the
        # provider's synchronous network request into the worker.
        from qiskit_ibm_runtime import EstimatorV2, SamplerV2

        del EstimatorV2, SamplerV2
        return effect.blocking(lambda: _submit(backend, executable, workload))
    return effect.defer(lambda: _submit(backend, executable, workload))


def make_runtime(effect: ModuleType) -> RuntimeModule:
    """Emulate an OCaml functor parameterized by an effect module."""

    return RuntimeModule(
        pure=effect.pure,
        bind=effect.bind,
        fail=effect.fail,
        resolve_backend=lambda target, environment: _resolve_effect(
            effect, target, environment
        ),
        compile_circuit=lambda backend, circuit, config: effect.defer(
            lambda: _compile_circuit(backend, circuit, config)
        ),
        submit=lambda backend, executable, workload: _submit_effect(
            effect, backend, executable, workload
        ),
        collect=lambda job: effect.wait_until(
            lambda: _is_finished(job),
            lambda: _collect(job),
            poll_interval=0.1 if job.provider == "aer" else 5.0,
            poll_in_thread=job.provider == "ibm",
        ),
    )
