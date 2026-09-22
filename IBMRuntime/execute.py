"""Immutable execution plans, failures, and normalized results."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from .backend import Aer, Target, validation_errors as target_errors
from .circuit import (
    ClassicallyControlledXXPlusYY,
    Circuit,
    Measure,
    SaveDensityMatrix,
    validation_errors as circuit_errors,
)
from .compile import CompilerConfig, validation_errors as compiler_errors
from .functional import concat_map, exists, filter_map, map_tuple
from .observable import Observable, validation_errors as observable_errors
from .result import Err, Ok, Result
from .validation import issue_if


@dataclass(frozen=True, slots=True)
class Sample:
    shots: int = 1024
    seed_simulator: int | None = 0


@dataclass(frozen=True, slots=True)
class SimulateDensityMatrices:
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Estimate:
    observables: tuple[Observable, ...]
    precision: float = 0.015625
    seed_simulator: int | None = 0


Workload: TypeAlias = Sample | SimulateDensityMatrices | Estimate


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    circuit: Circuit
    target: Target
    compiler: CompilerConfig
    workload: Workload


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    problems: tuple[str, ...]


ExecutionStage: TypeAlias = Literal[
    "resolve_backend",
    "compile",
    "submit",
    "collect",
]


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    stage: ExecutionStage
    message: str
    exception_type: str | None = None


RuntimeFailure: TypeAlias = ValidationFailure | ExecutionFailure


@dataclass(frozen=True, slots=True)
class SampleResult:
    counts: tuple[tuple[str, int], ...]
    shots: int
    backend_name: str
    job_id: str | None
    original_gate_count: int
    original_depth: int
    compiled_gate_count: int
    compiled_depth: int


DensityMatrix: TypeAlias = tuple[tuple[complex, ...], ...]


@dataclass(frozen=True, slots=True)
class DensityMatrixResult:
    snapshots: tuple[tuple[str, DensityMatrix], ...]
    backend_name: str
    job_id: str | None
    original_gate_count: int
    original_depth: int
    compiled_gate_count: int
    compiled_depth: int


@dataclass(frozen=True, slots=True)
class ExpectationValue:
    observable: str
    value: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class EstimateResult:
    values: tuple[ExpectationValue, ...]
    target_precision: float
    backend_name: str
    job_id: str | None
    original_gate_count: int
    original_depth: int
    compiled_gate_count: int
    compiled_depth: int


RunResult: TypeAlias = SampleResult | DensityMatrixResult | EstimateResult


def execution_plan(
    circuit: Circuit,
    target: Target = Aer(),
    compiler: CompilerConfig = CompilerConfig(),
    workload: Workload = Sample(),
) -> ExecutionPlan:
    return ExecutionPlan(
        circuit=circuit,
        target=target,
        compiler=compiler,
        workload=workload,
    )


def _sample_errors(plan: ExecutionPlan, workload: Sample) -> tuple[str, ...]:
    return (
        *issue_if(workload.shots <= 0, "shots must be greater than zero"),
        *issue_if(
            workload.seed_simulator is not None and workload.seed_simulator < 0,
            "the simulator seed cannot be negative",
        ),
        *issue_if(
            not exists(
                lambda operation: isinstance(operation, Measure),
                plan.circuit.operations,
            ),
            "a sampling circuit must contain a measurement",
        ),
    )


def _density_matrix_errors(
    plan: ExecutionPlan,
    workload: SimulateDensityMatrices,
) -> tuple[str, ...]:
    circuit_labels = filter_map(
        lambda operation: (
            operation.label
            if isinstance(operation, SaveDensityMatrix)
            else None
        ),
        plan.circuit.operations,
    )
    return (
        *issue_if(
            not isinstance(plan.target, Aer),
            "density-matrix snapshots require an Aer target",
        ),
        *issue_if(
            isinstance(plan.target, Aer) and plan.target.method != "density_matrix",
            "density-matrix snapshots require Aer method 'density_matrix'",
        ),
        *issue_if(
            not workload.labels,
            "density-matrix simulation requires at least one label",
        ),
        *issue_if(
            len(frozenset(workload.labels)) != len(workload.labels),
            "density-matrix labels must be unique",
        ),
        *issue_if(
            circuit_labels != workload.labels,
            "density-matrix labels must match the circuit snapshots",
        ),
    )


def _estimate_errors(
    plan: ExecutionPlan,
    workload: Estimate,
) -> tuple[str, ...]:
    names = map_tuple(lambda value: value.name, workload.observables)
    return (
        *issue_if(
            not workload.observables,
            "estimation requires at least one observable",
        ),
        *issue_if(
            not isfinite(workload.precision) or workload.precision <= 0.0,
            "estimation precision must be finite and greater than zero",
        ),
        *issue_if(
            workload.seed_simulator is not None and workload.seed_simulator < 0,
            "the simulator seed cannot be negative",
        ),
        *issue_if(
            len(frozenset(names)) != len(names),
            "observable names must be unique",
        ),
        *issue_if(
            exists(
                lambda operation: isinstance(operation, SaveDensityMatrix),
                plan.circuit.operations,
            ),
            "an estimation circuit cannot contain snapshots",
        ),
        *issue_if(
            exists(
                lambda operation: isinstance(operation, Measure),
                plan.circuit.operations,
            )
            and not exists(
                lambda operation: isinstance(
                    operation,
                    ClassicallyControlledXXPlusYY,
                ),
                plan.circuit.operations,
            ),
            (
                "an estimation circuit can contain a measurement only when "
                "it feeds dynamic classical control"
            ),
        ),
        *concat_map(
            lambda indexed_observable: map_tuple(
                lambda problem: (
                    f"observable {indexed_observable[0]} "
                    f"({indexed_observable[1].name!r}): {problem}"
                ),
                observable_errors(
                    indexed_observable[1],
                    plan.circuit.qubit_count,
                ),
            ),
            enumerate(workload.observables),
        ),
    )


def _workload_errors(plan: ExecutionPlan) -> tuple[str, ...]:
    match plan.workload:
        case Sample() as workload:
            return _sample_errors(plan, workload)
        case SimulateDensityMatrices() as workload:
            return _density_matrix_errors(plan, workload)
        case Estimate() as workload:
            return _estimate_errors(plan, workload)
        case _:
            return (f"unsupported workload: {type(plan.workload).__name__}",)


def validate(plan: ExecutionPlan) -> Result[ExecutionPlan, ValidationFailure]:
    problems = (
        *circuit_errors(plan.circuit),
        *target_errors(plan.target),
        *compiler_errors(plan.compiler),
        *_workload_errors(plan),
    )
    return Err(ValidationFailure(problems)) if problems else Ok(plan)


def normalize_counts(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return map_tuple(
        lambda item: (item[0], int(item[1])),
        sorted(counts.items()),
    )


def counts_dict(result: SampleResult) -> dict[str, int]:
    """Create a presentation copy; the stored result remains immutable."""

    return dict(result.counts)


def snapshots_dict(result: DensityMatrixResult) -> dict[str, DensityMatrix]:
    """Create a presentation copy; stored matrices remain immutable tuples."""

    return dict(result.snapshots)


def estimates_dict(result: EstimateResult) -> dict[str, float]:
    """Create a presentation copy of expectation values keyed by name."""

    return dict(map(lambda item: (item.observable, item.value), result.values))
