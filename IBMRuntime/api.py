"""Public execution functions."""

from .backend import Target
from .circuit import Circuit
from .compile import CompilerConfig
from .effects import async_ as async_effect
from .effects import sync as sync_effect
from .execute import ExecutionPlan, Sample
from .interpreters.qiskit import (
    compile_circuit_batch_sync as _compile_circuit_batch_sync,
    draw_transpiled_circuit_layout_sync as _draw_transpiled_circuit_layout_sync,
    make_runtime,
    run_sample_batch_sync as _run_sample_batch_sync,
)
from .runtime import RuntimeEnvironment
from .workflow import make_workflow


_SYNC_WORKFLOW = make_workflow(make_runtime(sync_effect))
_ASYNC_WORKFLOW = make_workflow(make_runtime(async_effect))


def compile_circuit_batch_sync(
    circuits: tuple[Circuit, ...],
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
):
    """Transpile circuits and return metrics without submitting a job."""

    return _compile_circuit_batch_sync(
        circuits,
        target,
        compiler,
        environment,
    )


def draw_transpiled_circuit_layout_sync(
    circuit: Circuit,
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
    *,
    view="virtual",
):
    """Compile a circuit and draw its backend qubit placement."""
    return _draw_transpiled_circuit_layout_sync(
        circuit,
        target,
        compiler,
        environment,
        view=view,
    )


def run_sync(
    plan: ExecutionPlan,
    environment: RuntimeEnvironment = RuntimeEnvironment(),
):
    return _SYNC_WORKFLOW.run(plan, environment)


async def run_async(
    plan: ExecutionPlan,
    environment: RuntimeEnvironment = RuntimeEnvironment(),
):
    return await _ASYNC_WORKFLOW.run(plan, environment)


def run_sample_batch_sync(
    circuits: tuple[Circuit, ...],
    target: Target,
    compiler: CompilerConfig = CompilerConfig(),
    workload: Sample = Sample(),
    environment: RuntimeEnvironment = RuntimeEnvironment(),
):
    """Compile and sample a tuple of circuits in one provider job."""

    return _run_sample_batch_sync(
        circuits,
        target,
        compiler,
        workload,
        environment,
    )
