"""Backend- and effect-independent execution workflow."""

from .execute import ExecutionPlan, validate
from .result import Err, Ok
from .runtime import RuntimeEnvironment, RuntimeModule, WorkflowModule


def make_workflow(runtime: RuntimeModule) -> WorkflowModule:
    """Emulate an OCaml functor ``Make_workflow (Runtime : RUNTIME)``."""

    def submit(plan: ExecutionPlan, environment: RuntimeEnvironment):
        match validate(plan):
            case Err(error):
                return runtime.fail(error)
            case Ok(valid_plan):
                return runtime.bind(
                    runtime.resolve_backend(valid_plan.target, environment),
                    lambda backend: runtime.bind(
                        runtime.compile_circuit(
                            backend,
                            valid_plan.circuit,
                            valid_plan.compiler,
                        ),
                        lambda executable: runtime.submit(
                            backend,
                            executable,
                            valid_plan.workload,
                        ),
                    ),
                )

    def run(plan: ExecutionPlan, environment: RuntimeEnvironment):
        return runtime.bind(submit(plan, environment), runtime.collect)

    return WorkflowModule(submit=submit, run=run)
