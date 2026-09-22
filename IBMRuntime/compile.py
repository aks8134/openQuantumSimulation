"""Pure compilation configuration and requests.

Actual Qiskit transpilation belongs to an interpreter, not this module.
"""

from dataclasses import dataclass

from .backend import Target
from .circuit import Circuit
from .validation import issue_if


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    optimization_level: int = 3
    seed_transpiler: int | None = 0


@dataclass(frozen=True, slots=True)
class CompileRequest:
    circuit: Circuit
    target: Target
    config: CompilerConfig


def request_compilation(
    circuit: Circuit,
    target: Target,
    config: CompilerConfig = CompilerConfig(),
) -> CompileRequest:
    return CompileRequest(circuit=circuit, target=target, config=config)


def validation_errors(config: CompilerConfig) -> tuple[str, ...]:
    return (
        *issue_if(
            config.optimization_level not in (0, 1, 2, 3),
            "optimization level must be one of 0, 1, 2, or 3",
        ),
        *issue_if(
            config.seed_transpiler is not None and config.seed_transpiler < 0,
            "the transpiler seed cannot be negative",
        ),
    )
