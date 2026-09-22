"""OCaml-module-like signatures represented as immutable records of functions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .backend import Target
from .circuit import Circuit
from .compile import CompilerConfig
from .execute import ExecutionPlan, RuntimeFailure, Workload


@dataclass(frozen=True, slots=True)
class IBMAccount:
    api_key: str = field(repr=False)
    instance: str | None = None
    channel: str = "ibm_quantum_platform"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    ibm_account: IBMAccount | None = None


@dataclass(frozen=True, slots=True)
class RuntimeModule:
    """The Python equivalent of an OCaml ``RUNTIME`` module signature.

    ``Any`` is restricted to this effect boundary because Python's type system
    cannot express the abstract higher-kinded type ``'a Runtime.t``.
    """

    pure: Callable[[Any], Any]
    bind: Callable[[Any, Callable[[Any], Any]], Any]
    fail: Callable[[RuntimeFailure], Any]
    resolve_backend: Callable[[Target, RuntimeEnvironment], Any]
    compile_circuit: Callable[[Any, Circuit, CompilerConfig], Any]
    submit: Callable[[Any, Any, Workload], Any]
    collect: Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class WorkflowModule:
    """Functions returned by the workflow functor."""

    submit: Callable[[ExecutionPlan, RuntimeEnvironment], Any]
    run: Callable[[ExecutionPlan, RuntimeEnvironment], Any]
