"""Functional digital-dynamical-decoupling circuit construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Literal

from .core import (
    AveragedResult,
    CircuitBatch,
    CircuitT,
    CircuitVariant,
    Stage,
    as_batch,
    average,
    positive_int,
    with_stage,
)


DDDRule = Literal["xx", "yy", "xyxy"]
InsertDDD = Callable[[Any, DDDRule, int, int], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class DDDPlan:
    rule: DDDRule = "xyxy"
    spacing: int = -1
    trials: int = 1

    def __post_init__(self) -> None:
        if self.rule not in ("xx", "yy", "xyxy"):
            raise ValueError(f"unsupported DDD rule: {self.rule}")
        if not isinstance(self.spacing, int) or isinstance(self.spacing, bool):
            raise TypeError("spacing must be an integer")
        if self.spacing < -1:
            raise ValueError("spacing must be -1 or non-negative")
        object.__setattr__(self, "trials", positive_int(self.trials, "trials"))


def ddd_plan(
    rule: DDDRule = "xyxy",
    *,
    spacing: int = -1,
    trials: int = 1,
) -> DDDPlan:
    return DDDPlan(rule, spacing, trials)


def _default_insert(
    circuit: Any,
    rule: DDDRule,
    spacing: int,
    trials: int,
) -> tuple[Any, ...]:
    from .mitiq import insert_ddd

    return insert_ddd(circuit, rule, spacing, trials)


def construct_ddd(
    circuits: CircuitT | CircuitBatch[CircuitT],
    plan: DDDPlan,
    *,
    insert: InsertDDD = _default_insert,
) -> CircuitBatch[CircuitT]:
    """Insert a DDD sequence into every circuit in a batch."""

    source = as_batch(circuits)

    def expand(variant: CircuitVariant[CircuitT]):
        generated = tuple(insert(variant.circuit, plan.rule, plan.spacing, plan.trials))
        if len(generated) != plan.trials:
            raise RuntimeError(
                f"DDD returned {len(generated)} circuits; expected {plan.trials}"
            )
        return tuple(
            map(
                lambda indexed_circuit: with_stage(
                    variant,
                    indexed_circuit[1],
                    Stage(
                        "ddd",
                        indexed_circuit[0],
                        (("rule", plan.rule), ("spacing", plan.spacing)),
                    ),
                ),
                enumerate(generated),
            )
        )

    variants = reduce(
        lambda accumulated, group: (*accumulated, *group),
        map(expand, source.variants),
        (),
    )
    return CircuitBatch(variants)


def combine_ddd(values) -> AveragedResult:
    return average(tuple(values))
