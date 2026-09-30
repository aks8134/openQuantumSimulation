"""Immutable Pauli-twirling ensemble construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable

from .core import (
    AveragedResult,
    CircuitBatch,
    CircuitT,
    CircuitVariant,
    Stage,
    as_batch,
    average,
    nonnegative_int,
    positive_int,
    with_stage,
)


Twirl = Callable[[Any, int, int | None], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class PauliTwirlingPlan:
    variants: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", positive_int(self.variants, "variants"))
        if self.seed is not None:
            nonnegative_int(self.seed, "seed")


def pauli_twirling_plan(
    *,
    variants: int = 10,
    seed: int | None = None,
) -> PauliTwirlingPlan:
    return PauliTwirlingPlan(variants, seed)


def _default_twirl(circuit: Any, variants: int, seed: int | None) -> tuple[Any, ...]:
    from .mitiq import pauli_twirl

    return pauli_twirl(circuit, variants, seed)


def construct_pauli_twirling(
    circuits: CircuitT | CircuitBatch[CircuitT],
    plan: PauliTwirlingPlan,
    *,
    twirl: Twirl = _default_twirl,
) -> CircuitBatch[CircuitT]:
    """Generate CNOT/CZ Pauli-twirled variants without executing them."""

    source = as_batch(circuits)

    def expand(indexed_variant: tuple[int, CircuitVariant[CircuitT]]):
        source_position, variant = indexed_variant
        seed = None if plan.seed is None else plan.seed + source_position
        generated = tuple(twirl(variant.circuit, plan.variants, seed))
        if len(generated) != plan.variants:
            raise RuntimeError(
                f"twirling returned {len(generated)} circuits; expected {plan.variants}"
            )
        return tuple(
            map(
                lambda indexed_circuit: with_stage(
                    variant,
                    indexed_circuit[1],
                    Stage(
                        "pauli_twirling",
                        indexed_circuit[0],
                        (("seed", seed),),
                    ),
                ),
                enumerate(generated),
            )
        )

    variants = reduce(
        lambda accumulated, group: (*accumulated, *group),
        map(expand, enumerate(source.variants)),
        (),
    )
    return CircuitBatch(variants)


def combine_pauli_twirling(values) -> AveragedResult:
    return average(tuple(values))
