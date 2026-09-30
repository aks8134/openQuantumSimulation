"""Functional zero-noise extrapolation planning and inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, reduce
from itertools import product
from numbers import Real
from typing import Any, Callable, Literal

import numpy as np

from .core import (
    CircuitBatch,
    CircuitT,
    CircuitVariant,
    Stage,
    as_batch,
    finite_real,
    nonnegative_int,
    positive_int,
    with_stage,
)


ScalingMethod = Literal["random_local", "fold_all", "global"]
InferenceMethod = Literal["linear", "richardson", "polynomial"]
ScaleNoise = Callable[[Any, float, int | None], Any]


@dataclass(frozen=True, slots=True)
class ZNEScaling:
    method: ScalingMethod = "random_local"
    fidelities: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.method not in ("random_local", "fold_all", "global"):
            raise ValueError(f"unsupported ZNE scaling method: {self.method}")
        converted = tuple(
            map(
                lambda item: (
                    str(item[0]),
                    finite_real(item[1], f"fidelity {item[0]}"),
                ),
                self.fidelities,
            )
        )
        if any(map(lambda item: not 0.0 <= item[1] <= 1.0, converted)):
            raise ValueError("every fidelity must lie in [0, 1]")
        object.__setattr__(self, "fidelities", converted)


@dataclass(frozen=True, slots=True)
class ZNEInference:
    method: InferenceMethod = "richardson"
    order: int | None = None

    def __post_init__(self) -> None:
        if self.method not in ("linear", "richardson", "polynomial"):
            raise ValueError(f"unsupported ZNE inference method: {self.method}")
        if self.method == "polynomial":
            if self.order is None:
                raise ValueError("polynomial inference requires an order")
            positive_int(self.order, "polynomial order")
        elif self.order is not None:
            raise ValueError("order is only valid for polynomial inference")


@dataclass(frozen=True, slots=True)
class ZNEPlan:
    scale_factors: tuple[float, ...]
    repetitions: int = 1
    scaling: ZNEScaling = ZNEScaling()
    inference: ZNEInference = ZNEInference()
    seed: int | None = None

    def __post_init__(self) -> None:
        factors = tuple(
            map(lambda value: finite_real(value, "scale factor"), self.scale_factors)
        )
        if len(factors) < 2:
            raise ValueError("ZNE requires at least two scale factors")
        if any(map(lambda value: value < 1.0, factors)):
            raise ValueError("ZNE scale factors must be at least one")
        if tuple(sorted(factors)) != factors or len(set(factors)) != len(factors):
            raise ValueError("ZNE scale factors must be unique and increasing")
        repetitions = positive_int(self.repetitions, "repetitions")
        if self.seed is not None:
            nonnegative_int(self.seed, "seed")
        degree = _degree(self.inference, len(factors))
        if degree >= len(factors):
            raise ValueError("inference degree must be smaller than factor count")
        object.__setattr__(self, "scale_factors", factors)
        object.__setattr__(self, "repetitions", repetitions)


@dataclass(frozen=True, slots=True)
class ZNEResult:
    estimate: float
    scale_factors: tuple[float, ...]
    scale_estimates: tuple[float, ...]
    scale_standard_errors: tuple[float, ...]
    raw_values: tuple[float, ...]
    coefficients: tuple[float, ...]
    residual_rms: float


def scaling(
    method: ScalingMethod = "random_local",
    *,
    fidelities: tuple[tuple[str, Real], ...] = (),
) -> ZNEScaling:
    return ZNEScaling(method, tuple(fidelities))


def inference(
    method: InferenceMethod = "richardson",
    *,
    order: int | None = None,
) -> ZNEInference:
    return ZNEInference(method, order)


def zne_plan(
    scale_factors: tuple[Real, ...],
    *,
    repetitions: int = 1,
    scaling: ZNEScaling = ZNEScaling(),
    inference: ZNEInference = ZNEInference(),
    seed: int | None = None,
) -> ZNEPlan:
    return ZNEPlan(tuple(scale_factors), repetitions, scaling, inference, seed)


def _degree(specification: ZNEInference, factor_count: int) -> int:
    return (
        1
        if specification.method == "linear"
        else factor_count - 1
        if specification.method == "richardson"
        else int(specification.order)
    )


def _default_scale(plan: ZNEPlan) -> ScaleNoise:
    from .mitiq import scale_noise

    return partial(scale_noise, scaling=plan.scaling)


def _seed(plan: ZNEPlan, source_position: int, index: int) -> int | None:
    return (
        None
        if plan.seed is None
        else plan.seed
        + source_position * len(plan.scale_factors) * plan.repetitions
        + index
    )


def construct_zne(
    circuits: CircuitT | CircuitBatch[CircuitT],
    plan: ZNEPlan,
    *,
    scale_noise: ScaleNoise | None = None,
) -> CircuitBatch[CircuitT]:
    """Construct all scaled circuits without executing them."""

    source = as_batch(circuits)
    transform = _default_scale(plan) if scale_noise is None else scale_noise
    coordinates = tuple(product(range(len(plan.scale_factors)), range(plan.repetitions)))

    def expand(indexed_variant: tuple[int, CircuitVariant[CircuitT]]):
        source_position, variant = indexed_variant

        def one(coordinate: tuple[int, int]) -> CircuitVariant[CircuitT]:
            factor_index, repetition = coordinate
            flat_index = factor_index * plan.repetitions + repetition
            factor = plan.scale_factors[factor_index]
            transformed = transform(
                variant.circuit,
                factor,
                _seed(plan, source_position, flat_index),
            )
            return with_stage(
                variant,
                transformed,
                Stage(
                    "zne",
                    flat_index,
                    (
                        ("scale_factor", factor),
                        ("repetition", repetition),
                        ("scaling", plan.scaling.method),
                    ),
                ),
            )

        return tuple(map(one, coordinates))

    variants = reduce(
        lambda accumulated, group: (*accumulated, *group),
        map(expand, enumerate(source.variants)),
        (),
    )
    return CircuitBatch(variants)


def _group(values: tuple[float, ...], count: int) -> tuple[tuple[float, ...], ...]:
    return tuple(map(lambda start: values[start : start + count], range(0, len(values), count)))


def combine_zne(plan: ZNEPlan, values: tuple[Real, ...]) -> ZNEResult:
    """Average repetitions and extrapolate the expectation to zero noise."""

    converted = tuple(map(lambda value: finite_real(value, "expectation"), values))
    expected = len(plan.scale_factors) * plan.repetitions
    if len(converted) != expected:
        raise ValueError(f"expected {expected} ZNE values, received {len(converted)}")
    groups = _group(converted, plan.repetitions)
    means = tuple(map(lambda group: float(np.mean(group)), groups))
    errors = tuple(
        map(
            lambda group: (
                0.0
                if len(group) == 1
                else float(np.std(group, ddof=1) / np.sqrt(len(group)))
            ),
            groups,
        )
    )
    degree = _degree(plan.inference, len(plan.scale_factors))
    design = np.vander(np.asarray(plan.scale_factors), degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, np.asarray(means), rcond=None)[0]
    fitted = design @ coefficients
    residual_rms = float(np.sqrt(np.mean(np.square(np.asarray(means) - fitted))))
    return ZNEResult(
        estimate=float(coefficients[0]),
        scale_factors=plan.scale_factors,
        scale_estimates=means,
        scale_standard_errors=errors,
        raw_values=converted,
        coefficients=tuple(map(float, coefficients)),
        residual_rms=residual_rms,
    )
