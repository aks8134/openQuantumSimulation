"""Our own fold selection and zero-noise extrapolation, with lazy NumPy use."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite, sqrt
from random import Random, SystemRandom

from .core import Failure, Result, attempt, integer, names, real


@dataclass(frozen=True, slots=True)
class FoldPlan:
    scale_factor: float = 1.0
    seed: int | None = 0
    gate_names: tuple[str, ...] | None = None
    excluded_names: tuple[str, ...] = ("rz",)
    weights: tuple[tuple[str, float], ...] = ()
    order: str = "random"
    max_inserted_operations: int = 1_000_000

    def __post_init__(self):
        factor = real(self.scale_factor, "scale_factor")
        if factor < 1:
            raise ValueError("scale_factor must be at least one")
        object.__setattr__(self, "scale_factor", factor)
        if self.seed is not None:
            object.__setattr__(self, "seed", integer(self.seed, "seed"))
        if self.gate_names is not None:
            object.__setattr__(self, "gate_names", names(self.gate_names, "gate_names"))
        object.__setattr__(self, "excluded_names", names(self.excluded_names, "excluded_names"))
        normalized = tuple((name, real(weight, "gate weight")) for name, weight in self.weights)
        names(tuple(name for name, _ in normalized), "weight gate names")
        if any(weight <= 0 for _, weight in normalized):
            raise ValueError("gate weights must be positive")
        object.__setattr__(self, "weights", normalized)
        if self.order not in ("random", "left", "right"):
            raise ValueError("fold order must be random, left, or right")
        object.__setattr__(self, "max_inserted_operations", integer(
            self.max_inserted_operations, "max_inserted_operations", minimum=1,
        ))


@dataclass(frozen=True, slots=True)
class FoldSelection:
    repetitions: tuple[int, ...]
    requested_scale: float
    achieved_scale: float
    seed: int
    eligible_weight: float


def select_folds(weights, plan: FoldPlan) -> Result[FoldSelection, Failure]:
    """Select whole local folds, rounding to the nearest randomized prefix.

    Each fold adds two occurrences. Weighted scaling is a declared proxy for
    noise, not a measurement of physical noise. Fractional scales are quantized.
    """
    def calculate():
        normalized = tuple(real(value, "weight") for value in weights)
        if any(value <= 0 for value in normalized):
            raise ValueError("selection weights must be positive")
        seed = plan.seed if plan.seed is not None else SystemRandom().getrandbits(64)
        total = sum(normalized)
        if not isfinite(total):
            raise ValueError("total selection weight must be finite; rescale the weights")
        if not total:
            if plan.scale_factor > 1:
                raise ValueError("cannot scale a circuit with no eligible gates")
            return FoldSelection((), plan.scale_factor, 1.0, seed, 0.0)
        rounds = floor((plan.scale_factor - 1) / 2)
        if 2 * rounds * len(normalized) > plan.max_inserted_operations:
            raise ValueError("fold selection exceeds max_inserted_operations")
        counts = [rounds] * len(normalized)
        residual = ((plan.scale_factor - 1) / 2 - rounds) * total
        order = list(range(len(normalized)))
        if plan.order == "random":
            Random(seed).shuffle(order)
        elif plan.order == "right":
            order.reverse()
        prefix, best_prefix, best_error = 0.0, 0, abs(residual)
        for position, index in enumerate(order, start=1):
            prefix += normalized[index]
            if abs(prefix - residual) < best_error:
                best_prefix, best_error = position, abs(prefix - residual)
        for index in order[:best_prefix]:
            counts[index] += 1
        if 2 * sum(counts) > plan.max_inserted_operations:
            raise ValueError("fold selection exceeds max_inserted_operations")
        achieved = 1 + 2 * sum(k * (w / total) for k, w in zip(counts, normalized))
        return FoldSelection(tuple(counts), plan.scale_factor, achieved, seed, total)
    return attempt("zne-selection", calculate)


@dataclass(frozen=True, slots=True)
class ZNEPlan:
    scale_factors: tuple[float, ...] = (1.0, 1.5, 2.0)
    inference: str = "linear"
    degree: int = 1
    max_condition: float = 1e12

    def __post_init__(self):
        scales = tuple(real(value, "scale_factor") for value in self.scale_factors)
        if not scales or any(value < 1 for value in scales):
            raise ValueError("scale factors must be nonempty and at least one")
        if any(b <= a for a, b in zip(scales, scales[1:])):
            raise ValueError("scale factors must be strictly increasing")
        if self.inference not in ("none", "linear", "polynomial", "richardson"):
            raise ValueError("inference must be none, linear, polynomial, or richardson")
        degree = integer(self.degree, "degree", minimum=1)
        needed = degree + 1 if self.inference == "polynomial" else 2
        if self.inference == "none":
            if scales != (1.0,):
                raise ValueError("inference=none requires just scale one")
        elif len(scales) < needed:
            raise ValueError("not enough scales for the requested fit")
        limit = real(self.max_condition, "max_condition")
        if limit <= 1:
            raise ValueError("max_condition must exceed one")
        object.__setattr__(self, "scale_factors", scales)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "max_condition", limit)


@dataclass(frozen=True, slots=True)
class ZNEEstimate:
    estimate: float
    standard_error: float | None
    weights: tuple[float, ...]
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    condition_number: float
    scales: tuple[float, ...]


def extrapolate(plan: ZNEPlan, values, *, standard_errors=None, covariance=None) -> Result[ZNEEstimate, Failure]:
    """Unweighted least-squares intercept, with optional covariance propagation.

    Supply actual achieved scales in the plan. Errors are NOT inferred from
    shot counts; covariance must include correlations when estimates share data.
    Values and extrapolated estimates are never clipped to observable bounds.
    """
    def calculate():
        import numpy as np
        y = np.asarray(tuple(real(value, "estimate") for value in values))
        n = len(plan.scale_factors)
        if len(y) != n:
            raise ValueError("one estimate is required per scale")
        degree = (n - 1 if plan.inference == "richardson" else
                  0 if plan.inference == "none" else
                  1 if plan.inference == "linear" else plan.degree)
        design = np.vander(plan.scale_factors, degree + 1, increasing=True)
        condition = float(np.linalg.cond(design))
        if condition > plan.max_condition:
            raise ValueError("extrapolation is ill-conditioned; choose better-separated scales")
        intercept = np.zeros(degree + 1)
        intercept[0] = 1
        weights = np.linalg.lstsq(design.T, intercept, rcond=None)[0]
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ coefficients
        if covariance is not None and standard_errors is not None:
            raise ValueError("supply covariance or standard_errors, not both")
        uncertainty = None
        if standard_errors is not None:
            errors = tuple(real(value, "standard_error") for value in standard_errors)
            if len(errors) != n or any(value < 0 for value in errors):
                raise ValueError("standard_errors must be nonnegative and match scales")
            covariance_matrix = np.diag(np.square(errors))
        elif covariance is not None:
            if np.iscomplexobj(covariance):
                raise ValueError("covariance must be real")
            covariance_matrix = np.asarray(covariance, dtype=float)
        else:
            covariance_matrix = None
        if covariance_matrix is not None:
            if covariance_matrix.shape != (n, n) or not np.isfinite(covariance_matrix).all():
                raise ValueError("covariance must be a finite square matrix matching scales")
            if not np.allclose(covariance_matrix, covariance_matrix.T, atol=1e-12, rtol=1e-10):
                raise ValueError("covariance must be symmetric")
            if np.linalg.eigvalsh(covariance_matrix).min() < -1e-12:
                raise ValueError("covariance must be positive semidefinite")
            uncertainty = sqrt(max(0.0, float(weights @ covariance_matrix @ weights)))
        return ZNEEstimate(
            float(weights @ y), uncertainty, tuple(map(float, weights)),
            tuple(map(float, fitted)), tuple(map(float, y - fitted)), condition,
            plan.scale_factors,
        )
    return attempt("zne-extrapolation", calculate)
