"""Classical sample-complexity study for Experiment 5.

This script performs no circuit construction and submits no jobs.  It evolves
density matrices under the same randomized single-boundary, even--odd Lie
strategy as Experiment 5 and compares its observable estimates with the exact
Lindblad solution.

The study jointly tunes R, the number of independently randomized complete
boundary-choice trajectories, and the maximum physical step ``delta_t``.
It selects the largest admissible ``delta_t`` and then the smallest admissible
R at that step.  ``R * S(delta_t)`` is retained as a cost diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical


J = 1.0
h_field = 0.0
gamma = 0.2

DEFAULT_NUMBER_OF_QUBITS = 4
DEFAULT_FINAL_TIME = 10.0
DEFAULT_TIME_POINTS = 51
DEFAULT_TROTTER_DELTA_TS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
DEFAULT_SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_REPLICATES = 20
DEFAULT_ERROR_TOLERANCE = 0.1
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 29

FAMILY_NAMES = ("populations", "correlations", "flows")
METRIC_NAMES = ("maximum_aggregate_error", "rmse", "maximum_absolute_error")


@dataclass(frozen=True, slots=True)
class TimeSchedule:
    """Observation times and independent substeps from zero to each time."""

    times: np.ndarray
    target_substep_dts: tuple[tuple[float, ...], ...]
    trotter_delta_t: float | None

    @property
    def substep_dts(self) -> tuple[float, ...]:
        return tuple(
            dt
            for target in self.target_substep_dts
            for dt in target
        )

    @property
    def target_end_offsets(self) -> tuple[int, ...]:
        return tuple(np.cumsum(tuple(map(len, self.target_substep_dts)), dtype=int))

    @property
    def final_time_substep_count(self) -> int:
        return len(self.target_substep_dts[-1])

    @property
    def validation_substep_count(self) -> int:
        return len(self.substep_dts)


@dataclass(frozen=True, slots=True)
class StepChannel:
    """One even--odd system factor followed by either boundary channel."""

    dt: float
    system_unitary: np.ndarray
    damping_cosine: float


@dataclass(frozen=True, slots=True)
class ClassicalProblem:
    """Matrices shared by exact and randomized simulations."""

    number_of_qubits: int
    initial_density_matrix: np.ndarray
    initial_single_excitation_amplitudes: np.ndarray
    liouvillian: csr_matrix
    observable_families: tuple[tuple[csr_matrix, ...], ...]
    observables: np.ndarray
    family_sizes: tuple[int, ...]
    step_channels: tuple[StepChannel, ...]
    schedule: TimeSchedule


@dataclass(frozen=True, slots=True)
class ComplexityStudy:
    """Complete in-memory result of a classical convergence study."""

    sample_counts: tuple[int, ...]
    exact_observables: np.ndarray
    expected_randomized_observables: np.ndarray
    replicate_estimates: np.ndarray
    errors_vs_exact: dict[str, np.ndarray]
    errors_vs_expected_randomized: dict[str, np.ndarray]
    summaries_vs_exact: tuple[dict[str, object], ...]
    summaries_vs_expected_randomized: tuple[dict[str, object], ...]
    bias_metrics: dict[str, float]
    first_passing_sample_count: int | None
    recommended_sample_count: int | None
    representative_replicate: int
    representative_observables: np.ndarray


@dataclass(frozen=True, slots=True)
class JointComplexityStudy:
    """A lexicographic search over physical step and trajectory count."""

    delta_ts: tuple[float, ...]
    problems: tuple[ClassicalProblem, ...]
    studies: tuple[ComplexityStudy, ...]
    pair_records: tuple[dict[str, object], ...]
    optimal_delta_t: float | None
    optimal_sample_count: int | None
    optimal_substep_count: int | None
    optimal_total_sampled_substeps: int | None
    optimal_delta_t_index: int | None
    optimal_sample_count_index: int | None


def positive_integer(value: str) -> int:
    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def at_least_two(value: str) -> int:
    converted = int(value)
    if converted < 2:
        raise argparse.ArgumentTypeError("number of qubits must be at least two")
    return converted


def positive_float(value: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return converted


def probability(value: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or not 0.0 < converted < 1.0:
        raise argparse.ArgumentTypeError("value must lie strictly between zero and one")
    return converted


def _validated_times(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must contain only finite values")
    if not np.isclose(values[0], 0.0):
        raise ValueError("the classical convergence grid must start at t=0")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return values


def _target_substeps(target_time: float, maximum_dt: float | None) -> tuple[float, ...]:
    if np.isclose(target_time, 0.0):
        return ()
    if maximum_dt is None:
        return (float(target_time),)
    step_count = max(1, int(np.ceil(target_time / maximum_dt - 1e-12)))
    remainder = target_time - (step_count - 1) * maximum_dt
    return (
        *(float(maximum_dt) for _ in range(step_count - 1)),
        float(remainder),
    )


def build_time_schedule(
    times: np.ndarray,
    trotter_delta_t: float | None,
) -> TimeSchedule:
    values = _validated_times(times)
    if trotter_delta_t is not None and (
        not np.isfinite(trotter_delta_t) or trotter_delta_t <= 0.0
    ):
        raise ValueError("trotter_delta_t must be finite and positive")
    targets = tuple(
        _target_substeps(float(target_time), trotter_delta_t)
        for target_time in values
    )
    return TimeSchedule(values, targets, trotter_delta_t)


def _single_excitation_layer_hamiltonian(
    number_of_qubits: int,
    bonds: tuple[int, ...],
) -> np.ndarray:
    layer = np.zeros((number_of_qubits, number_of_qubits), dtype=complex)
    for bond in bonds:
        layer[bond, bond + 1] = J
        layer[bond + 1, bond] = J
    return layer


def _system_unitary(
    dt: float,
    number_of_qubits: int,
) -> np.ndarray:
    even_bonds = tuple(range(0, number_of_qubits - 1, 2))
    odd_bonds = tuple(range(1, number_of_qubits - 1, 2))
    even_hamiltonian = _single_excitation_layer_hamiltonian(
        number_of_qubits,
        even_bonds,
    )
    odd_hamiltonian = _single_excitation_layer_hamiltonian(
        number_of_qubits,
        odd_bonds,
    )
    field_hamiltonian = (
        0.5
        * h_field
        * (number_of_qubits - 2)
        * np.eye(number_of_qubits, dtype=complex)
    )
    even = expm(-1j * dt * even_hamiltonian)
    odd = expm(-1j * dt * odd_hamiltonian)
    field = expm(-1j * dt * field_hamiltonian)
    return field @ odd @ even


def _step_channel(
    number_of_qubits: int,
    dt: float,
) -> StepChannel:
    return StepChannel(
        dt=dt,
        system_unitary=_system_unitary(dt, number_of_qubits),
        damping_cosine=float(np.cos(np.sqrt(2.0 * gamma * dt))),
    )


def build_problem(
    number_of_qubits: int,
    times: np.ndarray,
    trotter_delta_t: float | None,
) -> ClassicalProblem:
    """Build the exact model and finite-step randomized channel data."""

    schedule = build_time_schedule(times, trotter_delta_t)
    (
        hamiltonian,
        jump_operators,
        x_operators,
        y_operators,
        z_operators,
    ) = classical.build_linear_chain(
        number_of_qubits=number_of_qubits,
        J=J,
        h=h_field,
        gamma=gamma,
    )
    initial_state = classical.computational_state(
        number_of_qubits,
        excited_sites=(number_of_qubits // 2,),
    )
    initial_density_matrix = np.outer(initial_state, initial_state.conj())
    initial_single_excitation_amplitudes = np.zeros(
        number_of_qubits,
        dtype=complex,
    )
    initial_single_excitation_amplitudes[number_of_qubits // 2] = 1.0
    dimension = 2**number_of_qubits
    identity = eye(dimension, dtype=complex, format="csr")
    observable_families = tuple(
        tuple(family)
        for family in classical.build_observables(
            identity,
            x_operators,
            y_operators,
            z_operators,
            J,
        )
    )
    observables = np.asarray(
        tuple(
            operator.toarray()
            for family in observable_families
            for operator in family
        )
    )
    channel_cache: dict[float, StepChannel] = {}

    def channel_for(dt: float) -> StepChannel:
        key = round(float(dt), 15)
        if key not in channel_cache:
            channel_cache[key] = _step_channel(number_of_qubits, dt)
        return channel_cache[key]

    step_channels = tuple(map(channel_for, schedule.substep_dts))
    return ClassicalProblem(
        number_of_qubits=number_of_qubits,
        initial_density_matrix=initial_density_matrix,
        initial_single_excitation_amplitudes=(
            initial_single_excitation_amplitudes
        ),
        liouvillian=classical.build_liouvillian(hamiltonian, jump_operators),
        observable_families=observable_families,
        observables=observables,
        family_sizes=tuple(map(len, observable_families)),
        step_channels=step_channels,
        schedule=schedule,
    )


def _unitary_batch(density_matrices: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,bjk,lk->bil",
        unitary,
        density_matrices,
        unitary.conj(),
        optimize=True,
    )


def _observable_batch(
    density_matrices: np.ndarray,
    observables: np.ndarray,
) -> np.ndarray:
    values = np.einsum(
        "bij,oji->bo",
        density_matrices,
        observables,
        optimize=True,
    )
    return np.real_if_close(values).real


def _single_excitation_amplitude_observables(
    amplitudes: np.ndarray,
) -> np.ndarray:
    populations = np.abs(amplitudes) ** 2
    coherences = amplitudes[:, :-1] * amplitudes[:, 1:].conj()
    correlations = 4.0 * np.real(coherences)
    flows = 2.0 * J * np.imag(coherences)
    return np.concatenate((populations, correlations, flows), axis=1)


def _single_excitation_density_observables(
    density_matrices: np.ndarray,
) -> np.ndarray:
    populations = np.real(np.diagonal(density_matrices, axis1=1, axis2=2))
    sites = np.arange(density_matrices.shape[1] - 1)
    coherences = density_matrices[:, sites, sites + 1]
    correlations = 4.0 * np.real(coherences)
    flows = 2.0 * J * np.imag(coherences)
    return np.concatenate((populations, correlations, flows), axis=1)


def simulate_boundary_choices(
    problem: ClassicalProblem,
    boundary_choices: np.ndarray,
) -> np.ndarray:
    """Return exact per-trajectory observables for supplied L/R choices.

    The output shape is ``(trajectory, observable, saved_time)``. Choice zero
    means the left boundary and choice one means the right boundary.
    """

    choices = np.asarray(boundary_choices, dtype=int)
    if choices.ndim != 2:
        raise ValueError("boundary_choices must be a two-dimensional array")
    if choices.shape[1] != len(problem.step_channels):
        raise ValueError("boundary_choices has the wrong physical-substep count")
    if np.any((choices != 0) & (choices != 1)):
        raise ValueError("boundary choices must be zero (L) or one (R)")

    trajectory_count = choices.shape[0]
    amplitudes = np.broadcast_to(
        problem.initial_single_excitation_amplitudes,
        (trajectory_count, problem.number_of_qubits),
    ).copy()
    saved = np.empty(
        (
            trajectory_count,
            problem.observables.shape[0],
            problem.schedule.times.size,
        ),
        dtype=float,
    )
    saved[:, :, 0] = _single_excitation_amplitude_observables(amplitudes)
    channel_index = 0
    for time_index, target_substeps in enumerate(
        problem.schedule.target_substep_dts
    ):
        amplitudes = np.broadcast_to(
            problem.initial_single_excitation_amplitudes,
            (trajectory_count, problem.number_of_qubits),
        ).copy()
        for _ in target_substeps:
            channel = problem.step_channels[channel_index]
            amplitudes = amplitudes @ channel.system_unitary.T
            use_right = choices[:, channel_index].astype(bool)
            amplitudes[~use_right, 0] *= channel.damping_cosine
            amplitudes[use_right, -1] *= channel.damping_cosine
            channel_index += 1
        saved[:, :, time_index] = _single_excitation_amplitude_observables(amplitudes)
    return saved


def expected_randomized_observables(problem: ClassicalProblem) -> np.ndarray:
    """Evolve the exact mean finite-step channel, without Monte Carlo noise."""

    initial = problem.initial_single_excitation_amplitudes
    initial_density_matrix = np.outer(initial, initial.conj())[np.newaxis, :, :]
    saved = np.empty(
        (problem.observables.shape[0], problem.schedule.times.size),
        dtype=float,
    )
    channel_index = 0
    for time_index, target_substeps in enumerate(
        problem.schedule.target_substep_dts
    ):
        density_matrix = initial_density_matrix.copy()
        for _ in target_substeps:
            channel = problem.step_channels[channel_index]
            after_system = _unitary_batch(density_matrix, channel.system_unitary)
            left_scale = np.ones(problem.number_of_qubits)
            right_scale = np.ones(problem.number_of_qubits)
            left_scale[0] = channel.damping_cosine
            right_scale[-1] = channel.damping_cosine
            left = (
                after_system
                * left_scale[np.newaxis, :, np.newaxis]
                * left_scale[np.newaxis, np.newaxis, :]
            )
            right = (
                after_system
                * right_scale[np.newaxis, :, np.newaxis]
                * right_scale[np.newaxis, np.newaxis, :]
            )
            density_matrix = 0.5 * (left + right)
            channel_index += 1
        saved[:, time_index] = _single_excitation_density_observables(
            density_matrix
        )[0]
    return saved


def exact_lindblad_observables(problem: ClassicalProblem) -> np.ndarray:
    """Evaluate the unsplit Lindblad reference at every saved time."""

    times = problem.schedule.times
    initial_vector = problem.initial_density_matrix.reshape(-1, order="F")
    generator_trace = problem.liouvillian.diagonal().sum()

    def density_at(time: float) -> np.ndarray:
        if np.isclose(time, 0.0):
            return problem.initial_density_matrix
        vector = expm_multiply(
            time * problem.liouvillian,
            initial_vector,
            traceA=time * generator_trace,
        )
        return vector.reshape(problem.initial_density_matrix.shape, order="F")

    density_matrices = np.asarray(tuple(map(density_at, times)))
    return _observable_batch(density_matrices, problem.observables).T


def observable_error_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(estimate) - np.asarray(reference)
    aggregate_by_time = np.sqrt(np.sum(difference**2, axis=0))
    return {
        "maximum_aggregate_error": float(np.max(aggregate_by_time)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def _metric_arrays(
    estimates: np.ndarray,
    reference: np.ndarray,
) -> dict[str, np.ndarray]:
    replicate_count, sample_count_count = estimates.shape[:2]
    values = {
        name: np.empty((replicate_count, sample_count_count), dtype=float)
        for name in METRIC_NAMES
    }
    for replicate in range(replicate_count):
        for sample_index in range(sample_count_count):
            metrics = observable_error_metrics(
                estimates[replicate, sample_index],
                reference,
            )
            for name in METRIC_NAMES:
                values[name][replicate, sample_index] = metrics[name]
    return values


def _summary_records(
    sample_counts: tuple[int, ...],
    metric_arrays: dict[str, np.ndarray],
    tolerance: float,
    confidence: float,
) -> tuple[dict[str, object], ...]:
    lower_quantile = 0.5 * (1.0 - confidence)
    upper_quantile = 1.0 - lower_quantile

    def metric_record(name: str, values: np.ndarray, index: int) -> dict[str, float]:
        column = values[:, index]
        return {
            "mean": float(np.mean(column)),
            "median": float(np.median(column)),
            "standard_deviation": float(np.std(column, ddof=1))
            if column.size > 1
            else 0.0,
            "central_interval_lower": float(np.quantile(column, lower_quantile)),
            "central_interval_upper": float(np.quantile(column, upper_quantile)),
            "confidence_upper_bound": float(np.quantile(column, confidence)),
        }

    return tuple(
        {
            "sample_count": sample_count,
            "metrics": {
                name: metric_record(name, metric_arrays[name], index)
                for name in METRIC_NAMES
            },
            "success_probability": float(
                np.mean(
                    metric_arrays["maximum_aggregate_error"][:, index]
                    <= tolerance
                )
            ),
            "passes": bool(
                np.quantile(
                    metric_arrays["maximum_aggregate_error"][:, index],
                    confidence,
                )
                <= tolerance
            ),
        }
        for index, sample_count in enumerate(sample_counts)
    )


def _recommended_counts(
    summaries: tuple[dict[str, object], ...],
) -> tuple[int | None, int | None]:
    passing = tuple(bool(record["passes"]) for record in summaries)
    first = next(
        (
            int(summaries[index]["sample_count"])
            for index, passes in enumerate(passing)
            if passes
        ),
        None,
    )
    stable = next(
        (
            int(summaries[index]["sample_count"])
            for index in range(len(summaries))
            if all(passing[index:])
        ),
        None,
    )
    return first, stable


def _nested_ensemble_estimates(
    problem: ClassicalProblem,
    counts: tuple[int, ...],
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Simulate all replicated nested ensembles, one target time at a time."""

    max_samples = counts[-1]
    observable_count = problem.observables.shape[0]
    estimates = np.empty(
        (
            replicates,
            len(counts),
            observable_count,
            problem.schedule.times.size,
        ),
        dtype=float,
    )
    generators = tuple(
        map(
            np.random.default_rng,
            np.random.SeedSequence(seed).spawn(replicates),
        )
    )
    count_indices = np.asarray(counts, dtype=int) - 1
    count_denominators = np.asarray(counts, dtype=float)[
        np.newaxis,
        :,
        np.newaxis,
    ]
    channel_index = 0
    for time_index, target_substeps in enumerate(
        problem.schedule.target_substep_dts
    ):
        amplitudes = np.broadcast_to(
            problem.initial_single_excitation_amplitudes,
            (replicates, max_samples, problem.number_of_qubits),
        ).copy()
        choices = np.asarray(
            tuple(
                generator.integers(
                    0,
                    2,
                    size=(max_samples, len(target_substeps)),
                    dtype=np.int8,
                )
                for generator in generators
            )
        )
        for local_index in range(len(target_substeps)):
            channel = problem.step_channels[channel_index]
            flattened = amplitudes.reshape(
                replicates * max_samples,
                problem.number_of_qubits,
            )
            flattened = flattened @ channel.system_unitary.T
            amplitudes = flattened.reshape(
                replicates,
                max_samples,
                problem.number_of_qubits,
            )
            use_right = choices[:, :, local_index].astype(bool)
            amplitudes[:, :, 0] *= np.where(
                use_right,
                1.0,
                channel.damping_cosine,
            )
            amplitudes[:, :, -1] *= np.where(
                use_right,
                channel.damping_cosine,
                1.0,
            )
            channel_index += 1
        trajectory_values = _single_excitation_amplitude_observables(
            amplitudes.reshape(
                replicates * max_samples,
                problem.number_of_qubits,
            )
        ).reshape(replicates, max_samples, observable_count)
        cumulative = np.cumsum(trajectory_values, axis=1)
        estimates[:, :, :, time_index] = (
            cumulative[:, count_indices, :] / count_denominators
        )
    return estimates


def run_complexity_study(
    problem: ClassicalProblem,
    sample_counts: tuple[int, ...],
    replicates: int,
    tolerance: float,
    confidence: float,
    seed: int,
) -> ComplexityStudy:
    """Run independent nested-ensemble replications for every candidate R."""

    counts = tuple(sorted(set(map(int, sample_counts))))
    if not counts or counts[0] < 1:
        raise ValueError("sample_counts must contain positive integers")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    exact = exact_lindblad_observables(problem)
    expected = expected_randomized_observables(problem)
    estimates = _nested_ensemble_estimates(
        problem,
        counts,
        replicates,
        seed,
    )

    errors_exact = _metric_arrays(estimates, exact)
    errors_expected = _metric_arrays(estimates, expected)
    summaries_exact = _summary_records(
        counts,
        errors_exact,
        tolerance,
        confidence,
    )
    summaries_expected = _summary_records(
        counts,
        errors_expected,
        tolerance,
        confidence,
    )
    first, recommended = _recommended_counts(summaries_exact)
    selected_count = recommended if recommended is not None else counts[-1]
    selected_index = counts.index(selected_count)
    selected_errors = errors_exact["maximum_aggregate_error"][:, selected_index]
    median_error = np.median(selected_errors)
    representative = int(np.argmin(np.abs(selected_errors - median_error)))
    return ComplexityStudy(
        sample_counts=counts,
        exact_observables=exact,
        expected_randomized_observables=expected,
        replicate_estimates=estimates,
        errors_vs_exact=errors_exact,
        errors_vs_expected_randomized=errors_expected,
        summaries_vs_exact=summaries_exact,
        summaries_vs_expected_randomized=summaries_expected,
        bias_metrics=observable_error_metrics(expected, exact),
        first_passing_sample_count=first,
        recommended_sample_count=recommended,
        representative_replicate=representative,
        representative_observables=estimates[representative, selected_index],
    )


def run_joint_complexity_study(
    number_of_qubits: int,
    times: np.ndarray,
    delta_ts: tuple[float, ...],
    sample_counts: tuple[int, ...],
    replicates: int,
    tolerance: float,
    confidence: float,
    seed: int,
) -> JointComplexityStudy:
    """Search physical-step and ensemble-size candidates jointly."""

    steps = tuple(sorted(set(map(float, delta_ts)), reverse=True))
    if not steps or any(map(lambda dt: not np.isfinite(dt) or dt <= 0.0, steps)):
        raise ValueError("delta_ts must contain finite positive values")
    counts = tuple(sorted(set(map(int, sample_counts))))
    problems = tuple(
        build_problem(number_of_qubits, times, delta_t) for delta_t in steps
    )
    studies = tuple(
        run_complexity_study(
            problem,
            counts,
            replicates,
            tolerance,
            confidence,
            seed + 104729 * delta_index,
        )
        for delta_index, problem in enumerate(problems)
    )

    records: list[dict[str, object]] = []
    for delta_index, (delta_t, problem, study) in enumerate(
        zip(steps, problems, studies, strict=True)
    ):
        bias = study.bias_metrics["maximum_aggregate_error"]
        bias_feasible = bias <= tolerance
        passing = tuple(bool(record["passes"]) for record in study.summaries_vs_exact)
        for sample_index, (sample_count, summary) in enumerate(
            zip(counts, study.summaries_vs_exact, strict=True)
        ):
            stable_pass = all(passing[sample_index:])
            substep_count = problem.schedule.final_time_substep_count
            records.append(
                {
                    "delta_t": delta_t,
                    "delta_t_index": delta_index,
                    "sample_count": sample_count,
                    "sample_count_index": sample_index,
                    "physical_substeps_per_trajectory": substep_count,
                    "total_sampled_substeps": sample_count * substep_count,
                    "validation_substeps_across_all_times": (
                        problem.schedule.validation_substep_count
                    ),
                    "finite_step_bias": bias,
                    "bias_feasible": bias_feasible,
                    "confidence_upper_error": summary["metrics"][
                        "maximum_aggregate_error"
                    ]["confidence_upper_bound"],
                    "median_error": summary["metrics"][
                        "maximum_aggregate_error"
                    ]["median"],
                    "success_probability": summary["success_probability"],
                    "passes_at_this_R": bool(summary["passes"]),
                    "passes_for_this_and_all_larger_R": stable_pass,
                    "feasible": bool(bias_feasible and stable_pass),
                }
            )

    feasible = tuple(filter(lambda record: bool(record["feasible"]), records))
    optimal = (
        min(
            feasible,
            key=lambda record: (
                -float(record["delta_t"]),
                int(record["sample_count"]),
                float(record["confidence_upper_error"]),
            ),
        )
        if feasible
        else None
    )
    return JointComplexityStudy(
        delta_ts=steps,
        problems=problems,
        studies=studies,
        pair_records=tuple(records),
        optimal_delta_t=None if optimal is None else float(optimal["delta_t"]),
        optimal_sample_count=(
            None if optimal is None else int(optimal["sample_count"])
        ),
        optimal_substep_count=(
            None
            if optimal is None
            else int(optimal["physical_substeps_per_trajectory"])
        ),
        optimal_total_sampled_substeps=(
            None if optimal is None else int(optimal["total_sampled_substeps"])
        ),
        optimal_delta_t_index=(
            None if optimal is None else int(optimal["delta_t_index"])
        ),
        optimal_sample_count_index=(
            None if optimal is None else int(optimal["sample_count_index"])
        ),
    )


def selected_problem_and_study(
    joint: JointComplexityStudy,
) -> tuple[ClassicalProblem, ComplexityStudy, int]:
    """Return the optimal pair, or the finest/largest fallback for plotting."""

    delta_index = (
        joint.optimal_delta_t_index
        if joint.optimal_delta_t_index is not None
        else len(joint.delta_ts) - 1
    )
    sample_index = (
        joint.optimal_sample_count_index
        if joint.optimal_sample_count_index is not None
        else len(joint.studies[delta_index].sample_counts) - 1
    )
    return joint.problems[delta_index], joint.studies[delta_index], sample_index


def representative_at_sample_index(
    study: ComplexityStudy,
    sample_index: int,
) -> tuple[int, np.ndarray]:
    errors = study.errors_vs_exact["maximum_aggregate_error"][:, sample_index]
    replicate = int(np.argmin(np.abs(errors - np.median(errors))))
    return replicate, study.replicate_estimates[replicate, sample_index]


def split_observable_families(
    values: np.ndarray,
    family_sizes: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    boundaries = np.cumsum((0, *family_sizes), dtype=int)
    return tuple(
        values[boundaries[index] : boundaries[index + 1]]
        for index in range(len(family_sizes))
    )


def plot_joint_convergence(
    joint: JointComplexityStudy,
    tolerance: float,
    confidence: float,
    output_path: Path,
) -> None:
    counts = np.asarray(joint.studies[0].sample_counts)
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for delta_t, study in zip(joint.delta_ts, joint.studies, strict=True):
        primary = study.errors_vs_exact["maximum_aggregate_error"]
        confidence_upper = np.quantile(primary, confidence, axis=0)
        axes[0].plot(
            counts,
            confidence_upper,
            marker="o",
            label=rf"$\Delta t={delta_t:g}$",
        )
    axes[0].axhline(tolerance, color="C3", linestyle="--", label="tolerance")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("trajectory count R")
    axes[0].set_ylabel(
        f"{100.0 * confidence:.1f}% quantile of maximum aggregate error"
    )
    axes[0].legend(fontsize=8)

    substeps = np.asarray(
        tuple(
            problem.schedule.final_time_substep_count
            for problem in joint.problems
        )
    )
    biases = np.asarray(
        tuple(
            study.bias_metrics["maximum_aggregate_error"]
            for study in joint.studies
        )
    )
    axes[1].plot(substeps, biases, marker="o")
    axes[1].axhline(tolerance, color="C3", linestyle="--", label="tolerance")
    for step_count, bias, delta_t in zip(substeps, biases, joint.delta_ts, strict=True):
        axes[1].annotate(
            rf"$\Delta t={delta_t:g}$",
            (step_count, bias),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    axes[1].set_xlabel("physical substeps per trajectory")
    axes[1].set_ylabel("finite-step bias vs exact")
    axes[1].legend(fontsize=8)

    for delta_index, delta_t in enumerate(joint.delta_ts):
        records = tuple(
            filter(
                lambda record: int(record["delta_t_index"]) == delta_index,
                joint.pair_records,
            )
        )
        axes[2].scatter(
            tuple(record["total_sampled_substeps"] for record in records),
            tuple(record["confidence_upper_error"] for record in records),
            label=rf"$\Delta t={delta_t:g}$",
        )
    axes[2].axhline(tolerance, color="C3", linestyle="--", label="tolerance")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel(r"total sampled substeps $R\,S(\Delta t)$")
    axes[2].set_ylabel(
        f"{100.0 * confidence:.1f}% quantile of maximum aggregate error"
    )
    if joint.optimal_total_sampled_substeps is not None:
        optimal_record = next(
            record
            for record in joint.pair_records
            if float(record["delta_t"]) == joint.optimal_delta_t
            and int(record["sample_count"]) == joint.optimal_sample_count
        )
        axes[2].scatter(
            (joint.optimal_total_sampled_substeps,),
            (optimal_record["confidence_upper_error"],),
            marker="*",
            s=220,
            color="gold",
            edgecolor="black",
            label="maximum-step/minimum-R pair",
            zorder=10,
        )
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)

    recommendation = (
        "none in candidate grid"
        if joint.optimal_delta_t is None
        else (
            rf"$\Delta t={joint.optimal_delta_t:g}$, "
            rf"$R={joint.optimal_sample_count}$, "
            rf"$R S={joint.optimal_total_sampled_substeps}$"
        )
    )
    figure.suptitle(
        "Experiment 6: joint physical-step and trajectory complexity\n"
        f"maximum-step/minimum-R pair: {recommendation}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_representative_observables(
    problem: ClassicalProblem,
    study: ComplexityStudy,
    sample_index: int,
    output_path: Path,
) -> None:
    _, representative_observables = representative_at_sample_index(
        study,
        sample_index,
    )
    exact_families = split_observable_families(
        study.exact_observables,
        problem.family_sizes,
    )
    expected_families = split_observable_families(
        study.expected_randomized_observables,
        problem.family_sizes,
    )
    estimate_families = split_observable_families(
        representative_observables,
        problem.family_sizes,
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ylabels = ("population", r"$\langle XX+YY\rangle$", "flow")
    for axis, family_name, ylabel, exact, expected, estimate in zip(
        axes,
        FAMILY_NAMES,
        ylabels,
        exact_families,
        expected_families,
        estimate_families,
        strict=True,
    ):
        for series_index in range(exact.shape[0]):
            color = f"C{series_index % 10}"
            axis.plot(
                problem.schedule.times,
                exact[series_index],
                color=color,
                label=f"{series_index}, exact",
            )
            axis.plot(
                problem.schedule.times,
                expected[series_index],
                color=color,
                linestyle=":",
                label=f"{series_index}, expected finite-step",
            )
            axis.plot(
                problem.schedule.times,
                estimate[series_index],
                color=color,
                linestyle="--",
                label=f"{series_index}, sampled",
            )
        axis.set_title(family_name)
        axis.set_xlabel("time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    selected = study.sample_counts[sample_index]
    figure.suptitle(
        "Representative median-error replication at "
        rf"$\Delta t={problem.schedule.trotter_delta_t:g}$, $R={selected}$"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _array_payload(values: np.ndarray) -> list[object]:
    return np.asarray(values).tolist()


def _family_payload(
    values: np.ndarray,
    family_sizes: tuple[int, ...],
) -> dict[str, list[object]]:
    return {
        name: _array_payload(family)
        for name, family in zip(
            FAMILY_NAMES,
            split_observable_families(values, family_sizes),
            strict=True,
        )
    }


def result_payload(
    joint: JointComplexityStudy,
    options: argparse.Namespace,
) -> dict[str, object]:
    problem, study, sample_index = selected_problem_and_study(joint)
    representative_index, representative_observables = (
        representative_at_sample_index(study, sample_index)
    )
    return {
        "experiment": 6,
        "method": "classical randomized single-boundary sample-complexity study",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_complexity_definition": (
            "total sampled substeps R*S(delta_t), with R complete trajectories "
            "and S(delta_t) physical substeps per trajectory"
        ),
        "optimality_definition": (
            "largest tested delta_t having a feasible pair, then smallest R "
            "whose finite-step bias is within tolerance and whose configured "
            "error quantile passes for that R and every larger tested R at "
            "the same delta_t"
        ),
        "number_of_system_qubits": problem.number_of_qubits,
        "initial_excited_sites": [problem.number_of_qubits // 2],
        "model": {"J": J, "h": h_field, "gamma": gamma},
        "times": _array_payload(problem.schedule.times),
        "trotter_delta_ts": list(joint.delta_ts),
        "sample_counts": list(study.sample_counts),
        "replicates": options.replicates,
        "error_tolerance": options.error_tolerance,
        "confidence": options.confidence,
        "seed": options.seed,
        "primary_error_metric": "maximum aggregate observable error over saved times",
        "optimal_pair": None
        if joint.optimal_delta_t is None
        else {
            "trotter_delta_t": joint.optimal_delta_t,
            "sample_count": joint.optimal_sample_count,
            "physical_substeps_per_trajectory": joint.optimal_substep_count,
            "total_sampled_substeps": joint.optimal_total_sampled_substeps,
        },
        "pair_records": list(joint.pair_records),
        "delta_t_studies": [
            {
                "trotter_delta_t": delta_t,
                "physical_substeps_to_final_time": (
                    candidate_problem.schedule.final_time_substep_count
                ),
                "validation_substeps_across_all_times": (
                    candidate_problem.schedule.validation_substep_count
                ),
                "target_substep_dts": [
                    list(target)
                    for target in candidate_problem.schedule.target_substep_dts
                ],
                "finite_step_bias_vs_exact": candidate_study.bias_metrics,
                "first_passing_sample_count": (
                    candidate_study.first_passing_sample_count
                ),
                "recommended_sample_count_at_fixed_delta_t": (
                    candidate_study.recommended_sample_count
                ),
                "summaries_vs_exact": list(candidate_study.summaries_vs_exact),
                "summaries_vs_expected_randomized_channel": list(
                    candidate_study.summaries_vs_expected_randomized
                ),
                "expected_finite_step_randomized_observables": _family_payload(
                    candidate_study.expected_randomized_observables,
                    candidate_problem.family_sizes,
                ),
            }
            for delta_t, candidate_problem, candidate_study in zip(
                joint.delta_ts,
                joint.problems,
                joint.studies,
                strict=True,
            )
        ],
        "exact_observables": _family_payload(
            study.exact_observables,
            problem.family_sizes,
        ),
        "representative": {
            "trotter_delta_t": problem.schedule.trotter_delta_t,
            "replicate_index": representative_index,
            "sample_count": study.sample_counts[sample_index],
            "observables": _family_payload(
                representative_observables,
                problem.family_sizes,
            ),
        },
        "notes": [
            "No shot noise, circuit noise, Aer sampling, or hardware execution is included.",
            "Increasing R reduces trajectory Monte Carlo error but not finite-dt bias.",
            "Decreasing delta_t lowers bias but increases substeps per trajectory.",
            "The maximum-step/minimum-R choice is empirical and restricted to both supplied candidate grids.",
        ],
    }


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def _output_paths(options: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = f"classical_sample_complexity_N{options.n_qubits}"
    root = options.output_directory
    return (
        root / "results" / f"{stem}.json",
        root / "figures" / f"{stem}_convergence.png",
        root / "figures" / f"{stem}_observables.png",
    )


def _print_summary(
    joint: JointComplexityStudy,
    options: argparse.Namespace,
) -> None:
    problem = joint.problems[0]
    print("Experiment 6: classical randomized-boundary convergence")
    print(f"System qubits: {problem.number_of_qubits}")
    print(f"Independent replications: {options.replicates}")
    print()
    print("delta_t\tR\tsteps\tR*steps\tbias\tconf. upper\tfeasible")
    for record in joint.pair_records:
        print(
            f"{record['delta_t']:g}\t"
            f"{record['sample_count']}\t"
            f"{record['physical_substeps_per_trajectory']}\t"
            f"{record['total_sampled_substeps']}\t"
            f"{record['finite_step_bias']:.7g}\t"
            f"{record['confidence_upper_error']:.7g}\t"
            f"{record['feasible']}"
        )
    print()
    if joint.optimal_delta_t is None:
        print("No tested (delta_t, R) pair satisfies the requested criterion.")
    else:
        print(
            "Maximum-step/minimum-R pair: "
            f"delta_t={joint.optimal_delta_t:g}, "
            f"R={joint.optimal_sample_count}, "
            f"steps={joint.optimal_substep_count}, "
            f"R*steps={joint.optimal_total_sampled_substeps}"
        )


def main(options: argparse.Namespace) -> None:
    times = (
        np.asarray(options.times, dtype=float)
        if options.times is not None
        else np.linspace(0.0, options.t_final, options.time_points)
    )
    joint = run_joint_complexity_study(
        options.n_qubits,
        times,
        tuple(options.trotter_delta_ts),
        tuple(options.sample_counts),
        options.replicates,
        options.error_tolerance,
        options.confidence,
        options.seed,
    )
    problem, study, sample_index = selected_problem_and_study(joint)
    result_path, convergence_path, observables_path = _output_paths(options)
    save_json(result_payload(joint, options), result_path)
    if not options.no_plots:
        plot_joint_convergence(
            joint,
            options.error_tolerance,
            options.confidence,
            convergence_path,
        )
        plot_representative_observables(
            problem,
            study,
            sample_index,
            observables_path,
        )
    _print_summary(joint, options)
    print(f"Saved data: {result_path}")
    if not options.no_plots:
        print(f"Saved convergence figure: {convergence_path}")
        print(f"Saved observable figure: {observables_path}")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classically determine the Experiment 5 randomized-boundary "
            "trajectory and physical-step complexity relative to exact "
            "Lindblad evolution."
        )
    )
    parser.add_argument("--n-qubits", type=at_least_two, default=DEFAULT_NUMBER_OF_QUBITS)
    parser.add_argument(
        "--sample-counts",
        nargs="+",
        type=positive_integer,
        default=DEFAULT_SAMPLE_COUNTS,
        metavar="R",
        help="candidate numbers of complete trajectories",
    )
    parser.add_argument(
        "--replicates",
        type=positive_integer,
        default=DEFAULT_REPLICATES,
        help="independent repetitions used to estimate the error distribution",
    )
    parser.add_argument(
        "--error-tolerance",
        type=positive_float,
        default=DEFAULT_ERROR_TOLERANCE,
        help="maximum aggregate observable-error tolerance",
    )
    parser.add_argument(
        "--confidence",
        type=probability,
        default=DEFAULT_CONFIDENCE,
        help="error quantile that must lie below the tolerance",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=None,
        help="explicit increasing saved times beginning at zero",
    )
    parser.add_argument("--t-final", type=positive_float, default=DEFAULT_FINAL_TIME)
    parser.add_argument("--time-points", type=positive_integer, default=DEFAULT_TIME_POINTS)
    parser.add_argument(
        "--trotter-delta-ts",
        "--trotter-delta-t",
        dest="trotter_delta_ts",
        nargs="+",
        type=positive_float,
        default=DEFAULT_TROTTER_DELTA_TS,
        metavar="DT",
        help="candidate maximum physical substeps used by Experiment 5",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main(parse_arguments())
