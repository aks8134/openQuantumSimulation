"""Experiment 6, Strategy 3: randomized odd/JL and even/JR branches.

Write the generator as

    L = (L_Hodd + L_JL) + (L_Heven + L_JR).

At every nominal substep ``dt``, select one composite branch uniformly and
enhance its generator by two.  Within a selected branch, apply its commuting
Hamiltonian bond layer first and its boundary jump second.  ``R`` is the
number of complete independently randomized branch trajectories averaged for
an observable estimate.

This script is classical only.  It constructs no circuits and submits no jobs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Model1.Experiment6 import classical_sample_complexity as base


NUMBER_OF_BRANCHES = 2
ODD_LEFT_BRANCH = 0
EVEN_RIGHT_BRANCH = 1
BRANCH_NAMES = ("odd_hamiltonian_plus_left_jump", "even_hamiltonian_plus_right_jump")
BRANCH_PROBABILITIES = (0.5, 0.5)


@dataclass(frozen=True, slots=True)
class Strategy3StepChannel:
    """Enhanced maps associated with one nominal Strategy 3 substep."""

    dt: float
    odd_unitary: np.ndarray
    even_unitary: np.ndarray
    damping_cosine: float


def _layer_unitary(
    number_of_qubits: int,
    bonds: tuple[int, ...],
    evolution_time: float,
) -> np.ndarray:
    hamiltonian = base._single_excitation_layer_hamiltonian(
        number_of_qubits,
        bonds,
    )
    return base.expm(-1j * evolution_time * hamiltonian)


def build_problem(
    number_of_qubits: int,
    times: np.ndarray,
    trotter_delta_t: float | None,
) -> base.ClassicalProblem:
    """Build the exact model and enhanced odd/JL and even/JR maps."""

    problem = base.build_problem(number_of_qubits, times, trotter_delta_t)
    odd_bonds = tuple(range(1, number_of_qubits - 1, 2))
    even_bonds = tuple(range(0, number_of_qubits - 1, 2))
    cache: dict[float, Strategy3StepChannel] = {}

    def channel_for(dt: float) -> Strategy3StepChannel:
        key = round(float(dt), 15)
        if key not in cache:
            enhanced_dt = NUMBER_OF_BRANCHES * dt
            cache[key] = Strategy3StepChannel(
                dt=float(dt),
                odd_unitary=_layer_unitary(
                    number_of_qubits,
                    odd_bonds,
                    enhanced_dt,
                ),
                even_unitary=_layer_unitary(
                    number_of_qubits,
                    even_bonds,
                    enhanced_dt,
                ),
                damping_cosine=float(
                    np.cos(
                        np.sqrt(
                            NUMBER_OF_BRANCHES * base.gamma * dt
                        )
                    )
                ),
            )
        return cache[key]

    return replace(
        problem,
        step_channels=tuple(
            channel_for(channel.dt) for channel in problem.step_channels
        ),
    )


def _apply_selected_branches(
    amplitudes: np.ndarray,
    choices: np.ndarray,
    channel: Strategy3StepChannel,
) -> None:
    use_odd_left = choices == ODD_LEFT_BRANCH
    use_even_right = choices == EVEN_RIGHT_BRANCH
    amplitudes[use_odd_left] = (
        amplitudes[use_odd_left] @ channel.odd_unitary.T
    )
    amplitudes[use_odd_left, 0] *= channel.damping_cosine
    amplitudes[use_even_right] = (
        amplitudes[use_even_right] @ channel.even_unitary.T
    )
    amplitudes[use_even_right, -1] *= channel.damping_cosine


def simulate_branch_choices(
    problem: base.ClassicalProblem,
    branch_choices: np.ndarray,
) -> np.ndarray:
    """Return observables for prescribed odd/JL and even/JR trajectories."""

    choices = np.asarray(branch_choices, dtype=int)
    if choices.ndim != 2:
        raise ValueError("branch_choices must be a two-dimensional array")
    if choices.shape[1] != len(problem.step_channels):
        raise ValueError("branch_choices has the wrong physical-substep count")
    if np.any((choices < 0) | (choices >= NUMBER_OF_BRANCHES)):
        raise ValueError("branch choices must be 0 (odd/JL) or 1 (even/JR)")

    trajectory_count = choices.shape[0]
    saved = np.empty(
        (
            trajectory_count,
            problem.observables.shape[0],
            problem.schedule.times.size,
        ),
        dtype=float,
    )
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
            _apply_selected_branches(
                amplitudes,
                choices[:, channel_index],
                channel,
            )
            channel_index += 1
        saved[:, :, time_index] = base._single_excitation_amplitude_observables(
            amplitudes
        )
    return saved


def expected_randomized_observables(
    problem: base.ClassicalProblem,
) -> np.ndarray:
    """Propagate the exact finite-step mean of the two composite branches."""

    initial = problem.initial_single_excitation_amplitudes
    initial_density = np.outer(initial, initial.conj())[np.newaxis, :, :]
    saved = np.empty(
        (problem.observables.shape[0], problem.schedule.times.size),
        dtype=float,
    )
    channel_index = 0
    for time_index, target_substeps in enumerate(
        problem.schedule.target_substep_dts
    ):
        density_matrix = initial_density.copy()
        for _ in target_substeps:
            channel = problem.step_channels[channel_index]
            odd_branch = base._unitary_batch(
                density_matrix,
                channel.odd_unitary,
            )
            even_branch = base._unitary_batch(
                density_matrix,
                channel.even_unitary,
            )
            left_scale = np.ones(problem.number_of_qubits)
            right_scale = np.ones(problem.number_of_qubits)
            left_scale[0] = channel.damping_cosine
            right_scale[-1] = channel.damping_cosine
            odd_left_branch = (
                odd_branch
                * left_scale[np.newaxis, :, np.newaxis]
                * left_scale[np.newaxis, np.newaxis, :]
            )
            even_right_branch = (
                even_branch
                * right_scale[np.newaxis, :, np.newaxis]
                * right_scale[np.newaxis, np.newaxis, :]
            )
            density_matrix = 0.5 * (
                odd_left_branch + even_right_branch
            )
            channel_index += 1
        saved[:, time_index] = base._single_excitation_density_observables(
            density_matrix
        )[0]
    return saved


def _nested_ensemble_estimates(
    problem: base.ClassicalProblem,
    counts: tuple[int, ...],
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Simulate replicated nested ensembles of Strategy 3 trajectories."""

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
                    NUMBER_OF_BRANCHES,
                    size=(max_samples, len(target_substeps)),
                    dtype=np.int8,
                )
                for generator in generators
            )
        )
        for local_index in range(len(target_substeps)):
            flattened = amplitudes.reshape(
                replicates * max_samples,
                problem.number_of_qubits,
            )
            _apply_selected_branches(
                flattened,
                choices[:, :, local_index].reshape(-1),
                problem.step_channels[channel_index],
            )
            channel_index += 1
        trajectory_values = base._single_excitation_amplitude_observables(
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
    problem: base.ClassicalProblem,
    sample_counts: tuple[int, ...],
    replicates: int,
    tolerance: float,
    confidence: float,
    seed: int,
) -> base.ComplexityStudy:
    """Run the Strategy 3 convergence study at a fixed step."""

    counts = tuple(sorted(set(map(int, sample_counts))))
    if not counts or counts[0] < 1:
        raise ValueError("sample_counts must contain positive integers")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    exact = base.exact_lindblad_observables(problem)
    expected = expected_randomized_observables(problem)
    estimates = _nested_ensemble_estimates(problem, counts, replicates, seed)
    errors_exact = base._metric_arrays(estimates, exact)
    errors_expected = base._metric_arrays(estimates, expected)
    summaries_exact = base._summary_records(
        counts,
        errors_exact,
        tolerance,
        confidence,
    )
    summaries_expected = base._summary_records(
        counts,
        errors_expected,
        tolerance,
        confidence,
    )
    first, recommended = base._recommended_counts(summaries_exact)
    selected_count = recommended if recommended is not None else counts[-1]
    selected_index = counts.index(selected_count)
    selected_errors = errors_exact["maximum_aggregate_error"][:, selected_index]
    median_error = np.median(selected_errors)
    representative = int(np.argmin(np.abs(selected_errors - median_error)))
    return base.ComplexityStudy(
        sample_counts=counts,
        exact_observables=exact,
        expected_randomized_observables=expected,
        replicate_estimates=estimates,
        errors_vs_exact=errors_exact,
        errors_vs_expected_randomized=errors_expected,
        summaries_vs_exact=summaries_exact,
        summaries_vs_expected_randomized=summaries_expected,
        bias_metrics=base.observable_error_metrics(expected, exact),
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
) -> base.JointComplexityStudy:
    """Search Strategy 3 step and ensemble-size candidates jointly."""

    steps = tuple(sorted(set(map(float, delta_ts)), reverse=True))
    if not steps or any(
        not np.isfinite(dt) or dt <= 0.0 for dt in steps
    ):
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
        passing = tuple(
            bool(record["passes"]) for record in study.summaries_vs_exact
        )
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

    feasible = tuple(record for record in records if bool(record["feasible"]))
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
    return base.JointComplexityStudy(
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


def result_payload(
    joint: base.JointComplexityStudy,
    options: argparse.Namespace,
) -> dict[str, object]:
    payload = base.result_payload(joint, options)
    payload["method"] = "uniform randomized odd/JL and even/JR branch splitting"
    payload["strategy"] = 3
    payload["randomized_branches"] = {
        "names": list(BRANCH_NAMES),
        "probabilities": list(BRANCH_PROBABILITIES),
        "generator_enhancement": NUMBER_OF_BRANCHES,
        "internal_order": "Hamiltonian bond layer, then boundary jump",
        "sampled_maps": [
            "exp(2*dt*L_JL) exp(2*dt*L_Hodd)",
            "exp(2*dt*L_JR) exp(2*dt*L_Heven)",
        ],
    }
    payload["notes"] = [
        "Odd bonds use zero-based starts 1,3,...; even bonds use starts 0,2,... .",
        "Each nominal substep samples odd/JL or even/JR uniformly.",
        "R counts complete independently randomized branch trajectories.",
        *payload["notes"],
    ]
    return payload


def _output_paths(options: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = f"strategy3_classical_sample_complexity_N{options.n_qubits}"
    return (
        options.output_directory / "results" / f"{stem}.json",
        options.output_directory / "figures" / f"{stem}_convergence.png",
        options.output_directory / "figures" / f"{stem}_observables.png",
    )


def _print_summary(
    joint: base.JointComplexityStudy,
    options: argparse.Namespace,
) -> None:
    print("Experiment 6 Strategy 3: randomized odd/JL and even/JR convergence")
    print(f"System qubits: {joint.problems[0].number_of_qubits}")
    print(f"Independent replications: {options.replicates}")
    print("Sampling probabilities: odd/JL=1/2, even/JR=1/2")
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
    problem, study, sample_index = base.selected_problem_and_study(joint)
    result_path, convergence_path, observables_path = _output_paths(options)
    base.save_json(result_payload(joint, options), result_path)
    if not options.no_plots:
        base.plot_joint_convergence(
            joint,
            options.error_tolerance,
            options.confidence,
            convergence_path,
        )
        base.plot_representative_observables(
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
            "Classically tune randomized odd/JL and even/JR Strategy 3 "
            "against exact Lindblad evolution."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=base.at_least_two,
        default=base.DEFAULT_NUMBER_OF_QUBITS,
    )
    parser.add_argument(
        "--sample-counts",
        nargs="+",
        type=base.positive_integer,
        default=base.DEFAULT_SAMPLE_COUNTS,
        metavar="R",
    )
    parser.add_argument(
        "--replicates",
        type=base.positive_integer,
        default=base.DEFAULT_REPLICATES,
    )
    parser.add_argument(
        "--error-tolerance",
        type=base.positive_float,
        default=base.DEFAULT_ERROR_TOLERANCE,
    )
    parser.add_argument(
        "--confidence",
        type=base.probability,
        default=base.DEFAULT_CONFIDENCE,
    )
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--times", nargs="+", type=float, default=None)
    parser.add_argument(
        "--t-final",
        type=base.positive_float,
        default=base.DEFAULT_FINAL_TIME,
    )
    parser.add_argument(
        "--time-points",
        type=base.positive_integer,
        default=base.DEFAULT_TIME_POINTS,
    )
    parser.add_argument(
        "--trotter-delta-ts",
        "--trotter-delta-t",
        dest="trotter_delta_ts",
        nargs="+",
        type=base.positive_float,
        default=base.DEFAULT_TROTTER_DELTA_TS,
        metavar="DT",
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
