"""Recover completed Experiment 5 Sampler jobs without QPU submission."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Model1.Experiment4 import recover_dynamic_lie_trotter as recovery_tools
from Model1.Experiment5 import randomized_lie_trotter as experiment


def _recovery_output_paths(options, complete_time_count):
    directory = (
        Path(__file__).resolve().parent
        if options.output_directory is None
        else options.output_directory
    )
    stem = (
        f"recovered_randomized_lie_trotter_"
        f"{experiment._safe_name(options.backend)}_"
        f"N{options.n_qubits}_R{options.trajectories}_"
        f"{complete_time_count}_times"
    )
    return (
        directory / "figures" / f"{stem}.png",
        directory / "results" / f"{stem}.json",
    )


def save_recovery(options, recovered_results, job_records):
    times = experiment.hardware_tools.time_grid_from_options(options)
    circuits, metadata, schedule = experiment.build_sample_circuits(
        times,
        options.n_qubits,
        options.trajectories,
        options.seed_trajectories,
        options.trotter_delta_t,
    )
    if len(recovered_results) > len(metadata):
        raise ValueError(
            "selected jobs contain more circuit results than this run; "
            "narrow the time window or pass explicit --job-id values"
        )
    shot_counts = {result.shots for result in recovered_results}
    if len(shot_counts) != 1:
        raise ValueError("selected jobs have inconsistent shot counts")
    per_trajectory_shots = next(iter(shot_counts))
    options.shots = per_trajectory_shots * options.trajectories

    normal_paths = experiment.output_paths(
        options.backend,
        options.n_qubits,
        options.trajectories,
        options.output_directory,
    )
    checkpoint_path = (
        experiment.default_checkpoint_path(normal_paths["result"])
        if options.checkpoint_file is None
        else options.checkpoint_file
    )
    status = (
        "complete"
        if len(recovered_results) == len(metadata)
        else "partial"
    )
    experiment.save_checkpoint(
        checkpoint_path,
        options,
        times,
        schedule,
        recovered_results,
        metadata,
        status=status,
    )
    print(
        f"Recovered checkpoint: {checkpoint_path} "
        f"({len(recovered_results)}/{len(metadata)} circuits)"
    )

    circuits_per_time = options.trajectories * len(
        experiment.MEASUREMENT_BASES
    )
    complete_time_count = len(recovered_results) // circuits_per_time
    complete_circuit_count = complete_time_count * circuits_per_time
    if complete_time_count == 0:
        print(
            "No complete time point is available for plotting; individual "
            "results are safe in the checkpoint."
        )
        return checkpoint_path, None, None

    usable_results = recovered_results[:complete_circuit_count]
    usable_metadata = metadata[:complete_circuit_count]
    usable_times = times[:complete_time_count]
    usable_schedule = schedule.prefix(complete_time_count)
    (
        measured,
        uncertainties,
        trajectory_observables,
        grouped_counts,
    ) = experiment.observables_from_results(
        usable_results,
        usable_metadata,
        options.n_qubits,
        complete_time_count,
        options.trajectories,
    )
    exact = None
    if options.classical_reference:
        exact = experiment.calculate_exact_reference(
            usable_times,
            options.n_qubits,
        )
    figure_path, result_path = _recovery_output_paths(
        options,
        complete_time_count,
    )
    experiment.plot_results(
        usable_times,
        f"{options.backend} (recovered)",
        options.trajectories,
        measured,
        uncertainties,
        figure_path,
        exact,
    )
    experiment.save_results(
        result_path,
        options,
        usable_times,
        usable_schedule,
        usable_results,
        usable_metadata,
        measured,
        uncertainties,
        trajectory_observables,
        grouped_counts,
        metrics_summary=(),
        extra_payload={
            **(
                {}
                if exact is None
                else {
                    "exact_reference_observables": (
                        experiment._array_families_payload(exact)
                    ),
                    "aggregate_observable_error_vs_exact": (
                        experiment.chain_tools.aggregate_observable_error(
                            measured,
                            exact,
                        ).tolist()
                    ),
                }
            ),
            "recovery": {
                "selected_jobs": job_records,
                "recovered_circuit_count": len(recovered_results),
                "complete_time_point_count": complete_time_count,
                "compilation_metrics_available": False,
            }
        },
    )
    print(f"Saved recovered figure: {figure_path}")
    print(f"Saved recovered data: {result_path}")
    return checkpoint_path, figure_path, result_path


def main(options):
    service = recovery_tools._load_service(options.account_file)
    jobs = recovery_tools.find_jobs(service, options)
    if not jobs:
        raise RuntimeError("no jobs matched the supplied filters")
    print(f"Found {len(jobs)} job(s), ordered oldest to newest")
    if options.list_only:
        for job in jobs:
            record = recovery_tools._job_record(job)
            print(
                f"{record['created']}  {record['job_id']}  "
                f"{record['status']}  {record['primitive_id']}"
            )
        print("List-only mode: no result data was downloaded or written.")
        return
    recovered_results, job_records = (
        recovery_tools.recover_completed_results(jobs, options.backend)
    )
    if not recovered_results:
        raise RuntimeError("none of the selected jobs has recoverable results")
    save_recovery(options, recovered_results, job_records)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recover completed Experiment 5 IBM Sampler jobs. This script "
            "never submits or cancels QPU work."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=experiment.at_least_two,
        required=True,
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument(
        "--trajectories",
        "-R",
        type=experiment.positive_integer,
        required=True,
    )
    parser.add_argument(
        "--seed-trajectories",
        type=int,
        default=experiment.DEFAULT_SEED_TRAJECTORIES,
    )
    parser.add_argument(
        "--job-id",
        dest="job_ids",
        action="append",
        default=[],
        help="job ID in original order; repeat for every job",
    )
    parser.add_argument(
        "--created-after",
        type=recovery_tools.iso_datetime,
        default=None,
    )
    parser.add_argument(
        "--created-before",
        type=recovery_tools.iso_datetime,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=experiment.positive_integer,
        default=None,
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--account-file",
        type=Path,
        default=experiment.DEFAULT_ACCOUNT_FILE,
    )
    parser.add_argument(
        "--t-final",
        type=experiment.positive_float,
        default=experiment.T_FINAL,
    )
    parser.add_argument(
        "--time-points",
        type=experiment.at_least_two,
        default=experiment.NUMBER_OF_TIME_POINTS,
    )
    parser.add_argument("--times", nargs="+", default=None, metavar="T")
    parser.add_argument(
        "--trotter-delta-t",
        type=experiment.positive_float,
        default=None,
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=experiment.DEFAULT_OPTIMIZATION_LEVEL,
    )
    parser.add_argument(
        "--seed-transpiler",
        type=int,
        default=experiment.DEFAULT_SEED_TRANSPILER,
    )
    parser.add_argument(
        "--seed-simulator",
        type=int,
        default=experiment.DEFAULT_SEED_SIMULATOR,
    )
    parser.add_argument("--aer-method", default="automatic")
    parser.add_argument("--classical-reference", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main(parse_arguments())
