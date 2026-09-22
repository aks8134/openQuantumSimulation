"""Recover completed Experiment 4 Sampler jobs without submitting QPU work.

The selected jobs must be the consecutive jobs from one dynamic Lie run,
ordered from oldest to newest.  Results are written to the same checkpoint
format used by ``dynamic_lie_trotter.py --resume``.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from IBMRuntime import Err, Ok, SampleResult, load_ibm_account
from Model1.Experiment4 import dynamic_lie_trotter as experiment


COMPLETED_STATUSES = ("COMPLETED", "DONE")


def iso_datetime(value):
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use ISO 8601 format, for example "
            "2026-09-21T18:30:00-04:00"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            "the timestamp must include a UTC offset or trailing Z"
        )
    return parsed


def _status_name(job):
    status = job.status()
    name = getattr(status, "name", None)
    if name is not None:
        return str(name).upper()
    return str(status).rsplit(".", maxsplit=1)[-1].upper()


def _creation_date(job):
    value = job.creation_date
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _job_record(job):
    return {
        "job_id": str(job.job_id()),
        "status": _status_name(job),
        "created": _creation_date(job),
        "primitive_id": str(job.primitive_id),
    }


def _load_service(account_file):
    match load_ibm_account(account_file):
        case Err(error):
            raise RuntimeError(
                f"Could not load IBM account from {account_file}: "
                f"{error.message}"
            )
        case Ok(account):
            from qiskit_ibm_runtime import QiskitRuntimeService

            return QiskitRuntimeService(
                channel=account.channel,
                token=account.api_key,
                instance=account.instance,
            )


def find_jobs(service, options):
    """Return explicitly ordered jobs or discover them chronologically."""
    if options.job_ids:
        return tuple(service.job(job_id) for job_id in options.job_ids)
    if options.created_after is None:
        raise ValueError(
            "provide at least one --job-id, or provide --created-after "
            "to bound automatic discovery"
        )
    return tuple(
        service.jobs(
            limit=options.limit,
            backend_name=options.backend,
            created_after=options.created_after,
            created_before=options.created_before,
            descending=False,
        )
    )


def recover_completed_results(jobs, backend_name):
    """Download results only from completed Sampler jobs."""
    recovered = ()
    records = ()
    for job in jobs:
        record = _job_record(job)
        records = (*records, record)
        print(
            f"{record['created']}  {record['job_id']}  "
            f"{record['status']}  {record['primitive_id']}"
        )
        if "sampler" not in record["primitive_id"].lower():
            print("  skipped: not a Sampler job")
            continue
        if record["status"] not in COMPLETED_STATUSES:
            print("  skipped: the job is not completed")
            continue

        provider_results = tuple(job.result())
        job_id = record["job_id"]
        for provider_result in provider_results:
            counts = provider_result.join_data().get_counts()
            normalized_counts = tuple(
                sorted(
                    (str(bitstring), int(count))
                    for bitstring, count in counts.items()
                )
            )
            shots = sum(count for _, count in normalized_counts)
            if shots <= 0:
                raise ValueError(f"job {job_id} returned empty counts")
            recovered = (
                *recovered,
                SampleResult(
                    counts=normalized_counts,
                    shots=shots,
                    backend_name=backend_name,
                    job_id=job_id,
                    # These values are not preserved in a retrieved
                    # PrimitiveResult.  -1 explicitly marks unavailable
                    # compilation metadata instead of inventing values.
                    original_gate_count=-1,
                    original_depth=-1,
                    compiled_gate_count=-1,
                    compiled_depth=-1,
                ),
            )
    return recovered, records


def _recovery_output_paths(options, complete_time_count):
    directory = (
        Path(__file__).resolve().parent
        if options.output_directory is None
        else options.output_directory
    )
    stem = (
        f"recovered_dynamic_lie_trotter_"
        f"{experiment._safe_name(options.backend)}_"
        f"{options.n_qubits}_{complete_time_count}_times"
    )
    return (
        directory / "figures" / f"{stem}.png",
        directory / "results" / f"{stem}.json",
    )


def save_recovery(options, recovered_results, job_records):
    times = experiment.time_grid_from_options(options)
    (
        _,
        metadata,
        schedule,
    ) = experiment.build_sample_circuits(
        times,
        options.n_qubits,
        trotter_delta_t=options.trotter_delta_t,
    )
    if len(recovered_results) > len(metadata):
        raise ValueError(
            "the selected jobs contain more circuit results than this run; "
            "narrow the time window or pass explicit --job-id values"
        )

    shot_counts = {result.shots for result in recovered_results}
    if len(shot_counts) != 1:
        raise ValueError(
            "selected jobs have different shot counts and are not one run"
        )
    if recovered_results:
        options.shots = next(iter(shot_counts))

    _, normal_result_path = experiment.output_paths(
        options.backend,
        options.n_qubits,
        options.output_directory,
    )
    checkpoint_path = (
        experiment.default_checkpoint_path(normal_result_path)
        if options.checkpoint_file is None
        else options.checkpoint_file
    )
    checkpoint_status = (
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
        status=checkpoint_status,
    )
    print(
        f"Recovered checkpoint: {checkpoint_path} "
        f"({len(recovered_results)}/{len(metadata)} circuits)"
    )

    circuits_per_time = len(experiment.MEASUREMENT_BASES)
    complete_time_count = len(recovered_results) // circuits_per_time
    complete_circuit_count = complete_time_count * circuits_per_time
    if complete_time_count == 0:
        print(
            "No complete five-basis time point is available for plotting; "
            "the individual circuit results are safe in the checkpoint."
        )
        return checkpoint_path, None, None

    usable_results = recovered_results[:complete_circuit_count]
    usable_metadata = metadata[:complete_circuit_count]
    usable_times = times[:complete_time_count]
    measured, standard_errors, grouped_counts = (
        experiment.observables_from_results(
            usable_results,
            usable_metadata,
            options.n_qubits,
            complete_time_count,
        )
    )
    exact = coherent_lie = None
    if options.classical_reference:
        exact, coherent_lie = experiment.calculate_references(
            usable_times,
            options.n_qubits,
            schedule.prefix(complete_time_count),
        )

    figure_path, result_path = _recovery_output_paths(
        options,
        complete_time_count,
    )
    experiment.plot_results(
        usable_times,
        f"{options.backend} (recovered)",
        measured,
        standard_errors,
        figure_path,
        exact,
        coherent_lie,
    )
    experiment.save_results(
        result_path,
        options,
        usable_times,
        schedule.prefix(complete_time_count),
        usable_results,
        usable_metadata,
        measured,
        standard_errors,
        grouped_counts,
        extra_payload={
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
    service = _load_service(options.account_file)
    jobs = find_jobs(service, options)
    if not jobs:
        raise RuntimeError("no jobs matched the supplied filters")

    print(f"Found {len(jobs)} job(s), ordered oldest to newest")
    if options.list_only:
        for job in jobs:
            record = _job_record(job)
            print(
                f"{record['created']}  {record['job_id']}  "
                f"{record['status']}  {record['primitive_id']}"
            )
        print("List-only mode: no result data was downloaded or written.")
        return

    recovered_results, job_records = recover_completed_results(
        jobs,
        options.backend,
    )
    if not recovered_results:
        raise RuntimeError("none of the selected jobs has recoverable results")
    save_recovery(options, recovered_results, job_records)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recover completed Experiment 4 IBM Sampler jobs. This script "
            "never submits or cancels hardware work."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=experiment.at_least_two,
        required=True,
        help="number of system qubits used by the original run",
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="IBM backend used by the original run",
    )
    parser.add_argument(
        "--job-id",
        dest="job_ids",
        action="append",
        default=[],
        help=(
            "job ID in original submission order; repeat for every job "
            "(safest recovery mode)"
        ),
    )
    parser.add_argument(
        "--created-after",
        type=iso_datetime,
        default=None,
        help=(
            "discover jobs after this ISO timestamp when --job-id is not "
            "used, for example 2026-09-21T18:30:00-04:00"
        ),
    )
    parser.add_argument(
        "--created-before",
        type=iso_datetime,
        default=None,
        help="optional exclusive upper bound for job discovery",
    )
    parser.add_argument(
        "--limit",
        type=experiment.positive_integer,
        default=None,
        help="maximum jobs returned by automatic discovery",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print matching full job IDs and statuses without recovery",
    )
    parser.add_argument(
        "--account-file",
        type=Path,
        default=experiment.DEFAULT_ACCOUNT_FILE,
    )
    parser.add_argument(
        "--t-final",
        type=experiment.positive_float,
        default=experiment.dynamic_circuits.T_FINAL,
    )
    parser.add_argument(
        "--time-points",
        type=experiment.at_least_two,
        default=experiment.dynamic_circuits.NUMBER_OF_TIME_POINTS,
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "explicit saved times from the original run; accepts space- "
            "or comma-separated values and overrides --t-final and "
            "--time-points"
        ),
    )
    parser.add_argument(
        "--trotter-delta-t",
        type=experiment.positive_float,
        default=None,
        help=(
            "maximum internal Trotter substep used by the original run; "
            "omit for runs made before this option existed"
        ),
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
        help="retained only for checkpoint compatibility",
    )
    parser.add_argument(
        "--aer-method",
        default="automatic",
        help="retained only for checkpoint compatibility",
    )
    parser.add_argument(
        "--classical-reference",
        action="store_true",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main(parse_arguments())
