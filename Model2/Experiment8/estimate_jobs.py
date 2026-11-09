"""Count backend-specific mitigation+DD footprints and minimize planned jobs.

No Sampler, QPU execution, simulation, hardware manifest or QPY is produced.
Default: count every requested path. Explicit --sample-trajectories enables a
labelled projection; a projection never certifies a full-cohort minimum.
Rerunning the same command resumes hashed per-path numeric checkpoints using
the frozen cached target, without fetching a different calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, instruction_audit as audit, qpy_storage
from Model2.Experiment6 import estimate_jobs as packing
from Model2.Experiment6.progress import progress
from Model2.Experiment8 import run_hardware as run
from Model3.Experiment5 import mitigation

TYPE = "model2_mitigation_dd_job_estimate_v1"
_WORK = None


def seal(value):
    return dict(value, sha256=input_data.digest({k:v for k,v in value.items() if k != "sha256"}))


def verify_seal(value):
    if value.get("sha256") != seal(value)["sha256"]:
        raise ValueError("numeric checkpoint hash mismatch")
    return value


def versions():
    return {name:importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime", "numpy")}


def code_hashes():
    files = [Path(__file__), Path(run.__file__), Path(run.timing.__file__),
             Path(run.flat_block_order.__code__.co_filename), Path(run.protocol.__file__),
             Path(mitigation.__file__), Path(run.raw.__file__), Path(run.raw.circuits.__file__),
             Path(audit.__file__), Path(input_data.__file__)]
    return {p.relative_to(ROOT).as_posix():qpy_storage.file_sha256(p) for p in files}


def compilation_source(count):
    """Exact frozen seed streams; covariance propagation is unnecessary to count gates."""
    spec = input_data.load_spec()
    probabilities = [.125, .375, .125, .375]
    draws = input_data.choices(29, count, 44, probabilities)
    return dict(artifact_type="model2_instruction_count_input_v1",
        configuration=dict(physics=spec["physics"], trajectories=count, seed=29, **spec["channel"]),
        preparation=spec["preparation"],
        schedule=dict(step_dts=spec["schedule"]["step_dts"], jump_probabilities=probabilities,
                      jump_choices=draws.tolist()),
        scope="Compilation input only: exact frozen paths, no covariance/accuracy assessment.")


def case_rows():
    # Variant content/seeds are independent of job order and cohort size.
    cfg = dict(trajectory_count=2, randomizations=4, physical_path=list(range(12)), steps=44,
               fold_factors=[1., 1.25, 1.5], seed=29, shots=32, control_shots=2048,
               batch_size=41, controls_per_batch=True)
    rows, _ = run.protocol.plan(cfg)
    paths = [r for r in rows if r["kind"] == "trajectory" and r["trajectory"] == 0]
    controls = [r for r in rows if r["kind"] == "control"]
    assert len(paths) == 13 and len(controls) == 15
    return cfg, paths, controls


def initialize_worker(folder):
    global _WORK
    folder = Path(folder)
    context = verify_seal(json.loads((folder/"context.json").read_text()))
    for name in ("target.pkl", "source.json"):
        if qpy_storage.file_sha256(folder/name) != context["files"][name]:
            raise ValueError("cached target/source changed")
    # Only unpickle the locally created target after checking its recorded hash.
    with (folder/"target.pkl").open("rb") as stream:
        target = pickle.load(stream)
    with (folder/"source.json").open() as stream:
        source = json.load(stream)
    _WORK = context, target, source


def count_unit(kind, identity):
    context, target, source = _WORK
    started = time.monotonic()
    path = context["physical_path"]
    layout = path[1:-1] + [path[0], path[-1]]
    policy = dict(context["dd_policy"], physical_qubits=path[1:-1])
    cfg, path_rows, control_rows = case_rows()
    if kind == "trajectory":
        rows = [dict(r, trajectory=identity) for r in path_rows]
    elif kind == "controls" and identity == "all":
        rows = control_rows
    else:
        raise ValueError("unknown footprint unit")
    native_cache = {}
    vectors = {key:np.zeros(target.num_qubits, dtype=np.int64)
               for key in ("dd", "scheduled_without_dd", "before_scheduling")}
    added_x = np.zeros(target.num_qubits, dtype=np.int64)
    cases = []
    for row in rows:
        key = run.protocol.template_key(row)
        if key not in native_cache:
            control = row["kind"] == "control"
            logical = run.raw.build_body(source, 0 if control else identity,
                steps=0 if control else None, populations=row.get("expected_populations"))
            native = transpile(logical, target=target, initial_layout=layout,
                               optimization_level=3, seed_transpiler=11, num_processes=1)
            if native.layout.final_index_layout(filter_ancillas=True) != layout:
                raise ValueError("transpiler changed the fixed physical mapping")
            native_cache[key] = run.raw.circuits.append_readout(native, layout[:10], "z")
        variant = run.protocol.make_variant(native_cache[key], cfg, row)
        local, baseline_events = run.timing.schedule(variant, target,
            block_ordering_callable=run.flat_block_order)
        program, dd_events = run.timing.schedule(variant, target, policy=policy,
            block_ordering_callable=run.flat_block_order)
        check = run.timing.verify_pair(local, program, baseline_events, dd_events, policy,
                                       alignment_dt=int(target.pulse_alignment or 1))
        validate_isa_circuits([program], target)
        control = row["kind"] == "control"
        operations = program.count_ops()
        expected_cz = 0 if control else 880 + 2*row["folds"]
        if operations.get("cz", 0) != expected_cz or operations.get("reset", 0) != (0 if control else 44):
            raise ValueError("CZ folding or intentional reset count changed")
        outcome = next(r for r in program.cregs if r.name == "outcome")
        measured = sorted((program.find_bit(i.qubits[0]).index, list(outcome).index(i.clbits[0]))
                          for i in program.data if i.operation.name == "measure" and i.clbits[0] in outcome)
        if measured != sorted(zip(layout[:10], range(10))):
            raise ValueError("final readout mapping changed")
        for name, circuit in (("dd", program), ("scheduled_without_dd", local), ("before_scheduling", variant)):
            vectors[name] += audit.footprint(circuit)
        for q, value in check["added_x_by_qubit"].items():
            added_x[int(q)] += value
        cases.append(dict(arm=row["arm"], factor_index=row.get("factor_index"),
            randomization=row["randomization"], control=row.get("control"),
            cz=expected_cz, resets=operations.get("reset", 0), added_x=check["added_x_pulses"],
            original_operations_and_local_times_preserved=check["original_operations_and_local_times_preserved"]))
        del local, baseline_events, program, dd_events, variant
    return seal(dict(artifact_type="model2_dd_footprint_unit_v1", context_sha256=context["sha256"],
        kind=kind, identity=identity, circuits=len(rows),
        instructions={name:value.tolist() for name,value in vectors.items()},
        added_x_by_qubit=added_x.tolist(), cases=cases, seconds=time.monotonic()-started))


def unit_path(folder, kind, identity):
    return folder/("controls.json" if kind == "controls" else f"path_{identity:05d}.json")


def load_unit(folder, context, kind, identity):
    path = unit_path(folder, kind, identity)
    if not path.exists():
        return None
    unit = verify_seal(json.loads(path.read_text()))
    width = context["num_qubits"]
    if (unit.get("context_sha256") != context["sha256"] or unit.get("kind") != kind
            or unit.get("identity") != identity or unit.get("circuits") != (15 if kind == "controls" else 13)):
        raise ValueError(f"footprint context/case mismatch: {path.name}")
    for value in unit["instructions"].values():
        a = np.asarray(value)
        if a.shape != (width,) or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError(f"invalid instruction vector: {path.name}")
    if set(unit["instructions"]) != {"dd", "scheduled_without_dd", "before_scheduling"}:
        raise ValueError("incomplete DD/native comparison")
    return unit


def open_context(options, folder, *, backend=None, service=None):
    name = folder/"context.json"
    if name.exists():
        context = verify_seal(json.loads(name.read_text()))
        if (context["backend"] != options.backend or context["trajectory_count"] != options.trajectory_count
                or context["versions"] != versions() or context["code_hashes"] != code_hashes()
                or (options.physical_path and context["physical_path"] != options.physical_path)):
            raise ValueError("saved count context/settings/compiler changed; use a new output directory")
        initialize_worker(folder)
        print(f"Resuming cached target captured {context['captured_at_utc']}; no account lookup.", flush=True)
        return context
    if options.report_only:
        raise ValueError("no cached count context exists; report-only never queries IBM")
    if any(folder.iterdir()):
        raise ValueError("output directory has no complete context but contains files; use a fresh directory")
    with progress("Reading the backend target only; no Sampler or QPU submission"):
        _, backend = run.dd_runtime.get_backend(options, options.backend, backend=backend, service=service)
    if options.physical_path:
        path = options.physical_path
    else:
        reference = run.read_reference(options.historical_reference)
        if reference["configuration"]["backend"] != options.backend:
            raise ValueError("supply an explicit physical-path when using another backend")
        path = reference["configuration"]["physical_path"]
    run.raw.circuits.chain_layout(backend, 10, "boundary", path)
    model = run.timing.timing_model(backend, path)
    source = compilation_source(options.trajectory_count)
    # Target only: no backend service/account object is serialized.
    with (folder/"target.pkl").open("wb") as stream:
        pickle.dump(backend.target, stream, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_json(folder/"source.json", source)
    context = seal(dict(artifact_type=TYPE, backend=backend.name, trajectory_count=options.trajectory_count,
        num_qubits=backend.num_qubits, physical_path=list(path), timing_model=model,
        captured_at_utc=datetime.now(timezone.utc).isoformat(), versions=versions(), code_hashes=code_hashes(),
        spec_sha256=input_data.SPEC_SHA256, seed=29, seed_transpiler=11, optimization_level=3,
        dd_policy=dict(sequence="XX", spacings=[.25, .5, .25], min_idle_ratio=10.,
            system_sites_1based=list(range(1, 11)), skip_reset_qubits=True, insert_multiple_cycles=False),
        weights=audit.WEIGHTS, implicit_init=True,
        files={n:qpy_storage.file_sha256(folder/n) for n in ("target.pkl", "source.json")}))
    atomic_json(name, context)
    initialize_worker(folder)
    return context


def count_missing(folder, context, indices, workers):
    controls = load_unit(folder, context, "controls", "all")
    if controls is None:
        with progress("Compiling/counting all 15 DD-on job-local control circuits"):
            controls = count_unit("controls", "all")
        atomic_json(unit_path(folder, "controls", "all"), controls)
    units = {}
    for index in indices:
        unit = load_unit(folder, context, "trajectory", index)
        if unit is not None:
            units[index] = unit
    missing = [i for i in indices if i not in units]
    print(f"Verified cache: {len(units):,}/{len(indices):,} paths; {len(missing):,} to count; workers={workers}.", flush=True)
    started = time.monotonic()
    counted = 0

    def record(index, unit):
        nonlocal counted
        if unit["context_sha256"] != context["sha256"] or unit["identity"] != index:
            raise ValueError("worker returned another footprint context")
        atomic_json(unit_path(folder, "trajectory", index), unit)
        units[index] = unit
        counted += 1
        elapsed = time.monotonic()-started
        remaining = (len(missing)-counted)*elapsed/counted
        print(f"Counted DD path {len(units):,}/{len(indices):,}: id {index}, "
              f"13 variants, {unit['seconds']:.1f}s; session {elapsed/60:.1f} min, ETA {remaining/3600:.2f} h", flush=True)

    if workers == 1:
        for index in missing:
            with progress(f"Compiling/counting DD path {index}, all factors and randomizations"):
                unit = count_unit("trajectory", index)
            record(index, unit)
    elif missing:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_worker, initargs=(str(folder),))
        pending, iterator = {}, iter(missing)
        try:
            for _ in range(min(workers, len(missing))):
                index = next(iterator)
                pending[pool.submit(count_unit, "trajectory", index)] = index
            while pending:
                done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                if not done:
                    print(f"Counting DD paths: {len(units):,}/{len(indices):,} checkpointed; "
                          f"{len(pending)} active; session {(time.monotonic()-started)/60:.1f} min", flush=True)
                for future in done:
                    index = pending.pop(future)
                    record(index, future.result())
                    index = next(iterator, None)
                    if index is not None:
                        pending[pool.submit(count_unit, "trajectory", index)] = index
        finally:
            # Only workers current paths can be outstanding; nothing enqueues the whole cohort.
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    return [units[i] for i in indices], controls


def balanced_plan(vectors, controls, path_shots, control_shots, limit, greedy):
    """Check the grouping reproducible by the existing runner's batch-size option."""
    count = len(vectors)
    for wanted in range(greedy["lower_bound_jobs"], min(count//2, greedy["jobs"]+32)+1):
        capacity = math.ceil(count/wanted)
        actual = math.ceil(count/capacity)
        groups = np.array_split(np.arange(count), actual)
        if min(map(len, groups)) < 2:
            continue
        plan = packing.describe_groups(vectors, controls, groups, path_shots, control_shots, limit)
        if plan["within_counted_limits"]:
            plan["runner_batch_size_for_this_grouping"] = 13*capacity+15
            return plan
    return None


def estimate(units, controls, context, options):
    observed = np.asarray([u["instructions"]["dd"] for u in units], dtype=np.int64)
    control = np.asarray(controls["instructions"]["dd"], dtype=np.int64)
    count = options.trajectory_count
    exact = len(units) == count
    vectors = observed if exact else audit.repeat_rows(observed, count)
    order = np.random.default_rng(29).permutation(count)
    vectors = vectors[order]
    path_shots, control_shots = 4*options.shots, 6*options.control_shots
    budget = int(options.instruction_fraction*audit.INSTRUCTION_LIMIT)
    published = packing.pack(vectors, control, path_shots, control_shots, audit.INSTRUCTION_LIMIT)
    selected = packing.pack(vectors, control, path_shots, control_shots, budget)
    balanced_published = balanced_plan(vectors, control, path_shots, control_shots, audit.INSTRUCTION_LIMIT, published)
    balanced = balanced_plan(vectors, control, path_shots, control_shots, budget, selected)
    capacity = (options.circuit_cap-15)//13
    groups = np.array_split(np.arange(count), math.ceil(count/capacity))
    capped = packing.describe_groups(vectors, control, groups, path_shots, control_shots, budget, options.circuit_cap)
    if min(map(len, groups)) < 2:
        capped["within_counted_limits"] = False
        capped["minimum_for_supplied_counting_model"] = False
        capped["runner_refusal"] = "grouping contains fewer than two paths in a job"
    scenarios = dict(published_limit=published, balanced_runner_published_limit=balanced_published, selected_fraction=selected,
                     balanced_runner_selected_fraction=balanced, current_circuit_cap=capped)
    for plan in scenarios.values():
        if plan is not None:
            plan["minimum_certified_for_full_cohort_counting_model"] = bool(exact and plan["minimum_for_supplied_counting_model"])
    summaries = {}
    for name, candidates in (("published_limit", (published, balanced_published)),
                             ("selected_fraction", (selected, balanced))):
        feasible = min(p["jobs"] for p in candidates if p is not None and p["within_counted_limits"])
        lower = candidates[0]["lower_bound_jobs"]
        summaries[name] = dict(lower_bound_jobs=lower, best_feasible_jobs=feasible,
            full_cohort_minimum_jobs=feasible if exact and feasible == lower else None,
            status="minimum certified for the full-cohort counting model" if exact and feasible == lower else
                   "projection only" if not exact else "lower/upper bounds; minimum not certified")
    return dict(experiment=TYPE, created_at_utc=datetime.now(timezone.utc).isoformat(),
        backend=context["backend"], physical_path=context["physical_path"], target_captured_at_utc=context["captured_at_utc"],
        context_sha256=context["sha256"], configuration=dict(trajectory_count=count, shots_per_arm_factor=options.shots,
            control_shots_per_preparation_treatment=options.control_shots, randomizations=4, factors=[1.,1.25,1.5],
            dd_policy=context["dd_policy"], counted_paths=len(units), footprint_projection=not exact,
            instruction_fraction=options.instruction_fraction, comparison_circuit_cap=options.circuit_cap,
            current_hardware_runner_accepts_cohort=count <= 16384),
        counting=dict(limit_source=audit.LIMIT_SOURCE, instruction_limit=audit.INSTRUCTION_LIMIT,
            execution_limit=audit.EXECUTION_LIMIT, weights=audit.WEIGHTS, implicit_init=True,
            instructions_multiplied_by_shots=False, path_circuits=13, control_circuits_per_job=15,
            path_executions=path_shots, control_executions_per_job=control_shots),
        native_summary=dict(controls_instructions=control.tolist(),
            mean_dd_instructions_per_path=observed.mean(axis=0).tolist(),
            maximum_dd_instructions_per_path=observed.max(axis=0).tolist(),
            mean_added_x_per_path=float(np.mean([sum(u["added_x_by_qubit"]) for u in units])),
            mean_before_scheduling_instructions_per_path=np.mean([u["instructions"]["before_scheduling"] for u in units],axis=0).tolist(),
            mean_scheduled_without_dd_instructions_per_path=np.mean([u["instructions"]["scheduled_without_dd"] for u in units],axis=0).tolist()),
        path_order=order.tolist(), unit_hashes=[u["sha256"] for u in units], control_hash=controls["sha256"],
        minimum_without_circuit_cap=summaries, scenarios=scenarios,
        scope="Published instruction-counting heuristic on the frozen backend target, after folding/twirling/scheduling/DD. "
              "A minimum is certified only if all paths were counted AND feasible packing equals the lower bound. "
              "Explicit samples repeat numeric footprints only and cannot certify the full cohort. "
              "No QPU jobs, compiler acceptance test, QPU-time/dollar estimate, accuracy claim or hardware-plan mutation. "
              "Partitions are planning output; they are not installed into a hardware manifest.")


def save_report(report, folder):
    atomic_json(folder/"job_estimate.json", report)
    c = report["configuration"]
    summaries = report["minimum_without_circuit_cap"]
    leading = []
    for name, value in summaries.items():
        leading.append(f"**{name}: lower bound {value['lower_bound_jobs']}, best feasible "
                       f"{value['best_feasible_jobs']} jobs; {value['status']}.**")
    lines = ["# Model2 mitigation + DD: minimum-job analysis", "",
        f"Backend `{report['backend']}`; physical path `{report['physical_path']}`; target captured {report['target_captured_at_utc']}.", "",
        f"**{c['trajectory_count']:,} trajectories × {c['shots_per_arm_factor']} shots per arm/factor; "
        f"{c['counted_paths']:,} actual native DD-on paths counted. Projection: {c['footprint_projection']}.**", "",
        "All thirteen matched variants stay together. Each job includes all fifteen zero/one/half-mixture control circuits. "
        "Gate twirling and folding precede the actual Experiment8 ALAP + ratio-10 XX passes. Final explicit delays, DD X gates, "
        "resets, measurements and implicit initialization are counted. Shots are summed separately.", "",
        "| Packing | Feasible jobs | Lower bound | Full-cohort minimum certified | Circuits | Shots incl. controls | Peak instructions/qubit | Within counted limits |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for name, plan in report["scenarios"].items():
        if plan is None:
            continue
        lines.append(f"| {name} | {plan['jobs']} | {plan['lower_bound_jobs']} | "
            f"{plan['minimum_certified_for_full_cohort_counting_model']} | {plan['circuits']:,} | "
            f"{plan['shots']:,} | {plan['maximum_instructions']:,} | {plan['within_counted_limits']} |")
        path = folder/(name+"_jobs.csv")
        with path.open("w",newline="") as stream:
            fields = list(plan["partitions"][0])+["trajectory_ids"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in plan["partitions"]:
                ids = report["path_order"][row["path_start"]:row["path_stop"]]
                writer.writerow(dict(row, trajectory_ids=";".join(map(str,ids))))
    balanced = report["scenarios"]["balanced_runner_selected_fraction"]
    if balanced is not None:
        lines += ["", f"The explicitly checked balanced grouping corresponds to runner `--batch-size "
            f"{balanced['runner_batch_size_for_this_grouping']}` under the same target and cohort. "
            "This is a planning result. Actual production preparation still recompiles/validates every DD circuit; "
            "larger Sampler submissions also require more host memory."]
    lines += ["", "The lower bound is the maximum of each physical qubit's total path footprint divided by its "
        "per-job allowance after reserving controls, and the analogous execution bound. A feasible partition attaining "
        "that integer lower bound proves the minimum for this counting model. If the two numbers differ, this script "
        "reports an interval, not a proven minimum. The 80% allowance is a separate planning choice; "
        "the configured 512-circuit comparison cap is not an IBM minimum.", "",
        "A projection cannot establish the full-cohort minimum. CSV path_start/path_stop index the saved path_order, "
        "and trajectory_ids gives actual IDs. No existing hardware manifests, circuits, job IDs or controls are changed.", "",
        "The cache freezes the target and compiler environment for reproducibility; resuming does not silently mix "
        "calibrations. Fetch a new target into a new output directory if required. Counts are not server-compiler acceptance "
        "or measured instruction-memory usage.", "",
        "The estimator accepts 16,834 for accounting if requested, but the current hardware runner still caps new cohorts "
        "at 16,384; this script does not change that protocol limit.", "",
        f"[IBM job limits]({audit.LIMIT_SOURCE}). Machine-readable details: [job_estimate.json](job_estimate.json)."]
    lines[6:6] = leading + [""]
    (folder/"JobEstimate.md").write_text("\n".join(lines)+"\n")
    print(f"Counted {c['counted_paths']:,}/{c['trajectory_count']:,} paths; projection={c['footprint_projection']}.", flush=True)
    print("\n".join(leading), flush=True)
    capped = report["scenarios"]["current_circuit_cap"]
    print(f"Configured {c['comparison_circuit_cap']}-circuit cap comparison only: {capped['jobs']} jobs.", flush=True)
    print(f"Report: {folder/'JobEstimate.md'}", flush=True)


def parse_arguments(arguments=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="ibm_phoenix")
    p.add_argument("--account-file", type=Path)
    p.add_argument("--trajectory-count", type=int, default=16384)
    p.add_argument("--shots", type=int, default=32)
    p.add_argument("--control-shots", type=int, default=2048)
    p.add_argument("--sample-trajectories", type=int, help="explicit projection only; omit for exhaustive native counting")
    p.add_argument("--workers", type=int, default=1, help="independent compiler processes; no nested transpiler parallelism")
    p.add_argument("--physical-path", type=int, nargs=12)
    p.add_argument("--historical-reference", type=Path, default=run.DEFAULT_REFERENCE)
    p.add_argument("--instruction-fraction", type=float, default=.8)
    p.add_argument("--circuit-cap", type=int, default=512, help="comparison only, not imposed on the minimum-job search")
    p.add_argument("--output-directory", type=Path)
    p.add_argument("--report-only", action="store_true", help="read cached counts only; never connect to IBM or compile")
    a = p.parse_args(arguments)
    if (a.trajectory_count < 2 or a.shots < 8 or a.shots % 4 or a.control_shots < 8 or a.control_shots % 4
            or a.workers < 1 or not np.isfinite(a.instruction_fraction) or not 0 < a.instruction_fraction <= 1
            or a.circuit_cap < 41 or not a.backend.strip()
            or (a.sample_trajectories is not None and not 2 <= a.sample_trajectories <= a.trajectory_count)):
        p.error("need >=2 paths, shots/control shots divisible by 4 and >=8, positive workers, fraction in (0,1], cap>=41")
    if a.physical_path and (len(set(a.physical_path)) != 12 or min(a.physical_path) < 0):
        p.error("physical-path needs twelve distinct nonnegative qubits")
    if a.output_directory is None:
        backend_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in a.backend)
        a.output_directory = Path(__file__).parent/"job_estimates"/f"{backend_label}_{a.trajectory_count}"
    return a


def main(options=None, *, backend=None, service=None):
    options = options or parse_arguments()
    folder = options.output_directory.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    context = open_context(options, folder, backend=backend, service=service)
    indices = list(range(options.sample_trajectories or options.trajectory_count))
    if options.report_only:
        controls = load_unit(folder, context, "controls", "all")
        units = [load_unit(folder, context, "trajectory", i) for i in indices]
        if controls is None or any(u is None for u in units):
            raise ValueError("requested native counts are incomplete; no full-cohort minimum can be reported")
    else:
        units, controls = count_missing(folder, context, indices, options.workers)
    report = estimate(units, controls, context, options)
    save_report(report, folder)
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped locally. Verified numeric checkpoints are retained; rerun the same command to resume.") from None
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(str(error)) from error
