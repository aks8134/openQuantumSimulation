"""Estimate Model2 mitigation jobs; never submit a QPU job.

Default: 16,384 paths, 32 shots per arm/factor, the bundled 32-path
FakeKingston footprint sample. --fake-run recounts saved pilot QPY;
--manifest recounts a compiled Model2 mitigation archive. Sample expansion
repeats instruction footprints, never quantum trajectories. Reports compare
memory-based packing with the hardware runner's circuit cap. --backend
loads a live IBM target and locally compiles a sample (default 32 paths).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, instruction_audit as audit, qpy_storage

DEFAULT_PROFILE = DIRECTORY/"inputs"/"fake_kingston_mitigation_footprints.json"
PROFILE_TYPE = "model2_mitigation_footprints_v1"


def seal(profile):
    profile["sha256"] = input_data.digest({k: v for k, v in profile.items() if k != "sha256"})
    return profile


def validate_profile(profile):
    if (profile.get("artifact_type") != PROFILE_TYPE or profile.get("sha256") !=
            input_data.digest({k: v for k, v in profile.items() if k != "sha256"})):
        raise ValueError("footprint profile type/hash mismatch")
    config = profile["protocol"]
    if (config["spec_sha256"] != input_data.SPEC_SHA256 or config["steps"] != 44
            or config["fold_factors"] != [1., 1.25, 1.5] or config["randomizations"] != 4
            or profile["weights"] != audit.WEIGHTS or profile["implicit_init"] is not True):
        raise ValueError("need frozen Model2 cosine44, four randomizations, factors 1/1.25/1.5 and current counting weights")
    paths = np.asarray(profile["path_instructions"])
    controls = np.asarray(profile["control_instructions"])
    width = len(profile["physical_qubit_indices"])
    if (paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] != width or controls.shape != (width,)
            or paths.dtype.kind not in "iu" or controls.dtype.kind not in "iu"
            or np.any(paths < 0) or np.any(controls < 0) or not np.any(paths)
            or profile["path_ids"] != list(range(len(paths)))
            or len(set(profile["physical_qubit_indices"])) != width
            or profile["source"]["kind"] not in ("saved_fake_kingston", "compiled_mitigation_manifest", "backend_compilation")):
        raise ValueError("invalid or incomplete per-path/control footprint matrix")
    return paths.astype(np.int64), controls.astype(np.int64)


def profile_from_fake(folder):
    """Read/hash one compact QPY unit at a time; no noise simulation required."""
    from Model3.Experiment5 import run_mitigation as protocol
    folder = Path(folder).resolve()
    config = json.loads((folder/"configuration.json").read_text())
    source = json.loads((folder/"source.json").read_text())
    frozen = input_data.load_spec()
    if (config["spec_sha256"] != input_data.SPEC_SHA256 or source["configuration"]["physics"] != frozen["physics"]
            or source["experiment"] != "Model2/Experiment5" or source["preparation"] != frozen["preparation"]
            or source["schedule"]["step_dts"] != frozen["schedule"]["step_dts"]
            or source["configuration"]["seed"] != config["seed"]
            or source["configuration"]["trajectories"] != config["trajectory_count"]
            or source["configuration"]["order"] != "jump-first"
            or input_data.digest(source) != config["source_sha256"]):
        raise ValueError("FakeKingston source/spec hash or Model2 physics mismatch")
    planned, _ = protocol.plan(config)
    expected = {row["pub_index"]: row for row in planned}
    seen, vectors, controls, archives = set(), {}, None, []
    names = ["control_"+name for name in ("all_zero", "all_one", "initial_mixture")]
    names += [f"trajectory_{i}" for i in range(config["trajectory_count"])]
    for name in names:
        unit = json.loads((folder/(name+".json")).read_text())
        path = folder/unit["qpy_file"]
        if (qpy_storage.file_sha256(path) != unit["qpy_sha256"]
                or unit["source_sha256"] != config["source_sha256"]
                or unit["noise_model_sha256"] != config["noise_model_sha256"]):
            raise ValueError("FakeKingston QPY/source/noise identity mismatch")
        rows = unit["records"]
        programs, _ = next(qpy_storage.Archive(path, len(rows)).iter_batches([dict(start=0, stop=len(rows))]))
        for program, row in zip(programs, rows):
            index = row["pub_index"]
            if index in seen or index not in expected or any(row.get(k) != v for k, v in expected[index].items()):
                raise ValueError("missing/duplicate or changed FakeKingston mitigation variant")
            seen.add(index)
            operations = program.count_ops()
            control = row["kind"] == "control"
            if (program.num_qubits != 12 or operations.get("cz", 0) != (0 if control else 880+2*row["folds"])
                    or operations.get("reset", 0) != (0 if control else 44)
                    or audit.run_hardware.circuits.metrics(program) != row["metrics"]):
                raise ValueError("FakeKingston native dimensions, folds, resets or metrics changed")
            footprint = audit.footprint(program)
            if control:
                controls = footprint.copy() if controls is None else controls+footprint
            else:
                vectors.setdefault(row["trajectory"], np.zeros(12, dtype=np.int64))[:] += footprint
        archives.append(dict(file=path.name, sha256=unit["qpy_sha256"], circuits=len(rows)))
        del programs, program
        print(f"Counted {name}", flush=True)
    if seen != set(expected) or set(vectors) != set(range(config["trajectory_count"])):
        raise ValueError("incomplete FakeKingston mitigation archive")
    return make_profile(config, [vectors[i] for i in range(len(vectors))], controls,
        config["active_compact_to_physical"], dict(kind="saved_fake_kingston", backend="fake_kingston",
        configuration_sha256=qpy_storage.file_sha256(folder/"configuration.json"),
        snapshot_date=config["snapshot_date"], versions=config["versions"], archives=archives,
        scope="Measured native footprints from the saved compact FakeKingston pilot; not Phoenix circuits."))


def profile_from_manifest(path):
    from Model2.Experiment6 import run_hardware as runner
    if json.loads(Path(path).read_text()).get("experiment") != runner.EXPERIMENT:
        raise ValueError("need a Model2 Experiment6 mitigation manifest; a raw Experiment5 run lacks twirled/folded variants")
    folder, manifest, _ = runner.read_saved(Path(path), require_circuits=True)
    if not all(batch["compiled"] for batch in manifest["batches"]):
        raise ValueError("compile all mitigation jobs with --compile-only before estimating from their archive")
    config = manifest["configuration"]
    store = runner.Publications(manifest, folder)
    vectors, control_sets, archives = {}, [], []
    width = None
    for batch in manifest["batches"]:
        programs = store[batch["start"]:batch["stop"]]
        local_controls = None
        for program, row in zip(programs, manifest["circuits"][batch["start"]:batch["stop"]]):
            width = program.num_qubits if width is None else width
            if program.num_qubits != width:
                raise ValueError("compiled mitigation circuits have inconsistent allocated widths")
            value = audit.footprint(program)
            if row["kind"] == "control":
                local_controls = value.copy() if local_controls is None else local_controls+value
            else:
                vectors.setdefault(row["trajectory"], np.zeros(width, dtype=np.int64))[:] += value
        control_sets.append(local_controls)
        archives.append(dict(file=batch["isa_file"], sha256=batch["isa_sha256"], circuits=len(programs)))
        del programs, program
        print(f"Counted native circuits {batch['start']}..{batch['stop']-1}", flush=True)
    # Componentwise maximum reserves enough space for any observed control set.
    controls = np.max(control_sets, axis=0)
    if set(vectors) != set(range(config["trajectory_count"])):
        raise ValueError("incomplete compiled mitigation paths")
    return make_profile(config, [vectors[i] for i in range(len(vectors))], controls, list(range(width)),
        dict(kind="compiled_mitigation_manifest", backend=config["backend"], plan_sha256=manifest["plan_sha256"],
        versions=config["software_versions"], archives=archives,
        controls_identical=all(np.array_equal(c, controls) for c in control_sets),
        scope="Counted all saved native mitigation programs; compilation backend/snapshot retained, no backend query."))


def make_profile(config, vectors, controls, physical, source):
    profile = seal(dict(artifact_type=PROFILE_TYPE, protocol={key: config[key] for key in
        ("spec_sha256", "steps", "seed", "randomizations", "fold_factors")}, source=source,
        weights=audit.WEIGHTS, implicit_init=True, physical_qubit_indices=physical,
        path_ids=list(range(len(vectors))), path_instructions=np.asarray(vectors).tolist(),
        control_instructions=np.asarray(controls).tolist()))
    validate_profile(profile)
    return profile


def profile_from_backend(options, *, backend=None, service=None):
    """Count actual native variants one template at a time, without a Sampler.

    Only numeric footprints and compilation provenance are saved. No QPY
    archive or executable hardware manifest is generated by this mode.
    """
    from qiskit import transpile
    from Model2.Experiment6 import run_hardware as runner
    from Model3.Experiment5 import run_mitigation as protocol

    # Reuse the hardware runner's argument validation and calibration policy.
    arguments = ["--compile-only", "--backend", options.backend,
        "--trajectory-count", str(options.sample_trajectories), "--shots", str(options.shots),
        "--control-shots", str(options.control_shots), "--seed", str(options.seed),
        "--seed-transpiler", str(options.seed_transpiler), "--optimization-level", str(options.optimization_level),
        "--calibration-policy", options.calibration_policy, "--min-t1-us", str(options.min_t1_us),
        "--min-t2-us", str(options.min_t2_us), "--max-cz-error", str(options.max_cz_error),
        "--max-readout-error", str(options.max_readout_error)]
    if options.physical_path:
        arguments += ["--physical-path", *map(str, options.physical_path)]
    if options.reference_manifest:
        arguments += ["--reference-manifest", str(options.reference_manifest)]
    if options.avoid_qubits:
        arguments += ["--avoid-qubits", *map(str, options.avoid_qubits)]
    settings = runner.parse_arguments(arguments)
    frozen = input_data.load_spec()
    fixed, reference = settings.physical_path, None
    if settings.reference_manifest:
        fixed, reference = runner.reference_path(settings.reference_manifest, settings.backend, settings.seed, frozen)
    if backend is None:
        service = service or runner.raw.runtime.make_service(options)
        backend = service.backend(options.backend, use_fractional_gates=False)
    if backend.name != options.backend:
        raise ValueError("returned backend differs from the requested estimation backend")
    if "cz" not in backend.target.operation_names:
        raise ValueError("this mitigation protocol requires a native CZ backend; no CZ-to-other-gate substitution is estimated")
    sample = input_data.make_source(frozen, options.sample_trajectories, options.seed)
    path_settings = dict(avoid_qubits=settings.avoid_qubits, min_t1_seconds=settings.min_t1_us*1e-6,
        min_t2_seconds=settings.min_t2_us*1e-6, max_cz_error=settings.max_cz_error,
        max_readout_error=settings.max_readout_error, max_expansions=settings.max_path_expansions)
    indices = list(range(options.sample_trajectories))
    if fixed is None:
        selected = runner.raw.quality_path.select(backend, sample, indices, avoid=settings.avoid_qubits,
            min_t1=path_settings["min_t1_seconds"], min_t2=path_settings["min_t2_seconds"],
            max_cz_error=settings.max_cz_error, max_readout_error=settings.max_readout_error,
            max_expansions=settings.max_path_expansions)
    elif settings.calibration_policy == "screen":
        selected = runner.raw.quality_path.validate(backend, fixed, sample, indices, path_settings)
    else:
        runner.raw.circuits.chain_layout(backend, 10, "boundary", fixed)
        selected = dict(physical_path=fixed, policy="record-only", scope="Explicit mapping; no quality filtering.")
    path = selected["physical_path"]
    layout = path[1:-1]+[path[0], path[-1]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = options.output_directory/("compilation_"+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/"source.json", sample)
    atomic_json(folder/"selected_path.json", selected)
    runner.raw.runtime.save_backend_snapshot(backend, path, [], folder)
    provenance = dict(kind="backend_compilation", backend=backend.name, physical_path=path,
        captured_at_utc=datetime.now(timezone.utc).isoformat(), versions=runner.versions(),
        seed_transpiler=options.seed_transpiler, optimization_level=options.optimization_level,
        calibration_policy=settings.calibration_policy, path_settings=path_settings, reference=reference,
        backend_status=runner.raw.backend_readiness(backend),
        backend_snapshot_sha256=qpy_storage.file_sha256(folder/"backend_snapshot.json"),
        source_sha256=qpy_storage.file_sha256(folder/"source.json"),
        snapshot_directory=str(folder.resolve()), requested_sample_paths=options.sample_trajectories,
        compilation_complete=False,
        scope="Actual locally compiled native mitigation variants for this backend/path/target snapshot. No QPU job submitted or provider compiler validation performed.")
    config = dict(spec_sha256=input_data.SPEC_SHA256, trajectory_count=options.sample_trajectories,
        steps=44, seed=options.seed, physical_path=path, fold_factors=[1., 1.25, 1.5], randomizations=4,
        shots=options.shots, control_shots=options.control_shots, controls_per_batch=True, batch_size=512)
    records, _ = protocol.plan(config)
    units = {}
    for row in records:
        units.setdefault(protocol.template_key(row), []).append(row)
    vectors, controls = [], np.zeros(backend.num_qubits, dtype=np.int64)
    print(f"Counting {options.sample_trajectories} Model2 paths compiled for {backend.name}; physical path {path}.", flush=True)
    print(f"Compilation provenance: {folder}. No Sampler or job submission.", flush=True)
    # Compile controls once: frames depend on control name, not its job number.
    keys = [("control", name) for name in ("all_zero", "all_one", "initial_mixture")]
    keys += [("trajectory", i) for i in indices]
    for kind, key in keys:
        rows = units[(kind, key)]
        # Identical control sets are repeated in protocol.plan for each job.
        rows = rows[:5] if kind == "control" else rows
        expected_count = 5 if kind == "control" else 13
        if len(rows) != expected_count:
            raise ValueError("incomplete native mitigation template")
        logical = (runner.raw.build_body(sample, 0, steps=0, populations=rows[0]["expected_populations"])
                   if kind == "control" else runner.raw.build_body(sample, key))
        native = transpile(logical, backend=backend, initial_layout=layout,
            optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler)
        if native.layout.final_index_layout(filter_ancillas=True) != layout:
            raise ValueError("transpiler changed the fixed physical mapping")
        native = runner.raw.circuits.append_readout(native, layout[:10], "z")
        total = np.zeros(backend.num_qubits, dtype=np.int64)
        for row in rows:
            program = protocol.make_variant(native, config, row)
            operations = program.count_ops()
            if (operations.get("cz", 0) != (0 if kind == "control" else 880+2*row["folds"])
                    or operations.get("reset", 0) != (0 if kind == "control" else 44)):
                raise ValueError("native CZ fold or intentional reset count differs from the frozen protocol")
            protocol.assert_supported(program, backend)
            total += audit.footprint(program)
            del program
        del native, logical
        if kind == "control":
            controls += total
        else:
            vectors.append(total)
            # A partial profile remains a labelled sample, not a full count.
            if len(vectors) >= 2 and (len(vectors) % 32 == 0 or len(vectors) == len(indices)):
                provenance["compilation_complete"] = len(vectors) == len(indices)
                profile = make_profile(config, vectors, controls, list(range(backend.num_qubits)), dict(provenance))
                atomic_json(folder/"footprints.json", profile)
            print(f"Counted native path {len(vectors)}/{len(indices)}", flush=True)
    return profile


def describe_groups(vectors, controls, groups, path_shots, control_shots, memory_limit, circuit_cap=None):
    jobs = []
    for indices in groups:
        load = controls+vectors[indices].sum(axis=0)
        shots = len(indices)*path_shots+control_shots
        circuits = 13*len(indices)+15
        jobs.append(dict(job=len(jobs)+1, path_start=int(indices[0]), path_stop=int(indices[-1])+1,
            trajectories=len(indices), circuits=circuits, shots=shots,
            maximum_instructions=int(load.max()), bottleneck_position=int(load.argmax()),
            within_limits=bool(load.max() <= memory_limit and shots <= audit.EXECUTION_LIMIT
                               and (circuit_cap is None or circuits <= circuit_cap))))
    capacity = memory_limit-controls
    lower = max(1, max(math.ceil(int(t)/int(c)) for t, c in zip(vectors.sum(axis=0), capacity)),
                math.ceil(len(vectors)*path_shots/(audit.EXECUTION_LIMIT-control_shots)))
    if circuit_cap is not None:
        lower = max(lower, math.ceil(len(vectors)/((circuit_cap-15)//13)))
    return dict(jobs=len(jobs), lower_bound_jobs=lower,
        minimum_for_supplied_counting_model=all(j["within_limits"] for j in jobs) and lower == len(jobs),
        within_counted_limits=all(j["within_limits"] for j in jobs),
        circuits=sum(j["circuits"] for j in jobs), shots=sum(j["shots"] for j in jobs),
        evolution_shots=len(vectors)*path_shots, control_shots=len(jobs)*control_shots,
        minimum_paths_per_job=min(j["trajectories"] for j in jobs), maximum_paths_per_job=max(j["trajectories"] for j in jobs),
        maximum_instructions=max(j["maximum_instructions"] for j in jobs),
        instruction_limit=memory_limit, circuit_cap=circuit_cap, partitions=jobs)


def pack(vectors, controls, path_shots, control_shots, memory_limit):
    """Greedy contiguous whole-path packing, with at least two paths per job."""
    if np.any(controls >= memory_limit) or control_shots >= audit.EXECUTION_LIMIT:
        raise ValueError("controls alone exhaust the instruction/execution budget")
    if np.any(vectors+controls > memory_limit) or path_shots+control_shots > audit.EXECUTION_LIMIT:
        raise ValueError("one whole matched path plus controls exceeds the limits")
    groups, begin, load, shots = [], 0, controls.copy(), control_shots
    for index, value in enumerate(vectors):
        if np.any(load+value > memory_limit) or shots+path_shots > audit.EXECUTION_LIMIT:
            groups.append(np.arange(begin, index))
            begin, load, shots = index, controls.copy(), control_shots
        load += value
        shots += path_shots
    groups.append(np.arange(begin, len(vectors)))
    if len(groups[-1]) == 1 and len(groups) > 1 and len(groups[-2]) >= 3:
        candidate = np.arange(groups[-2][-1], len(vectors))
        if (np.all(controls+vectors[candidate].sum(axis=0) <= memory_limit)
                and control_shots+len(candidate)*path_shots <= audit.EXECUTION_LIMIT):
            groups[-2], groups[-1] = groups[-2][:-1], candidate
    if any(len(g) < 2 for g in groups):
        raise ValueError("counted limits cannot pack this path order with at least two paths per job")
    return describe_groups(vectors, controls, groups, path_shots, control_shots, memory_limit)


def validate_request(trajectory_count, shots, control_shots, instruction_fraction, circuit_cap):
    if (not 2 <= trajectory_count <= 16384 or min(shots, control_shots) < 8
            or shots % 4 or control_shots % 4 or not np.isfinite(instruction_fraction)
            or not 0 < instruction_fraction <= 1 or circuit_cap < 41):
        raise ValueError("need 2..16384 paths, shots/control-shots divisible by four and >=8, fraction in (0,1], circuit-cap>=41")


def estimate(profile, *, trajectory_count=16384, shots=32, control_shots=2048, instruction_fraction=.8, circuit_cap=512):
    observed, controls = validate_profile(profile)
    validate_request(trajectory_count, shots, control_shots, instruction_fraction, circuit_cap)
    # Projection cyclically repeats footprints only; all same-path quantum data stay untouched.
    vectors = audit.repeat_rows(observed, trajectory_count)
    order = np.random.default_rng(profile["protocol"]["seed"]).permutation(trajectory_count)
    vectors = vectors[order]
    path_shots, controls_shots = 4*shots, 6*control_shots
    budget = int(audit.INSTRUCTION_LIMIT*instruction_fraction)
    full = pack(vectors, controls, path_shots, controls_shots, audit.INSTRUCTION_LIMIT)
    selected = pack(vectors, controls, path_shots, controls_shots, budget)
    envelope = pack(np.repeat(observed.max(axis=0)[None, :], trajectory_count, axis=0),
                    controls, path_shots, controls_shots, budget)
    capacity = (circuit_cap-15)//13
    groups = np.array_split(np.arange(trajectory_count), math.ceil(trajectory_count/capacity))
    if any(len(g) < 2 for g in groups):
        raise ValueError("circuit cap yields a singleton job; adjust trajectory-count or circuit-cap")
    if np.any(controls >= budget) or controls_shots >= audit.EXECUTION_LIMIT:
        raise ValueError("controls exhaust the configured counting budget")
    current = describe_groups(vectors, controls, groups, path_shots, controls_shots, budget, circuit_cap)
    return dict(experiment="Model2/Experiment6/mitigation_job_estimate_v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(), profile_sha256=profile["sha256"], source=profile["source"],
        configuration=dict(trajectory_count=trajectory_count, shots_per_arm_factor=shots,
            shots_per_twirled_variant=shots//4, control_shots_per_preparation_treatment=control_shots,
            randomizations=4, fold_factors=[1., 1.25, 1.5], instruction_fraction=instruction_fraction,
            current_runner_circuit_cap=circuit_cap, observed_paths=len(observed),
            footprint_expansion=trajectory_count > len(observed), path_order="numpy.default_rng(saved_seed).permutation(target_count)",
            seed=profile["protocol"]["seed"], spec_sha256=profile["protocol"]["spec_sha256"]),
        counting=dict(limit_source=audit.LIMIT_SOURCE, instruction_limit=audit.INSTRUCTION_LIMIT,
            execution_limit=audit.EXECUTION_LIMIT, weights=audit.WEIGHTS, implicit_init=True,
            instructions_multiplied_by_shots=False, path_circuits=13, control_circuits_per_job=15,
            controls_reservation="componentwise maximum across observed complete job control sets"),
        physical_qubit_indices=profile["physical_qubit_indices"],
        scenarios=dict(at_published_limit=full, at_selected_fraction=selected,
            observed_maximum_envelope=envelope, current_runner_circuit_cap=current),
        scope="Local planning. Expanded samples repeat footprints, not quantum trajectories. A backend compilation uses its fetched target, not a QPU execution or provider compiler acceptance test. No submission, hardware-manifest changes or 0.02 accuracy certification. An observed-maximum envelope does not bound unseen circuits.")


def save_report(report, folder):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/"job_estimate.json", report)
    config = report["configuration"]
    source = report["source"]
    detail = (f"Native variants were compiled against the requested backend target on physical path `{source['physical_path']}`. "
              "If footprint expansion is false, every requested path was counted; otherwise this is a backend-specific sample projection."
              if source["kind"] == "backend_compilation" else
              "Counts are from the supplied archive/profile. The bundled default sample is FakeKingston, not Phoenix; expanded samples are projections.")
    lines = ["# Model2 full mitigation job estimate", "",
        f"**{config['trajectory_count']:,} trajectories × {config['shots_per_arm_factor']} shots per arm/factor.** "
        "Four variants per twirled factor, three factors (1, 1.25, 1.5), paired raw arm. "
        f"Controls: {config['control_shots_per_preparation_treatment']:,} shots per preparation/treatment/job.", "",
        f"Source: `{source['kind']}`, backend `{source['backend']}`, **{config['observed_paths']} counted paths**. "
        f"Footprint expansion: **{config['footprint_expansion']}**. "+detail, "",
        "Instruction counts use [IBM's published heuristic]("+audit.LIMIT_SOURCE+"). Shots are summed separately; "
        "complete controls repeat in every job. All matched variants for a path stay together, with at least two paths per job.", "",
        "| Scenario | Jobs | Counting lower bound | Circuits | Shots including controls | Peak instructions/qubit | Within counted limits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    labels = dict(at_published_limit="No circuit cap, published instruction allowance",
        at_selected_fraction=f"No circuit cap, {config['instruction_fraction']:.0%} instruction allowance",
        observed_maximum_envelope=f"Observed maximum at every position, {config['instruction_fraction']:.0%}",
        current_runner_circuit_cap=f"Current runner cap: {config['current_runner_circuit_cap']} circuits/job")
    for name, result in report["scenarios"].items():
        lines.append(f"| {labels[name]} | {result['jobs']} | {result['lower_bound_jobs']} | {result['circuits']:,} | "
                     f"{result['shots']:,} | {result['maximum_instructions']:,} | {result['within_counted_limits']} |")
        with (folder/(name+"_jobs.csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["partitions"][0]))
            writer.writeheader()
            writer.writerows(result["partitions"])
    lines += ["", "The first two scenarios construct a contiguous greedy partition in the saved-seed path order. "
        "Equality with the lower bound establishes the minimum only for the supplied footprint/counting model. "
        "The observed-maximum scenario is a sensitivity calculation, not a bound on unseen Phoenix circuits. "
        "CSV path_start/path_stop are positions in the documented permutation, not physical trajectory IDs.", "",
        "The circuit-cap row reproduces the runner's balanced path grouping. If its limits check is false, the current "
        "runner would refuse submission; the number of jobs in that row is not a feasible hardware plan.", "",
        "**Planning only:** memory-packed partitions are not installed in the hardware runner. Neither the estimator "
        "nor these CSV files change its default 64-path pilot, 512-circuit cap, job IDs, source, QPY or manifest. "
        "A smaller projected job count does not imply that the full run is ready to submit.", "",
        "This does not estimate QPU seconds, dollars, compiler acceptance or statistical accuracy. All input archives "
        "remain read-only. Source hashes, weights, per-job shot/control totals and assumptions are in job_estimate.json.", ""]
    (folder/"JobEstimate.md").write_text("\n".join(lines), encoding="utf-8")


def main(arguments=None, *, backend=None, service=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--footprints", type=Path, help="portable footprint summary; default is the bundled FakeKingston sample")
    source.add_argument("--fake-run", type=Path, help="recount the saved Model2 FakeKingston pilot QPY")
    source.add_argument("--manifest", type=Path, help="recount a fully compiled Model2 Experiment6 mitigation manifest")
    source.add_argument("--backend", help="query this live IBM target and locally compile/count a Model2 sample; never submits")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--sample-trajectories", type=int, help="backend compilation size: default min(32, target); set equal to target for a full count")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--physical-path", type=int, nargs="+")
    mapping.add_argument("--reference-manifest", type=Path, help="reuse only the path of a matching raw Model2 Experiment5 run")
    parser.add_argument("--calibration-policy", choices=("screen", "record-only"), default="screen")
    parser.add_argument("--avoid-qubits", type=int, nargs="*", default=[])
    parser.add_argument("--min-t1-us", type=float, default=80.)
    parser.add_argument("--min-t2-us", type=float, default=60.)
    parser.add_argument("--max-cz-error", type=float, default=.005)
    parser.add_argument("--max-readout-error", type=float, default=.02)
    parser.add_argument("--trajectory-count", type=int, default=16384)
    parser.add_argument("--shots", type=int, default=32, help="total per trajectory per raw arm or twirled factor")
    parser.add_argument("--control-shots", type=int, default=2048)
    parser.add_argument("--instruction-fraction", type=float, default=.8)
    parser.add_argument("--circuit-cap", type=int, default=512, help="comparison with the current runner, not an IBM circuit limit")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"job_estimates")
    parser.add_argument("--export-footprints", type=Path, help="also save a portable hash-checked summary of the counted input")
    argv = sys.argv[1:] if arguments is None else list(arguments)
    options = parser.parse_args(argv)
    validate_request(options.trajectory_count, options.shots, options.control_shots, options.instruction_fraction, options.circuit_cap)
    backend_flags = {"--account-file", "--sample-trajectories", "--seed", "--seed-transpiler", "--optimization-level",
        "--physical-path", "--reference-manifest", "--calibration-policy", "--avoid-qubits",
        "--min-t1-us", "--min-t2-us", "--max-cz-error", "--max-readout-error"}
    if not options.backend and any(word.split("=", 1)[0] in backend_flags for word in argv):
        raise ValueError("backend compilation/mapping settings require --backend")
    if options.backend:
        if options.sample_trajectories is None:
            options.sample_trajectories = min(32, options.trajectory_count)
        if not 2 <= options.sample_trajectories <= options.trajectory_count:
            raise ValueError("sample-trajectories must be between two and the requested trajectory-count")
        if options.reference_manifest:
            input_root = options.reference_manifest.resolve().parent
            if (options.output_directory.resolve().is_relative_to(input_root) or
                    options.export_footprints and options.export_footprints.resolve().is_relative_to(input_root)):
                raise ValueError("estimation outputs must be outside the referenced raw hardware run")
        profile = profile_from_backend(options, backend=backend, service=service)
    else:
        input_path = options.fake_run or options.manifest or options.footprints or DEFAULT_PROFILE
        input_root = options.fake_run or (options.manifest.parent if options.manifest else input_path.parent)
        if options.output_directory.resolve().is_relative_to(input_root.resolve()):
            raise ValueError("output-directory must be outside the input archive/profile directory")
        if options.export_footprints and options.export_footprints.resolve().is_relative_to(input_root.resolve()):
            raise ValueError("export-footprints must be outside the input archive/profile directory")
    if options.fake_run:
        profile = profile_from_fake(options.fake_run)
    elif options.manifest:
        profile = profile_from_manifest(options.manifest)
    elif not options.backend:
        profile = json.loads(input_path.read_text())
    report = estimate(profile, trajectory_count=options.trajectory_count, shots=options.shots,
        control_shots=options.control_shots, instruction_fraction=options.instruction_fraction, circuit_cap=options.circuit_cap)
    if options.export_footprints:
        atomic_json(options.export_footprints, profile)
    save_report(report, options.output_directory)
    print(f"Planning {options.trajectory_count:,} trajectories; counted source: {profile['source']['backend']}, {len(profile['path_ids'])} paths.")
    for name, result in report["scenarios"].items():
        print(f"{name}: {result['jobs']} jobs, {result['circuits']:,} circuits, {result['shots']:,} shots; within counted limits={result['within_counted_limits']}")
    print("Estimate only; live backend queried for local compilation, no QPU jobs submitted or saved hardware runs changed."
          if options.backend else "Estimate only; no IBM account, backend query, submission or saved-run changes.")
    print(options.output_directory/"JobEstimate.md")
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
