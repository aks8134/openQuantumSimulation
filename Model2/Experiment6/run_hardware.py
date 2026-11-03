"""Live Model2 cosine44 population mitigation; default is an offline plan.

Only --execute submits. --resume alone remains offline. Native CZ twirling,
folding and balanced terminal masks reuse the tested Model3 protocol. The
frozen Model2 input, per-job controls, compiled QPY and job IDs are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from qiskit import qpy, transpile

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, instruction_audit, qpy_storage
from Model2.Experiment5 import run_hardware as raw
from Model3.Experiment5 import mitigation as mit, run_mitigation as protocol
from Model3.Experiment5.analyze_hardware import serializable

EXPERIMENT = "Model2/Experiment6/hardware_mitigation_v1"
SAMPLER_OPTIONS = protocol.SAMPLER_OPTIONS
CONTROLS = ("all_zero", "all_one", "initial_mixture")


def parse_arguments(arguments=None):
    argv = sys.argv[1:] if arguments is None else list(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_phoenix")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--spec", type=Path, default=input_data.DEFAULT_SPEC)
    parser.add_argument("--trajectory-count", type=int, default=64)
    parser.add_argument("--shots", type=int, default=256, help="total per path per arm/noise factor; split across randomizations")
    parser.add_argument("--randomizations", type=int, default=4)
    parser.add_argument("--control-shots", type=int, default=2048, help="total per preparation per raw/twirled treatment per job")
    parser.add_argument("--fold-factors", type=float, nargs="+", default=[1., 1.25, 1.5])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512, help="maximum circuits per Runtime job, including complete controls")
    parser.add_argument("--instruction-fraction", type=float, default=.8, help="fraction of the published per-qubit memory limit allowed per job")
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--physical-path", type=int, nargs="+", help="left bath, ten system sites, right bath")
    mapping.add_argument("--reference-manifest", type=Path, help="reuse only the selected path from a matching raw Model2 run")
    parser.add_argument("--calibration-policy", choices=("screen", "record-only"), default="screen")
    parser.add_argument("--avoid-qubits", type=int, nargs="*", default=[])
    parser.add_argument("--min-t1-us", type=float, default=80.)
    parser.add_argument("--min-t2-us", type=float, default=60.)
    parser.add_argument("--max-cz-error", type=float, default=.005)
    parser.add_argument("--max-readout-error", type=float, default=.02)
    parser.add_argument("--max-path-expansions", type=int, default=2_000_000)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--analyze-only", action="store_true", help="requires --resume; no account or submission")
    action.add_argument("--inspect-job", metavar="JOB_ID", help="requires --resume; read-only provider status/error/usage")
    parser.add_argument("--retry-failed-job", metavar="JOB_ID", help="explicit terminal ERROR/6056 retry; requires --resume --execute")
    parser.add_argument("--resume", type=Path, metavar="MANIFEST")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    parser.add_argument("--analysis-output-directory", type=Path)
    options = parser.parse_args(argv)
    if (options.analyze_only or options.inspect_job) and not options.resume:
        raise ValueError("analysis/inspection requires --resume MANIFEST")
    if options.retry_failed_job and not (options.resume and options.execute):
        raise ValueError("retry requires --resume MANIFEST --execute")
    if options.resume:
        allowed = {"--resume", "--execute", "--compile-only", "--plan-only", "--analyze-only", "--inspect-job",
                   "--retry-failed-job", "--account-file", "--analysis-output-directory"}
        if any(word.split("=", 1)[0] not in allowed for word in argv if word.startswith("--")):
            raise ValueError("resume uses saved scientific/run settings; only action, account and analysis output may change")
        return options
    if (not 2 <= options.trajectory_count <= 16384 or min(options.seed, options.seed_transpiler) < 0
            or options.bootstrap < 100 or not options.backend.strip() or options.randomizations < 2
            or options.randomizations % 2 or min(options.shots, options.control_shots) < 2*options.randomizations
            or options.shots % options.randomizations or options.control_shots % options.randomizations):
        raise ValueError("invalid paths, seeds, bootstrap, backend or shots/randomizations")
    factors = options.fold_factors
    if (len(factors) < 3 or factors[0] != 1 or any(not np.isfinite(f) or not 1 <= f <= 3 for f in factors)
            or any(b <= a for a, b in zip(factors, factors[1:]))):
        raise ValueError("need >=3 increasing CZ factors from 1 through at most 3")
    if options.batch_size < 3*(1+options.randomizations)+2*(1+options.randomizations*len(factors)):
        raise ValueError("batch-size must fit controls and at least two complete matched paths")
    if not np.isfinite(options.instruction_fraction) or not 0 < options.instruction_fraction <= 1:
        raise ValueError("instruction-fraction must lie in (0, 1]")
    if (options.physical_path is not None and (len(options.physical_path) != 12
            or len(set(options.physical_path)) != 12 or min(options.physical_path) < 0)):
        raise ValueError("physical-path needs twelve distinct nonnegative qubits")
    settings = [options.min_t1_us, options.min_t2_us, options.max_cz_error, options.max_readout_error]
    if (not np.all(np.isfinite(settings)) or min(settings[:2]) <= 0 or any(not 0 < x < 1 for x in settings[2:])
            or min(options.avoid_qubits, default=0) < 0 or options.max_path_expansions < 1):
        raise ValueError("invalid calibration settings")
    if options.calibration_policy == "record-only" and not (options.physical_path or options.reference_manifest):
        raise ValueError("record-only requires an explicitly fixed or reference path")
    return options


def versions():
    return {name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime", "numpy")}


def records_for(config):
    records, batches = protocol.plan({**config, "physical_path": config["physical_path"] or list(range(12))})
    for index, batch in enumerate(batches):
        for row in records[batch["start"]:batch["stop"]]:
            row["batch"] = index
    return records, batches


def plan_hash(manifest):
    return input_data.digest({"experiment": manifest["experiment"], "configuration": manifest["configuration"],
        "circuits": manifest["circuits"], "compilation": [{k: b[k] for k in
        ("start", "stop", "compiled", "isa_file", "isa_sha256", "resources", "compiler_versions") if k in b}
        for b in manifest["batches"]]})


def checkpoint(manifest, folder):
    manifest["plan_sha256"] = plan_hash(manifest)
    atomic_json(folder/"manifest.json", manifest)


def reference_path(path, backend, seed, spec):
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != raw.EXPERIMENT or manifest.get("plan_sha256") != raw.plan_hash(manifest):
        raise ValueError("reference is not an intact raw Model2 hardware manifest")
    config = manifest["configuration"]
    if (config["backend"] != backend or config["seed"] != seed or config["spec_sha256"] != input_data.digest(spec)
            or config["physics"] != spec["physics"] or config["physical_path"] is None):
        raise ValueError("reference backend, seed, frozen physics/spec or selected path differs")
    return config["physical_path"], {"manifest": str(path.resolve()), "plan_sha256": manifest["plan_sha256"],
                                    "scope": "Physical path only; no QPY, results, jobs or calibration responses reused."}


def new_plan(options):
    spec = input_data.load_spec(options.spec)
    requested, reference = options.physical_path, None
    if options.reference_manifest:
        requested, reference = reference_path(options.reference_manifest, options.backend, options.seed, spec)
    source = input_data.make_source(spec, options.trajectory_count, options.seed)
    config = {key: getattr(options, key) for key in ("backend", "trajectory_count", "shots", "randomizations",
        "control_shots", "fold_factors", "seed", "seed_transpiler", "optimization_level", "batch_size", "bootstrap",
        "instruction_fraction", "calibration_policy")}
    config.update(physics=spec["physics"], steps=44, controls_per_batch=True, physical_path=None,
        requested_physical_path=requested, reference=reference, source_sha256=None,
        spec_sha256=input_data.digest(spec), sampler_options=json.loads(json.dumps(SAMPLER_OPTIONS)), software_versions=versions(),
        path_settings={"avoid_qubits": options.avoid_qubits, "min_t1_seconds": options.min_t1_us*1e-6,
            "min_t2_seconds": options.min_t2_us*1e-6, "max_cz_error": options.max_cz_error,
            "max_readout_error": options.max_readout_error, "max_expansions": options.max_path_expansions},
        scope="Fixed first R seed streams, half-mixed preparation, population-only mitigation pilot; no 0.02 certification.")
    records, batches = records_for(config)
    config.update(total_circuits=len(records), total_shots=sum(r["shots"] for r in records), total_jobs=len(batches))
    manifest = dict(experiment=EXPERIMENT, status="planned", configuration=config, circuits=records, batches=batches)
    verify_manifest(manifest, source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = options.output_directory/f"mitigation44_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/"source.json", source)
    atomic_json(folder/"spec.json", spec)
    config["source_sha256"] = qpy_storage.file_sha256(folder/"source.json")
    checkpoint(manifest, folder)
    write_plan(manifest, source, folder)
    return folder, manifest, source


def verify_manifest(manifest, source=None):
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("not a Model2 mitigation manifest")
    config = manifest["configuration"]
    if (config["steps"] != 44 or config["spec_sha256"] != input_data.SPEC_SHA256
            or config["sampler_options"] != SAMPLER_OPTIONS or config["controls_per_batch"] is not True
            or not 2 <= config["trajectory_count"] <= 16384):
        raise ValueError("frozen experiment or explicit manual mitigation options changed")
    path = config["physical_path"]
    if path is not None and (len(path) != 12 or len(set(path)) != 12 or min(path) < 0):
        raise ValueError("invalid saved path")
    expected, batches = records_for(config)
    if len(expected) != len(manifest["circuits"]) or len(batches) != len(manifest["batches"]):
        raise ValueError("incomplete matched mitigation plan")
    for planned, saved in zip(expected, manifest["circuits"]):
        if any(saved.get(key) != value for key, value in planned.items()):
            raise ValueError("mitigation record, seed order, factor, mask or shots changed")
    for planned, saved in zip(batches, manifest["batches"]):
        if (saved["start"], saved["stop"]) != (planned["start"], planned["stop"]):
            raise ValueError("batch coverage changed")
        if saved["status"] not in ("not_submitted", "submitting", "submitted", "collected"):
            raise ValueError("unknown batch submission state")
        if saved["status"] in ("submitted", "collected") and not saved.get("job_id"):
            raise ValueError("submitted job has no saved ID; reconcile it before resuming")
        if sum(r["shots"] for r in expected[saved["start"]:saved["stop"]]) > instruction_audit.EXECUTION_LIMIT:
            raise ValueError("job exceeds the 10-million-execution limit; reduce shots or batch-size")
        if saved["compiled"] and (path is None or not saved.get("isa_file") or not saved.get("isa_sha256") or not saved.get("resources")):
            raise ValueError("compiled batch lacks its path, archive or resource assessment")
    if (config["total_circuits"] != len(expected) or config["total_jobs"] != len(batches)
            or config["total_shots"] != sum(r["shots"] for r in expected)):
        raise ValueError("workload totals changed")
    if source is not None:
        spec = input_data.load_spec()
        if (source.get("experiment") != "Model2/Experiment5" or source["configuration"]["physics"] != spec["physics"]
                or config["physics"] != spec["physics"] or source["configuration"]["trajectories"] != config["trajectory_count"]
                or source["configuration"]["seed"] != config["seed"] or source["preparation"] != spec["preparation"]
                or source["schedule"]["step_dts"] != spec["schedule"]["step_dts"]
                or source["configuration"]["order"] != "jump-first"):
            raise ValueError("source does not match the frozen Model2 physics, paths or preparation")


def read_saved(path, *, require_circuits=False):
    folder = path.resolve().parent
    manifest = json.loads(path.read_text())
    if manifest.get("plan_sha256") != plan_hash(manifest):
        raise ValueError("mitigation plan/compilation hash mismatch")
    config = manifest["configuration"]
    if qpy_storage.file_sha256(folder/"source.json") != config["source_sha256"]:
        raise ValueError("source hash mismatch")
    if input_data.digest(input_data.load_spec(folder/"spec.json")) != config["spec_sha256"]:
        raise ValueError("specification hash mismatch")
    with (folder/"source.json").open() as handle:
        source = json.load(handle)
    verify_manifest(manifest, source)
    for batch in manifest["batches"]:
        if not batch["compiled"]:
            continue
        archive = folder/batch["isa_file"]
        if not archive.exists():
            if require_circuits:
                raise ValueError("compiled QPY missing; restore the recorded archive before execution")
        elif qpy_storage.file_sha256(archive) != batch["isa_sha256"]:
            raise ValueError("native QPY hash mismatch")
    return folder, manifest, source


def write_plan(manifest, source, folder):
    config = manifest["configuration"]
    base = 880
    factors = [1+2*mit.fold_count(f, base)/base for f in config["fold_factors"]]
    lines = ["# Model2 cosine44 hardware mitigation plan", "",
        f"Backend: `{config['backend']}`. Status: `{manifest['status']}`.", "",
        f"**{config['trajectory_count']} paths × {config['shots']} shots per arm/factor; "
        f"{config['total_jobs']} jobs, {config['total_circuits']:,} circuits, {config['total_shots']:,} shots including controls.**", "",
        "N=10, J=1, Γ/J=2, pL=pR=0.25, exchange prefactor 1/4, Δ=0; half-mixed preparation. "
        "The frozen 44-step warped cosine, jump-first order and even/odd exchange layers are unchanged.", "",
        f"Raw plus twirled factors {factors}; {config['randomizations']} randomizations split each factor into "
        f"{config['shots']//config['randomizations']} shots per variant. "
        f"Zero, one and held-out half-mixture controls each receive {config['control_shots']} raw and "
        f"{config['control_shots']} twirled shots inside every job.", "",
        "Native CZ→CZ·CZ·CZ folding and Pauli frames preserve the intentional bath channel. All 44 resets remain. "
        "Balanced terminal X masks are decoded from final outcome counts; preparation bits are ignored. "
        "Only one-qubit frame optimization follows folding. Runtime twirling and DD are disabled; initialization is enabled.", "",
        f"Selected path: `{config['physical_path']}`. Requested/reference path: `{config['requested_physical_path']}`. "
        f"Calibration policy: `{config['calibration_policy']}`.", "",
        f"Exact finite-path assessment: `{source['finite_path_assessment']}`", "",
        "This is a population-only mitigation comparison. It does not measure currents or certify the 0.02 target. "
        "All six predefined profiles and descriptive uncertainty are reported without clipping or target fitting. "
        "SPAM corrections use each job's own calibration before weighted averaging; the mixture is held out.", "",
        f"Each compiled job is checked against {config['instruction_fraction']:.0%} of the published "
        "26.8-million-instructions-per-qubit limit and the 10-million-execution limit. These are validation heuristics, "
        "not proof of provider compiler acceptance. All jobs are compiled before the first submission.", "",
        "Default invocation and plain --resume are offline. --compile-only queries the backend without submitting. "
        "Only --execute submits; known job IDs are retrieved, and ambiguous submissions are never automatically repeated.", ""]
    (folder/"Plan.md").write_text("\n".join(lines), encoding="utf-8")


def select_path(manifest, source, backend, folder):
    config = manifest["configuration"]
    if backend.name != config["backend"]:
        raise ValueError("backend differs from the saved mitigation plan")
    settings = config["path_settings"]
    fixed = config["physical_path"] or config["requested_physical_path"]
    if fixed is None:
        selected = raw.quality_path.select(backend, source, list(range(config["trajectory_count"])),
            avoid=settings["avoid_qubits"], min_t1=settings["min_t1_seconds"], min_t2=settings["min_t2_seconds"],
            max_cz_error=settings["max_cz_error"], max_readout_error=settings["max_readout_error"],
            max_expansions=settings["max_expansions"])
    elif config["calibration_policy"] == "screen":
        selected = raw.quality_path.validate(backend, fixed, source, list(range(config["trajectory_count"])), settings)
    else:
        raw.circuits.chain_layout(backend, 10, "boundary", fixed)
        selected = {"physical_path": fixed, "policy": "record-only", "scope": "Explicit fixed path retained; quality thresholds not applied."}
    config["physical_path"] = selected["physical_path"]
    if not (folder/"selected_path.json").exists():
        atomic_json(folder/"selected_path.json", selected)
    if not (folder/"backend_snapshot.json").exists():
        raw.runtime.save_backend_snapshot(backend, config["physical_path"], manifest["circuits"], folder)
    checkpoint(manifest, folder)
    print(f"Mitigation physical path: {config['physical_path']}", flush=True)


def validate_circuits(manifest, programs, records, backend=None):
    config = manifest["configuration"]
    layout = config["physical_path"][1:-1]+[config["physical_path"][0], config["physical_path"][-1]]
    if len(programs) != len(records):
        raise ValueError("native batch count mismatch")
    for program, row in zip(programs, records):
        metadata = program.metadata or {}
        if any(metadata.get(k) != v for k, v in {"experiment": EXPERIMENT, "pub_index": row["pub_index"],
                "batch": row["batch"], "mitigation_case_id": row["case_id"], "kind": row["kind"]}.items()):
            raise ValueError("native mitigation circuit identity changed")
        control = row["kind"] == "control"
        if metadata.get("steps") != (0 if control else 44) or metadata.get("physical_sites") != layout[:10]:
            raise ValueError("native steps or physical readout mapping changed")
        identity = "control" if control else "trajectory"
        if metadata.get(identity) != row[identity]:
            raise ValueError("native preparation/trajectory identity changed")
        counts = program.count_ops()
        if counts.get("cz", 0) != (0 if control else 880+2*row["folds"]) or counts.get("reset", 0) != (0 if control else 44):
            raise ValueError("native CZ fold or intentional bath reset count changed")
        if row["arm"] == "twirled":
            details = metadata.get("mitigation", {})
            if (details.get("measurement_mask") != row["measurement_mask"] or details.get("fold_count") != row.get("folds", 0)
                    or details.get("seed") != protocol.transform_seed(config, row)):
                raise ValueError("native twirl mask or folding metadata changed")
        outcome = next((r for r in program.cregs if r.name == "outcome"), None)
        if outcome is None or len(outcome) != 10:
            raise ValueError("native circuit lacks a ten-bit final outcome register")
        measured = [(program.find_bit(item.qubits[0]).index, list(outcome).index(item.clbits[0]))
                    for item in program.data if item.operation.name == "measure" and item.clbits[0] in outcome]
        if measured != list(zip(layout[:10], range(10))):
            raise ValueError("native final measurements differ from the fixed mapping")
        if raw.circuits.metrics(program) != row["metrics"]:
            raise ValueError("native circuit metrics changed")
        if backend is not None:
            protocol.assert_supported(program, backend)


def resource_check(programs, records, config):
    total = None
    for program in programs:
        value = instruction_audit.footprint(program)
        total = value.copy() if total is None else total+value
    limit = int(config["instruction_fraction"]*instruction_audit.INSTRUCTION_LIMIT)
    executions = sum(r["shots"] for r in records)
    if total is None or total.max() > limit or executions > instruction_audit.EXECUTION_LIMIT:
        raise ValueError("compiled job exceeds the configured instruction/execution budget; reduce batch-size")
    return {"instructions_per_qubit": total.tolist(), "maximum_instructions": int(total.max()),
            "bottleneck_qubit": int(total.argmax()), "instruction_budget": limit, "executions": executions,
            "scope": "Published per-qubit weighted instruction heuristic, including implicit init; shots checked separately."}


class Publications:
    def __init__(self, manifest, folder, backend=None):
        self.manifest, self.folder, self.backend = manifest, folder, backend

    def __len__(self):
        return len(self.manifest["circuits"])

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("load one complete Runtime job slice")
        batch = next((b for b in self.manifest["batches"] if (b["start"], b["stop"]) == (selection.start, selection.stop)), None)
        if batch is None or not batch["compiled"]:
            raise ValueError("requested job is not completely compiled")
        records = self.manifest["circuits"][batch["start"]:batch["stop"]]
        path = self.folder/batch["isa_file"]
        if qpy_storage.file_sha256(path) != batch["isa_sha256"]:
            raise ValueError("native QPY hash mismatch")
        archive = qpy_storage.Archive(path, len(records))
        programs, _ = next(archive.iter_batches([dict(start=0, stop=len(records))]))
        validate_circuits(self.manifest, programs, records, self.backend)
        if resource_check(programs, records, self.manifest["configuration"]) != batch["resources"]:
            raise ValueError("native resource assessment changed")
        return programs


def compile_batches(manifest, source, backend, folder):
    config = manifest["configuration"]
    layout = config["physical_path"][1:-1]+[config["physical_path"][0], config["physical_path"][-1]]
    manifest["status"] = "compiling"
    checkpoint(manifest, folder)
    store = Publications(manifest, folder, backend)
    for index, batch in enumerate(manifest["batches"], 1):
        if batch.get("observations") is not None or batch.get("job_id"):
            continue
        if batch["compiled"]:
            programs = store[batch["start"]:batch["stop"]]
            del programs
            continue
        records = manifest["circuits"][batch["start"]:batch["stop"]]
        last_use = {protocol.template_key(row): j for j, row in enumerate(records)}
        templates, programs = {}, []
        running = None
        for position, row in enumerate(records):
            key = protocol.template_key(row)
            if key not in templates:
                logical = raw.build_body(source, 0, steps=0, populations=row["expected_populations"]) if row["kind"] == "control" else raw.build_body(source, row["trajectory"])
                native = transpile(logical, backend=backend, initial_layout=layout,
                    optimization_level=config["optimization_level"], seed_transpiler=config["seed_transpiler"])
                if native.layout.final_index_layout(filter_ancillas=True) != layout:
                    raise ValueError("transpiler changed the fixed physical mapping")
                templates[key] = raw.circuits.append_readout(native, layout[:10], "z")
            program = protocol.make_variant(templates[key], config, row)
            program.metadata.update(experiment=EXPERIMENT, pub_index=row["pub_index"], batch=row["batch"], kind=row["kind"])
            if row["kind"] == "control":
                program.metadata["control"] = row["control"]
            row.update(metrics=raw.circuits.metrics(program), final_logical_to_physical=list(layout))
            protocol.assert_supported(program, backend)
            value = instruction_audit.footprint(program)
            running = value.copy() if running is None else running+value
            if running.max() > int(config["instruction_fraction"]*instruction_audit.INSTRUCTION_LIMIT):
                raise ValueError("compiled job exceeds the configured instruction budget; reduce batch-size")
            programs.append(program)
            if position == last_use[key]:
                del templates[key]
            if (position+1) % 64 == 0 or position+1 == len(records):
                print(f"Compiled job {index}/{len(manifest['batches'])}: {position+1}/{len(records)} variants", flush=True)
        validate_circuits(manifest, programs, records, backend)
        resources = resource_check(programs, records, config)
        name = f"circuits_{batch['start']:06d}_{batch['stop']:06d}.qpy"
        temporary = folder/(name+".partial")
        with temporary.open("wb") as handle:
            qpy.dump(programs, handle)
        del programs, program, native
        temporary.replace(folder/name)
        batch.update(compiled=True, isa_file=name, isa_sha256=qpy_storage.file_sha256(folder/name),
                     resources=resources, compiler_versions=versions())
        checkpoint(manifest, folder)
    manifest["status"] = "compiled"
    checkpoint(manifest, folder)
    write_plan(manifest, source, folder)
    return store


def inspect_job(options, service=None, backend=None):
    folder = options.resume.resolve().parent
    manifest = json.loads(options.resume.read_text())
    if manifest.get("plan_sha256") != plan_hash(manifest):
        raise ValueError("mitigation plan hash mismatch")
    verify_manifest(manifest)
    service = service or raw.runtime.make_service(options)
    _, details = raw.job_diagnostic(manifest, service, options.inspect_job)
    backend = backend or service.backend(manifest["configuration"]["backend"], use_fractional_gates=False)
    details["backend_status"] = raw.backend_readiness(backend)
    atomic_json(folder/"last_job_diagnostic.json", details)
    print(json.dumps(details, indent=2, default=str), flush=True)
    print("Read-only mitigation job inspection; no jobs submitted or reset.", flush=True)
    return details


def bundle(folder, manifest, output=None):
    names = ["manifest.json", "source.json", "spec.json", "Plan.md", "selected_path.json", "backend_snapshot.json",
             "execution_backend_snapshot.json", "hardware_results.json", "last_job_diagnostic.json"]
    names += [batch["isa_file"] for batch in manifest["batches"] if batch.get("isa_file")]
    with zipfile.ZipFile(folder/"results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(names):
            if (folder/name).exists():
                archive.write(folder/name, arcname=name)
        if output is not None:
            for path in output.iterdir():
                if path.is_file():
                    archive.write(path, arcname="analysis/"+path.name)


def finish_analysis(folder, manifest, output=None):
    from Model2.Experiment6.analyze_hardware import analyze
    report = analyze(folder)
    output = output or DIRECTORY/"analysis"/folder.name
    protocol.save_analysis(report, output)
    atomic_json(folder/"hardware_results.json", serializable(report))
    bundle(folder, manifest, output)
    print(f"Analysis: {output/'Analysis.md'}\nTransfer: {folder/'results_bundle.zip'}", flush=True)
    return report


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.inspect_job:
        return inspect_job(options, service, backend)
    if options.resume:
        folder, manifest, source = read_saved(options.resume)
    else:
        folder, manifest, source = new_plan(options)
    config = manifest["configuration"]
    if options.analysis_output_directory and options.analysis_output_directory.resolve().is_relative_to(folder.resolve()):
        raise ValueError("analysis output must be outside the raw run directory")
    print(f"Plan: {folder/'manifest.json'}", flush=True)
    print(f"Mitigation: {config['total_jobs']} jobs, {config['total_circuits']:,} circuits, {config['total_shots']:,} shots", flush=True)
    print(f"Finite-path assessment: {source['finite_path_assessment']}", flush=True)
    if options.analyze_only:
        return finish_analysis(folder, manifest, options.analysis_output_directory)
    if not options.execute and not options.compile_only:
        print("Offline plan only; no account loaded, backend queried, or jobs submitted.", flush=True)
        return manifest
    if manifest["status"] == "collected":
        if options.retry_failed_job:
            raise ValueError("collected mitigation runs cannot be retried")
        return finish_analysis(folder, manifest, options.analysis_output_directory)
    if any(b["status"] == "submitting" and not b.get("job_id") for b in manifest["batches"]):
        raise ValueError("submission interrupted before its job ID was saved; reconcile with IBM; no automatic retry")
    retry = None
    if options.retry_failed_job:
        service = service or raw.runtime.make_service(options)
        retry = raw.job_diagnostic(manifest, service, options.retry_failed_job)
        batch, details = retry
        if (batch.get("observations") is not None or details["status"] != "ERROR"
                or not re.search(r"\b6056\b", str(details["error_message"]))):
            raise ValueError("retry refused: need a saved terminal ERROR/6056 job without collected results")
        if not batch["compiled"] or not (folder/batch["isa_file"]).exists():
            raise ValueError("restore the compiled QPY archive before retrying this job")
    needs_new = retry is not None or any(b.get("observations") is None and not b.get("job_id") for b in manifest["batches"])
    for batch in manifest["batches"]:
        if batch["compiled"] and not batch.get("job_id") and batch.get("observations") is None and not (folder/batch["isa_file"]).exists():
            raise ValueError("compiled QPY missing; restore the recorded archive before execution")
    if needs_new:
        if backend is None:
            service = service or raw.runtime.make_service(options)
            backend = service.backend(config["backend"], use_fractional_gates=False)
        if options.execute:
            raw.backend_readiness(backend, require_ready=True)
        select_path(manifest, source, backend, folder)
        publications = compile_batches(manifest, source, backend, folder)
    else:
        publications = Publications(manifest, folder, backend)
    if options.compile_only:
        bundle(folder, manifest)
        print(f"Compiled only; no jobs submitted. Execute with --resume {folder/'manifest.json'} --execute", flush=True)
        return manifest
    if service is None and sampler_factory is None:
        service = raw.runtime.make_service(options)
    if backend is not None:
        snapshot = folder/"execution_backend_snapshot.json"
        # Use a separate temporary directory so the compilation snapshot remains intact.
        import tempfile
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            raw.runtime.save_backend_snapshot(backend, config["physical_path"], manifest["circuits"], Path(temporary))
            atomic_json(snapshot, json.loads((Path(temporary)/"backend_snapshot.json").read_text()))
    if retry is not None:
        raw.backend_readiness(backend, require_ready=True)
        batch, details = retry
        batch.setdefault("failed_attempts", []).append(details)
        batch.update(job_id=None, status="not_submitted")
        checkpoint(manifest, folder)
        print(f"Explicit retry of {options.retry_failed_job}; previous ID/error/usage retained.", flush=True)
    from qiskit_ibm_runtime.exceptions import RuntimeJobFailureError
    try:
        raw.runtime.collect_and_submit(manifest, folder, publications, service, backend, sampler_factory,
                                       before_submit=lambda: raw.backend_readiness(backend, require_ready=True))
    except RuntimeJobFailureError as error:
        batch = next(b for b in manifest["batches"] if b.get("job_id") and b.get("observations") is None)
        batch["last_result_error"] = str(error)
        checkpoint(manifest, folder)
        raise ValueError(f"IBM failed mitigation job {batch['job_id']}; saved ID retained. Use --resume MANIFEST --inspect-job JOB_ID; no automatic resubmission. {error}") from error
    return finish_analysis(folder, manifest, options.analysis_output_directory)


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
