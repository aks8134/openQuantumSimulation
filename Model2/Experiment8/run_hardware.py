"""Model2 cosine44: manual CZ/measurement twirling, ZNE, job-local SPAM, and XX DD.

Every submitted circuit uses the DD policy. There is no DD-off hardware arm.
plan is offline; prepare compiles; only submit sends explicit dry-run/hardware
jobs. collect, render, analyze, and bundle are separate resumable stages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

import numpy as np
from qiskit import qpy, transpile
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, instruction_audit, qpy_storage
from Model2.Experiment5 import run_hardware as raw
from Model2.Experiment6 import run_hardware as base
from Model2.Experiment6.progress import progress
from Model2.Experiment7 import run_hardware as dd_runtime, schedule_tools as timing
from Model3.Experiment5 import run_mitigation as protocol
from Model2.Experiment8.ordering import flat_block_order

EXPERIMENT = "Model2/Experiment8/mitigation_dd_v1"
MAX_TRAJECTORIES = 16834
MAX_SHARD_CIRCUITS = 13
MAX_SHARD_BYTES = 64*1024*1024
DEFAULT_REFERENCE = ROOT/"Model2/Experiment6/hardware_runs/mitigation44_20261103T145657_688162Z/results_bundle.zip"


def batch_time(value):
    from qiskit_ibm_runtime.utils.converters import hms_to_seconds
    try:
        seconds = int(value) if value.isdecimal() else hms_to_seconds(value, "Invalid batch-max-time: ")
        if seconds <= 0:
            raise ValueError("batch-max-time must be positive")
        return seconds
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def sampler_options(capture_timing):
    options = json.loads(json.dumps(base.SAMPLER_OPTIONS))
    if capture_timing:
        options["experimental"] = {"execution": {"scheduler_timing": True}}
    return options


def parse_arguments(arguments=None):
    argv = sys.argv[1:] if arguments is None else list(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "prepare"):
        p = sub.add_parser(action)
        p.add_argument("--backend", default="ibm_phoenix")
        p.add_argument("--trajectory-count", type=int, default=16384)
        p.add_argument("--shots", type=int, default=32, help="per path per arm/factor; four twirled variants share this budget")
        p.add_argument("--control-shots", type=int, default=2048)
        p.add_argument("--batch-size", type=int, default=512, help="circuit cap including controls; timing inspection also caps jobs at 128")
        p.add_argument("--bootstrap", type=int, default=2000)
        p.add_argument("--instruction-fraction", type=float, default=.8)
        p.add_argument("--execution-mode", choices=("job", "batch"), default="job",
                       help="hardware submission mode; batch queues all jobs in one Runtime Batch")
        p.add_argument("--batch-max-time", type=batch_time,
                       help="optional Runtime Batch TTL, e.g. 2h; defaults to the provider/account limit")
        p.add_argument("--capture-timing", action="store_true", help="small inspection cohort only (<=64 paths); IBM scheduler metadata")
        p.add_argument("--physical-path", type=int, nargs=12)
        p.add_argument("--calibration-policy", choices=("screen", "record-only"), default="screen")
        p.add_argument("--historical-reference", type=Path, default=DEFAULT_REFERENCE,
                       help="Experiment6 result bundle or analysis JSON; used only as historical context")
        p.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
        p.add_argument("--account-file", type=Path)
        if action == "prepare":
            p.add_argument("--resume", type=Path, help="continue this compiled/partially compiled manifest with its saved settings")
    for action in ("submit", "collect", "render", "analyze", "bundle", "clone-for-hardware", "inspect-job"):
        p = sub.add_parser(action)
        p.add_argument("--manifest", required=True, type=Path)
        if action in ("submit", "collect", "inspect-job"):
            p.add_argument("--account-file", type=Path)
        if action == "submit":
            mode = p.add_mutually_exclusive_group(required=True)
            mode.add_argument("--dry-run", action="store_true")
            mode.add_argument("--hardware", action="store_true")
            p.add_argument("--new-batch-for-remaining", action="store_true",
                           help="explicit recovery only: replace a non-accepting Runtime Batch for jobs without IDs")
            p.add_argument("--batch-max-time", type=batch_time,
                           help="optional TTL override when creating a Runtime Batch; does not require recompilation")
        if action == "collect":
            p.add_argument("--job-details", action="store_true", help="optional metrics after counts have been checkpointed")
        if action == "render":
            p.add_argument("--trajectory", type=int, default=0)
            p.add_argument("--arm", choices=("raw", "twirled"), default="twirled")
            p.add_argument("--factor-index", type=int, choices=(0, 1, 2), help="omit to inspect all three factors")
            p.add_argument("--randomization", type=int, choices=range(4), default=0)
            p.add_argument("--view", choices=("full", "active"), default="full")
            p.add_argument("--overwrite", action="store_true")
        if action == "analyze":
            p.add_argument("--output-directory", type=Path)
        if action == "bundle":
            p.add_argument("--include-qpy", action="store_true")
        if action == "clone-for-hardware":
            p.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
        if action == "inspect-job":
            p.add_argument("--job-id", required=True)
    a = parser.parse_args(argv)
    if a.action == "prepare" and a.resume:
        if any(word.split("=", 1)[0] not in ("--resume", "--account-file") for word in argv if word.startswith("--")):
            parser.error("prepare --resume uses all saved settings; only --account-file may be supplied")
    if a.action in ("plan", "prepare") and not getattr(a, "resume", None):
        if (not 2 <= a.trajectory_count <= MAX_TRAJECTORIES or a.shots < 8 or a.shots % 4
                or a.control_shots < 8 or a.control_shots % 4 or a.batch_size < 41 or a.bootstrap < 100
                or not np.isfinite(a.instruction_fraction) or not 0 < a.instruction_fraction <= 1
                or not a.backend.strip()):
            parser.error(f"need 2..{MAX_TRAJECTORIES} paths, shots/control-shots divisible by four and >=8, cap>=41, bootstrap>=100, fraction in (0,1]")
        if a.batch_max_time is not None and a.execution_mode != "batch":
            parser.error("--batch-max-time requires --execution-mode batch")
        if a.capture_timing and a.trajectory_count > 64:
            parser.error("capture-timing is restricted to <=64 inspection paths; full production saves counts without server timing")
        if a.physical_path and (len(set(a.physical_path)) != 12 or min(a.physical_path) < 0):
            parser.error("physical-path requires twelve distinct nonnegative qubits")
    if a.action == "render" and a.trajectory < 0:
        parser.error("trajectory must be nonnegative")
    if a.action == "submit" and a.new_batch_for_remaining and not a.hardware:
        parser.error("--new-batch-for-remaining is for hardware Batch recovery only")
    return a


def read_reference(path):
    """Save a small immutable reference instead of reopening the old run later."""
    if not path.is_file():
        raise ValueError(f"historical reference missing: {path}; supply --historical-reference from Experiment6")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            name = "analysis/analysis.json" if "analysis/analysis.json" in archive.namelist() else "hardware_results.json"
            with archive.open(name) as handle:
                report = json.load(handle)
    else:
        with path.open() as handle:
            report = json.load(handle)
    c = report["configuration"]
    if (report.get("experiment") != base.EXPERIMENT or c["spec_sha256"] != input_data.SPEC_SHA256
            or not report.get("analysis", {}).get("complete") or len(report["job_ids"]) != c["total_jobs"]):
        raise ValueError("historical reference must be a complete frozen Experiment6 analysis")
    return dict(artifact_type="model2_historical_mitigation_reference_v1", original_file=str(path.resolve()),
        file_sha256=qpy_storage.file_sha256(path), configuration=c, job_ids=report["job_ids"],
        profiles=report["statistics"]["profiles"], ideal_same_paths=report["ideal_same_paths"],
        physical_target=report["physical_target"],
        scope="Historical DD-off data only; different execution dates/calibration, no paired DD-effect inference.")


def static_plan(m):
    return dict(experiment=m["experiment"], configuration=m["configuration"], files=m["files"], circuits=m["circuits"],
        batches=[{k:b[k] for k in ("start", "stop", "compiled", "isa_file", "isa_sha256", "resources", "compiler_versions", "tag") if k in b}
                 for b in m["batches"]])


def checkpoint(folder, m):
    m["plan_sha256"] = input_data.digest(static_plan(m))
    atomic_json(folder/"manifest.json", m)


def register(folder, m, name, **details):
    if Path(name).name != name:
        raise ValueError("experiment assets must be local filenames")
    m["files"][name] = dict(sha256=qpy_storage.file_sha256(folder/name), **details)


def verify_manifest(m, source=None):
    c = m["configuration"]
    base.verify_manifest(m, source, expected_experiment=EXPERIMENT,
                         expected_sampler_options=sampler_options(c["capture_timing"]),
                         maximum_trajectories=MAX_TRAJECTORIES)
    if c.get("execution_mode", "job") not in ("job", "batch"):
        raise ValueError("unknown saved execution mode")
    maximum = c.get("batch_max_time_seconds")
    if maximum is not None and (c.get("execution_mode", "job") != "batch"
                                or not isinstance(maximum, int) or maximum <= 0):
        raise ValueError("invalid saved Runtime Batch maximum time")
    if (c["randomizations"] != 4 or c["fold_factors"] != [1., 1.25, 1.5] or c["seed"] != 29
            or c["seed_transpiler"] != 11 or c["optimization_level"] != 3 or c["dd_condition"] != "on"
            or c["dd_policy"] != dict(sequence="XX", spacings=[.25, .5, .25], min_idle_ratio=10.,
                system_sites_1based=list(range(1, 11)), skip_reset_qubits=True, insert_multiple_cycles=False)):
        raise ValueError("frozen manual mitigation/DD protocol changed")
    if any(row.get("dd_condition") != "on" for row in m["circuits"]):
        raise ValueError("Experiment8 may contain only DD-on hardware circuits")
    if c["capture_timing"] and (c["trajectory_count"] > 64 or c["batch_size"] > 128):
        raise ValueError("timing cohort exceeds bounded inspection limits")


def read_saved(path, *, native_files=False):
    folder = Path(path).resolve().parent
    with Path(path).open() as handle:
        m = json.load(handle)
    if m.get("plan_sha256") != input_data.digest(static_plan(m)):
        raise ValueError("Experiment8 plan/asset hash mismatch")
    for name in ("source.json", "spec.json", "historical_reference.json"):
        if qpy_storage.file_sha256(folder/name) != m["files"][name]["sha256"]:
            raise ValueError(f"saved input changed: {name}")
    with (folder/"source.json").open() as handle:
        source = json.load(handle)
    input_data.load_spec(folder/"spec.json")
    if m["configuration"]["source_sha256"] != m["files"]["source.json"]["sha256"]:
        raise ValueError("source identity changed")
    verify_manifest(m, source)
    if native_files:
        for name, info in m["files"].items():
            if qpy_storage.file_sha256(folder/name) != info["sha256"]:
                raise ValueError(f"compiled asset changed: {name}")
    return folder, m, source


def new_plan(options):
    spec = input_data.load_spec()
    reference = read_reference(options.historical_reference)
    historical = reference["configuration"]
    if historical["backend"] != options.backend:
        raise ValueError("historical reference backend differs; use a reference for the requested backend")
    source = input_data.make_source(spec, options.trajectory_count, 29)
    requested = options.physical_path or historical["physical_path"]
    c = dict(backend=options.backend, trajectory_count=options.trajectory_count, shots=options.shots,
        control_shots=options.control_shots, randomizations=4, fold_factors=[1., 1.25, 1.5],
        seed=29, seed_transpiler=11, optimization_level=3, bootstrap=options.bootstrap,
        batch_size=min(options.batch_size, 128) if options.capture_timing else options.batch_size,
        instruction_fraction=options.instruction_fraction, controls_per_batch=True,
        execution_mode=options.execution_mode, batch_max_time_seconds=options.batch_max_time,
        calibration_policy=options.calibration_policy, physics=spec["physics"], steps=44,
        spec_sha256=input_data.SPEC_SHA256, physical_path=None, requested_physical_path=list(requested),
        capture_timing=options.capture_timing, sampler_options=sampler_options(options.capture_timing),
        dd_condition="on", dd_policy=dict(sequence="XX", spacings=[.25, .5, .25], min_idle_ratio=10.,
            system_sites_1based=list(range(1, 11)), skip_reset_qubits=True, insert_multiple_cycles=False),
        compiler_versions=dd_runtime.versions(), run_token=uuid.uuid4().hex[:16],
        measurement_correction="Manual balanced X masks + job-local marginal zero/one SPAM inversion; no Runtime TREX",
        scope="DD on for raw and twirled/folded profiles; populations only. Experiment6 is an unpaired historical reference.")
    records, batches = base.records_for(c)
    for row in records:
        row["dd_condition"] = "on"
    for i, b in enumerate(batches):
        b["tag"] = f"m2mitdd-{c['run_token']}-{i:04d}"
    c.update(total_jobs=len(batches), total_circuits=len(records), total_shots=sum(r["shots"] for r in records))
    folder = options.output_directory/("mitigation_dd44_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    m = dict(experiment=EXPERIMENT, status="planned", configuration=c, files={}, circuits=records, batches=batches)
    for name, data in (("source.json", source), ("spec.json", spec), ("historical_reference.json", reference)):
        atomic_json(folder/name, data); register(folder, m, name)
    c["source_sha256"] = m["files"]["source.json"]["sha256"]
    verify_manifest(m, source)
    checkpoint(folder, m); write_plan(folder, m)
    return folder, m, source


def write_plan(folder, m):
    c = m["configuration"]
    lines = ["# Model2 Experiment8: twirling + ZNE + SPAM + DD", "",
        f"State: {m['status']}; backend `{c['backend']}`; **DD on only**.",
        f"{c['trajectory_count']:,} seed-29 trajectories; {c['shots']} shots per raw arm/factor, four variants per factor.",
        f"**{c['total_jobs']} jobs, {c['total_circuits']:,} circuits, {c['total_shots']:,} shots including controls.**",
        f"Circuit cap {c['batch_size']}; instruction allowance {c['instruction_fraction']:.0%}; compiler timing capture {c['capture_timing']}.",
        f"Hardware execution mode: {c.get('execution_mode', 'job')}; Runtime Batch max_time seconds: {c.get('batch_max_time_seconds')} (None uses provider default).",
        "These are the saved grouping's counts. Native backend instruction checks must pass before submission; job count is not a QPU-time estimate.", "",
        "Frozen N=10, J=1, Γ/J=2, equal p=.25 reservoirs, half mixed, 44-step warped cosine, jump first/even-odd.",
        "Native CZ folding at factors 1/1.25/1.5 and manual gate/terminal-measurement twirling precede ALAP scheduling and XX insertion.",
        "XX uses ratio 10, spacing [.25,.5,.25], all ten system sites, reset-aware single cycles; original gates and 44 intentional resets retained.",
        "The physical cutoff is ratio × the actual two-X duration (0.8 us for two 40 ns pulses); target timing is saved and checked on submission.",
        "Zero/one/held-out half-mixture controls repeat within each job for raw and twirled treatments, with the same DD policy.",
        "All six intermediate/final profiles use DD; no newly submitted DD-off arm. The local no-DD schedule exists only for construction checks.",
        "Runtime automatic DD/twirling are disabled. Measurement correction is the custom per-qubit, per-job SPAM model, not Runtime TREX.",
        "Local gate/timing/window preservation is checked. DD-on server timing can be inspected with draw_circuit_schedule_timing; no paired DD-off server timing is claimed.",
        "The Experiment6 comparison is historical and cannot isolate DD from date/calibration changes. Population-only accuracy is reported against both saved ideal paths and steady state.",
        "Only submit --hardware or submit --dry-run submits jobs. Collection and analysis reuse saved IDs/counts. Mock counts are never used as hardware evidence.", ""]
    (folder/"Plan.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Manifest: {(folder/'manifest.json').resolve()}", flush=True)
    print(f"DD on: {c['total_jobs']} jobs, {c['total_circuits']:,} circuits, {c['total_shots']:,} shots; state {m['status']}", flush=True)


def load_shard(folder, m, name, backend=None):
    info = m["files"][name]
    if (Path(name).name != name or not 1 <= info["circuit_count"] <= MAX_SHARD_CIRCUITS
            or (folder/name).stat().st_size > MAX_SHARD_BYTES
            or qpy_storage.file_sha256(folder/name) != info["sha256"]):
        raise ValueError("QPY shard size/count/hash mismatch")
    with (folder/name).open("rb") as handle:
        programs = qpy.load(handle)
    rows = [m["circuits"][index] for index in info["pub_indices"]]
    rows.sort(key=lambda r:r["isa_index"])
    if len(programs) != len(rows) or len(rows) != info["circuit_count"]:
        raise ValueError("QPY shard coverage changed")
    base.validate_circuits(m, programs, rows, backend, expected_experiment=EXPERIMENT)
    allowed = set(m["configuration"]["physical_path"][1:-1])
    for p, row in zip(programs, rows):
        pulses = [item for item in p.data if item.operation.label == timing.DD_LABEL]
        if len(pulses) != row["dd_check"]["added_x_pulses"] or not row["dd_check"]["original_operations_and_local_times_preserved"]:
            raise ValueError("saved manual DD pulse check changed")
        if any(item.operation.name != "x" or p.find_bit(item.qubits[0]).index not in allowed for item in pulses):
            raise ValueError("DD pulses outside the system allowlist")
        if (p.metadata or {}).get("dd_policy") != m["configuration"]["dd_policy"]:
            raise ValueError("native DD policy metadata changed")
    return programs


def prepare(options, *, backend=None, service=None):
    if getattr(options, "resume", None):
        folder, m, source = read_saved(options.resume)
        if any(b.get("job_id") or b["status"] == "submitting" for b in m["batches"]):
            raise ValueError("submitted plans are immutable; collect or submit the existing ledger")
        if (m["configuration"]["compiler_versions"] != dd_runtime.versions()
                and any(name.endswith(".qpy") for name in m["files"])):
            raise ValueError("resume compilation in the saved compiler environment")
        m["configuration"]["compiler_versions"] = dd_runtime.versions()
    else:
        folder, m, source = new_plan(options)
    c = m["configuration"]
    service, backend = dd_runtime.get_backend(options, c["backend"], service=service, backend=backend)
    path = c["requested_physical_path"]
    model = timing.timing_model(backend, path)
    if c.get("timing_model") and model != c["timing_model"]:
        raise ValueError("target timing changed; create a new plan instead of mixing schedules")
    settings = dict(avoid_qubits=[], min_t1_seconds=80e-6, min_t2_seconds=60e-6,
        max_cz_error=.005, max_readout_error=.02, max_expansions=2_000_000)
    if c["calibration_policy"] == "screen":
        selection = raw.quality_path.validate(backend, path, source, list(range(c["trajectory_count"])), settings)
    else:
        raw.circuits.chain_layout(backend, 10, "boundary", path)
        selection = dict(physical_path=path, policy="explicit record-only mapping")
    c.update(physical_path=list(path), timing_model=model)
    if "backend_snapshot.json" not in m["files"]:
        raw.runtime.save_backend_snapshot(backend, path, [], folder)
        register(folder, m, "backend_snapshot.json")
        atomic_json(folder/"selected_path.json", selection); register(folder, m, "selected_path.json")
    m["status"] = "compiling"; checkpoint(folder, m)
    layout = path[1:-1]+[path[0], path[-1]]
    policy = dict(c["dd_policy"], physical_qubits=path[1:-1])
    templates = {}
    for i, batch in enumerate(m["batches"]):
        if batch["compiled"]:
            if qpy_storage.file_sha256(folder/batch["isa_file"]) != batch["isa_sha256"]:
                raise ValueError("compiled shard index changed")
            for name in json.loads((folder/batch["isa_file"]).read_text())["shards"]:
                load_shard(folder, m, name, backend)
            continue
        rows = m["circuits"][batch["start"]:batch["stop"]]
        groups = {}
        for row in rows:
            groups.setdefault(protocol.template_key(row), []).append(row)
        shards, vector = [], np.zeros(backend.num_qubits, dtype=np.int64)
        for key, group in groups.items():
            name = f"batch_{i:04d}_{key[0]}_{key[1]}.qpy"
            if name in m["files"]:
                with progress(f"Checking previously compiled shard {name}"):
                    programs = load_shard(folder, m, name, backend)
                    vector += sum((instruction_audit.footprint(p) for p in programs), np.zeros_like(vector))
                    del programs
                shards.append(name)
                continue
            control = group[0]["kind"] == "control"
            if key not in templates:
                logical = raw.build_body(source, 0 if control else key[1], steps=0 if control else None,
                                         populations=group[0].get("expected_populations"))
                native = transpile(logical, backend=backend, initial_layout=layout,
                    optimization_level=3, seed_transpiler=11)
                if native.layout.final_index_layout(filter_ancillas=True) != layout:
                    raise ValueError("transpiler changed the fixed mapping")
                templates[key] = raw.circuits.append_readout(native, layout[:10], "z")
            programs, windows = [], []
            for row in group:
                variant = protocol.make_variant(templates[key], c, row)
                variant.metadata.update(experiment=EXPERIMENT, pub_index=row["pub_index"], batch=i, kind=row["kind"],
                    dd_condition="on", dd_policy=c["dd_policy"])
                variant.metadata["control" if control else "trajectory"] = row["control"] if control else row["trajectory"]
                with progress(f"Scheduling DD-on job {i+1}, {row['case_id']}"):
                    local, original = timing.schedule(variant, backend.target, block_ordering_callable=flat_block_order)
                    program, events = timing.schedule(variant, backend.target, policy=policy, block_ordering_callable=flat_block_order)
                    check = timing.verify_pair(local, program, original, events, policy,
                                               alignment_dt=int(backend.target.pulse_alignment or 1))
                row.update(isa_file=name, isa_index=len(programs), metrics=raw.circuits.metrics(program),
                    final_logical_to_physical=layout, dd_check={k:v for k,v in check.items() if k != "windows"})
                if c["capture_timing"]:
                    windows.append(dict(pub_index=row["pub_index"], case_id=row["case_id"], **check))
                base.validate_circuits(m, [program], [row], backend, expected_experiment=EXPERIMENT)
                validate_isa_circuits([program], backend.target)
                programs.append(program)
                vector += instruction_audit.footprint(program)
                del local, original, events, variant
            temporary = folder/(name+".partial")
            with temporary.open("wb") as handle:
                qpy.dump(programs, handle)
            if temporary.stat().st_size > MAX_SHARD_BYTES or len(programs) > MAX_SHARD_CIRCUITS:
                raise ValueError("compiled shard exceeds its bounded storage limit")
            temporary.replace(folder/name)
            register(folder, m, name, circuit_count=len(programs), pub_indices=[r["pub_index"] for r in group])
            if windows:
                filename = name.replace(".qpy", "_dd_windows.json")
                atomic_json(folder/filename, windows); register(folder, m, filename)
            shards.append(name)
            del programs, program
            if not control:
                del templates[key]
        allowance = int(c["instruction_fraction"]*instruction_audit.INSTRUCTION_LIMIT)
        shots = sum(r["shots"] for r in rows)
        if vector.max() > allowance or shots > instruction_audit.EXECUTION_LIMIT:
            checkpoint(folder, m)
            raise ValueError("actual DD-on job exceeds instruction/execution allowance; prepare a new plan with a smaller circuit cap")
        name = f"job_{i:04d}_compilation.json"
        atomic_json(folder/name, dict(artifact_type="qpy_shard_index_v1", shards=shards,
            instructions_per_qubit=vector.tolist(), shots=shots))
        register(folder, m, name)
        batch.update(compiled=True, isa_file=name, isa_sha256=m["files"][name]["sha256"],
            resources=dict(circuits=len(rows), shots=shots, maximum_instructions=int(vector.max()),
                           instruction_budget=allowance), compiler_versions=dd_runtime.versions())
        checkpoint(folder, m)
        print(f"Compiled job {i+1}/{len(m['batches'])}; all variants use DD.", flush=True)
    for factor in range(3):
        if not any(r["kind"] == "trajectory" and r["arm"] == "twirled" and r["factor_index"] == factor
                   and r["dd_check"]["added_x_pulses"] for r in m["circuits"]):
            raise ValueError("no eligible XX protection at one noise factor; inspect the local schedules")
    m["status"] = "compiled"; verify_manifest(m, source); checkpoint(folder, m); write_plan(folder, m)
    return folder, m


def submit(options, *, backend=None, service=None, sampler_factory=None, batch_factory=None):
    folder, m, _ = read_saved(options.manifest)
    c = m["configuration"]
    mode = "dry_run" if options.dry_run else "hardware"
    use_batch = options.hardware and c.get("execution_mode", "job") == "batch"
    if getattr(options, "new_batch_for_remaining", False) and not use_batch:
        raise ValueError("--new-batch-for-remaining requires a saved batch-mode hardware plan")
    if getattr(options, "batch_max_time", None) is not None and not use_batch:
        raise ValueError("--batch-max-time submission override requires a saved batch-mode hardware plan")
    if m["status"] not in ("compiled", "collected") or not all(b["compiled"] for b in m["batches"]):
        raise ValueError("finish compilation before submitting any job")
    if m.get("submission_mode") not in (None, mode):
        raise ValueError("mock and hardware counts require separate ledgers; use clone-for-hardware after inspection")
    if any(b["status"] == "submitting" and not b.get("job_id") for b in m["batches"]):
        raise ValueError("submission interrupted before its ID was saved; reconcile the saved job tag with IBM; no automatic retry")
    if any(b.get("provider_runtime_batch_id") is not None
           and b["provider_runtime_batch_id"] != b.get("runtime_batch_id") for b in m["batches"]):
        raise ValueError("a recorded job reports a different Runtime Batch; reconcile it before further submissions")
    if all(b.get("job_id") for b in m["batches"]):
        if use_batch:
            from Model2.Experiment8.batch_execution import BatchSubmission
            lifecycle = BatchSubmission(folder, m, options, backend=backend, service=service,
                checkpoint=checkpoint, service_factory=raw.runtime.make_service, factory=batch_factory)
            lifecycle.close()
        print("All job IDs already recorded; use collect to retrieve results.", flush=True)
        return m
    sampler_factory = sampler_factory or dd_runtime.resolve_sampler(dry_run=options.dry_run)
    service, backend = dd_runtime.get_backend(options, c["backend"], service=service, backend=backend)
    if timing.timing_model(backend, c["physical_path"]) != c["timing_model"]:
        raise ValueError("backend timing changed; create a fresh compiled plan")
    if options.hardware and getattr(backend, "is_mock", False):
        raise ValueError("a mock backend cannot produce physical results")
    m["submission_mode"] = mode; checkpoint(folder, m)
    lifecycle = None
    if use_batch:
        from Model2.Experiment8.batch_execution import BatchSubmission
        lifecycle = BatchSubmission(folder, m, options, backend=backend, service=service,
            checkpoint=checkpoint, service_factory=raw.runtime.make_service, factory=batch_factory)
        lifecycle.check_existing()
    for i, batch in enumerate(m["batches"]):
        if batch.get("job_id"):
            print(f"Keeping job {i+1}: {batch['job_id']}", flush=True)
            continue
        if options.hardware:
            raw.backend_readiness(backend, require_ready=True)
        rows = m["circuits"][batch["start"]:batch["stop"]]
        cache, pubs = {}, []
        with progress(f"Loading one bounded submission job {i+1}/{len(m['batches'])}"):
            for row in rows:
                name = row["isa_file"]
                if name not in cache:
                    cache[name] = load_shard(folder, m, name, backend)
                program = cache[name][row["isa_index"]]
                validate_isa_circuits([program], backend.target)
                pubs.append((program, None, row["shots"]))
            measured = base.resource_check([p[0] for p in pubs], rows, c)
            if (measured["maximum_instructions"] != batch["resources"]["maximum_instructions"]
                    or measured["executions"] != batch["resources"]["shots"]):
                raise ValueError("native submission footprint changed")
        opts = json.loads(json.dumps(c["sampler_options"]))
        opts["environment"] = dict(job_tags=[batch["tag"], "model2-experiment8-dd-on"])
        sampler = sampler_factory(mode=lifecycle.mode() if lifecycle else backend, options=opts)
        batch["status"] = "submitting"; checkpoint(folder, m)
        with progress(f"Submitting {mode} job {i+1}"):
            kwargs = {"dry_run":True} if options.dry_run else {}
            job = sampler.run(pubs, **kwargs)
            job_id = job.job_id()
            if not isinstance(job_id, str) or not job_id.strip():
                raise ValueError("no usable job ID returned; reconcile the tagged submission before retrying")
            batch.update(job_id=job_id, status="submitted")
            if lifecycle:
                batch["runtime_batch_id"] = lifecycle.batch_id
            checkpoint(folder, m)
            if lifecycle:
                lifecycle.check_job(job, batch)
        print(f"Submitted {mode} job {i+1}/{len(m['batches'])}: {job_id}", flush=True)
        del pubs, cache, program, sampler, job
        gc.collect()
    if lifecycle:
        lifecycle.close()
    return m


def read_counts(folder, m, index):
    """Also recognize an atomic result file saved before an interrupted checkpoint."""
    b = m["batches"][index]
    name = f"job_{index:04d}_counts.json"
    if b.get("counts_file", name) != name:
        raise ValueError("count file name differs from the job index")
    path = folder/name
    if not path.exists():
        if b.get("counts_file"):
            raise ValueError("recorded count file is missing")
        return None
    if b.get("counts_sha256") and qpy_storage.file_sha256(path) != b["counts_sha256"]:
        raise ValueError("saved count-file hash mismatch")
    with path.open() as handle:
        saved = json.load(handle)
    identity = dict(experiment=EXPERIMENT, plan_sha256=m["plan_sha256"], job_id=b["job_id"],
                    submission_mode=m.get("submission_mode"), batch=index, start=b["start"], stop=b["stop"])
    if any(saved.get(k) != v for k,v in identity.items()):
        raise ValueError("counts belong to another job, mode, or compiled plan")
    rows = m["circuits"][b["start"]:b["stop"]]
    base.validate_observations(saved["observations"], rows)
    if m["configuration"]["capture_timing"]:
        expected = [r["pub_index"] for r in rows]
        if [r["pub_index"] for r in saved["timing_files"]] != expected:
            raise ValueError("incomplete saved compiler timing coverage")
        for r in saved["timing_files"]:
            wanted = f"timing/pub_{r['pub_index']:06d}.txt"
            if r["file"] != wanted or qpy_storage.file_sha256(folder/wanted) != r["sha256"]:
                raise ValueError("compiler timing changed or missing")
    return saved


def collect(options, *, service=None):
    folder, m, _ = read_saved(options.manifest)
    if not any(b.get("job_id") for b in m["batches"]):
        raise ValueError("no recorded job IDs; collection never submits jobs")
    for i, b in enumerate(m["batches"]):
        if not b.get("job_id"):
            print(f"Job {i+1}: no recorded ID; unchanged", flush=True)
            continue
        with progress(f"Checking saved count file for job {i+1}"):
            saved = read_counts(folder, m, i)
        job = None
        if saved is None:
            service = service or raw.runtime.make_service(options)
            with progress(f"Checking IBM job {i+1}: {b['job_id']}"):
                job = service.job(b["job_id"])
                status = job.status()
                status = str(getattr(status, "name", status)).upper()
            print(f"Job {i+1}: {status}", flush=True)
            if status != "DONE":
                if status in ("ERROR", "CANCELLED"):
                    print("Terminal job retained; inspect-job reports its diagnostics. No automatic replacement.", flush=True)
                continue
            with progress(f"Downloading/decoding job {i+1}"):
                result = job.result()
            rows = m["circuits"][b["start"]:b["stop"]]
            if len(result) != len(rows):
                raise ValueError("provider PUB coverage differs from saved job")
            observations, timing_files = [], []
            with progress(f"Saving counts and requested timing for job {i+1}"):
                for pub, row in zip(result, rows):
                    observations.append(dict(outcome_counts={str(k):int(v) for k,v in pub.data.outcome.get_counts().items()}))
                    if m["configuration"]["capture_timing"]:
                        value = pub.metadata.get("compilation", {}).get("scheduler_timing", {}).get("timing")
                        value = timing.timing_text(value)
                        timing.parse_timing(value)
                        name = f"timing/pub_{row['pub_index']:06d}.txt"
                        dest = folder/name; dest.parent.mkdir(exist_ok=True)
                        tmp = dest.with_suffix(".txt.partial")
                        tmp.write_text(value, encoding="utf-8"); tmp.replace(dest)
                        timing_files.append(dict(pub_index=row["pub_index"], file=name, sha256=qpy_storage.file_sha256(dest)))
                base.validate_observations(observations, rows)
                saved = dict(experiment=EXPERIMENT, plan_sha256=m["plan_sha256"], batch=i,
                    job_id=b["job_id"], submission_mode=m["submission_mode"], start=b["start"], stop=b["stop"],
                    observations=observations, timing_files=timing_files,
                    collected_at_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(folder/f"job_{i:04d}_counts.json", saved)
            del result, pub, observations
        name = f"job_{i:04d}_counts.json"
        if b["status"] != "collected":
            b.update(status="collected", counts_file=name, counts_sha256=qpy_storage.file_sha256(folder/name),
                     collected_at_utc=saved["collected_at_utc"])
            with progress(f"Checkpointing collected job {i+1}"):
                checkpoint(folder, m)
        del saved
        if options.job_details and not b.get("details_file"):
            service = service or raw.runtime.make_service(options)
            job = job or service.job(b["job_id"])
            details = dict(job_id=b["job_id"])
            for method in ("metrics", "usage"):
                try:
                    with progress(f"Optional {method}; job {i+1} counts already saved"):
                        details[method] = json.loads(json.dumps(getattr(job, method)(), default=str))
                except Exception as error:
                    details[method+"_unavailable"] = str(error)
            name = f"job_{i:04d}_details.json"; atomic_json(folder/name, details)
            b.update(details_file=name, details_sha256=qpy_storage.file_sha256(folder/name)); checkpoint(folder, m)
        gc.collect()
    if all(b["status"] == "collected" for b in m["batches"]):
        m["status"] = "collected"; checkpoint(folder, m)
    count = sum(b["status"] == "collected" for b in m["batches"])
    print(f"Collected {count}/{len(m['batches'])} jobs. No submission, analysis, or HTML rendering performed.", flush=True)
    return m


def render(options):
    folder, m, _ = read_saved(options.manifest)
    c = m["configuration"]
    if not c["capture_timing"]:
        raise ValueError("this production plan did not request compiler timing; inspect a small --capture-timing cohort")
    selected = [r for r in m["circuits"] if r["kind"] == "trajectory" and r["trajectory"] == options.trajectory
                and r["arm"] == options.arm and (options.arm == "raw" or
                    (r["randomization"] == options.randomization and
                     (options.factor_index is None or r["factor_index"] == options.factor_index)))]
    if not selected:
        raise ValueError("no saved trajectory/variant matches the requested inspection")
    cached = {}
    out = folder/"timing"; out.mkdir(exist_ok=True)
    index = out/"render_cache.json"
    cache = json.loads(index.read_text()) if index.exists() else {}
    paths = []
    for row in selected:
        batch = row["batch"]
        if batch not in cached:
            saved = read_counts(folder, m, batch)
            if saved is None:
                raise ValueError("collect the requested timing job before rendering")
            cached[batch] = {r["pub_index"]:r for r in saved["timing_files"]}
        entry = cached[batch][row["pub_index"]]
        value = (folder/entry["file"]).read_text(encoding="utf-8")
        label = f"path_{row['trajectory']:05d}_{row['case_id']}_{options.view}"
        dest = out/(label+".html")
        identity = input_data.digest(dict(timing_sha256=entry["sha256"], view=options.view, path=c["physical_path"]))
        old = cache.get(label, {})
        if (not options.overwrite and dest.is_file() and old.get("input_sha256") == identity
                and old.get("html_sha256") == qpy_storage.file_sha256(dest)):
            print(f"Reusing timing view: {dest}", flush=True)
        else:
            with progress(f"Drawing compiler-returned DD-on schedule: {row['case_id']}"):
                timing.draw(value, dest, channels=None if options.view == "full" else [f"Qubit {q}" for q in c["physical_path"]])
                timing.write_csv(out/(label+"_instructions.csv"), timing.parse_timing(value))
            cache[label] = dict(input_sha256=identity, html_sha256=qpy_storage.file_sha256(dest))
            atomic_json(index, cache)
        paths.append(dest)
    print("Timing views:\n"+"\n".join(map(str, paths)), flush=True)
    return paths


def clone_for_hardware(options):
    parent, m, _ = read_saved(options.manifest, native_files=True)
    if m.get("submission_mode") != "dry_run" or m["status"] != "collected":
        raise ValueError("clone requires a completely collected cloud dry-run")
    if not m["configuration"]["capture_timing"]:
        raise ValueError("a compiler timing preflight requires --capture-timing")
    for i in range(len(m["batches"])):
        read_counts(parent, m, i)
    provenance = dict(plan_sha256=m["plan_sha256"], jobs=[b["job_id"] for b in m["batches"]],
        scope="Same DD-on QPY passed cloud compilation; no DD-off server timing comparison or noise-performance validation.")
    folder = options.output_directory/("mitigation_dd44_hardware_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    for name in m["files"]:
        shutil.copy2(parent/name, folder/name)
    c = m["configuration"]; c.update(run_token=uuid.uuid4().hex[:16], verified_cloud_compilation=provenance)
    for i,b in enumerate(m["batches"]):
        m["batches"][i] = {k:b[k] for k in ("start", "stop", "compiled", "isa_file", "isa_sha256", "resources", "compiler_versions")}
        m["batches"][i].update(status="not_submitted", job_id=None, tag=f"m2mitdd-{c['run_token']}-{i:04d}")
    m.pop("submission_mode", None); m["status"] = "compiled"
    for key in ("runtime_batch", "runtime_batch_history"):
        m.pop(key, None)
    checkpoint(folder, m); write_plan(folder, m)
    return folder, m


def bundle(options):
    folder, m, _ = read_saved(options.manifest)
    names = ["manifest.json", "Plan.md"]+[name for name in m["files"] if options.include_qpy or not name.endswith(".qpy")]
    for b in m["batches"]:
        names.extend(b[key] for key in ("counts_file", "details_file") if b.get(key))
    for subdir in ("analysis", "timing"):
        if (folder/subdir).exists():
            names.extend(p.relative_to(folder).as_posix() for p in (folder/subdir).iterdir()
                         if p.is_file() and p.suffix not in (".html", ".partial"))
    temporary = folder/"results_bundle.zip.partial"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name in dict.fromkeys(names):
            if (folder/name).is_file():
                with progress(f"Packing {name}"):
                    archive.write(folder/name, arcname=name)
        pointer = folder/"analysis_location.json"
        if pointer.exists():
            location = json.loads(pointer.read_text())
            if location["plan_sha256"] != m["plan_sha256"]:
                raise ValueError("analysis belongs to a different plan")
            output = Path(location["directory"])
            if not (output/"analysis.json").exists():
                raise ValueError("analysis files moved; rerun analyze or restore the recorded output directory")
            for path in output.iterdir():
                if path.is_file():
                    archive.write(path, arcname="analysis/"+path.name)
    temporary.replace(folder/"results_bundle.zip")
    print(f"Transfer: {folder/'results_bundle.zip'}", flush=True)
    return folder/"results_bundle.zip"


def inspect_job(options, *, service=None):
    _, m, _ = read_saved(options.manifest)
    if options.job_id not in [b.get("job_id") for b in m["batches"]]:
        raise ValueError("job ID is not recorded in this experiment")
    service = service or raw.runtime.make_service(options)
    job = service.job(options.job_id)
    status = job.status()
    report = dict(job_id=options.job_id, status=str(getattr(status, "name", status)).upper())
    for method in ("error_message", "metrics"):
        try:
            with progress(f"Inspecting {method}"):
                report[method] = json.loads(json.dumps(getattr(job, method)(), default=str))
        except Exception as error:
            report[method+"_unavailable"] = str(error)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(options):
    if options.action == "plan":
        return new_plan(options)
    if options.action == "analyze":
        from Model2.Experiment8.analyze_hardware import analyze
        return analyze(options.manifest, output=options.output_directory)
    return {"prepare":prepare, "submit":submit, "collect":collect, "render":render,
            "bundle":bundle, "clone-for-hardware":clone_for_hardware, "inspect-job":inspect_job}[options.action](options)


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except KeyboardInterrupt:
        raise SystemExit("Stopped locally; recorded job IDs and atomic checkpoints are retained.") from None
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(str(error)) from error
