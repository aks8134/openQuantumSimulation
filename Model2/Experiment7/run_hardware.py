"""Schedule-first Model2 DD experiment using IBM compiler timing.

prepare: compile a DD-off scout without submitting.
submit: explicitly choose cloud --dry-run or real --hardware; saves job IDs.
collect: fetch only completed known jobs, then render their server timing.
policy: record the system sites chosen after inspecting collected scout timing.
apply-policy: build a fresh matched DD-off/XX experiment on those same paths.
render/analyze: work from saved results without contacting IBM.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid

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
from Model2.Experiment7 import schedule_tools as timing

EXPERIMENT = "Model2/Experiment7/schedule_first_dd_v1"
SAMPLER_OPTIONS = {"dynamical_decoupling": {"enable": False},
    "twirling": {"enable_gates": False, "enable_measure": False},
    "execution": {"init_qubits": True},
    "experimental": {"execution": {"scheduler_timing": True}}}
CONTROLS = {"all_zero": 0., "all_one": 1., "half_mixed": .5}
MAX_QPY_SHARD_BYTES = 64*1024*1024
MAX_QPY_SHARD_CIRCUITS = 9


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    def account(p):
        p.add_argument("--account-file", type=Path)
    def output(p):
        p.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    p = sub.add_parser("prepare", help="compile a timing scout; no submission")
    account(p); output(p)
    p.add_argument("--backend", default="ibm_phoenix")
    p.add_argument("--trajectory-count", type=int, default=4, help="first R fixed seed streams; scout size, not accuracy certification")
    p.add_argument("--shots", type=int, default=32)
    p.add_argument("--control-shots", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--seed-transpiler", type=int, default=11)
    p.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    p.add_argument("--physical-path", type=int, nargs="+", help="left bath, ten system sites, right bath")
    p.add_argument("--calibration-policy", choices=("screen", "record-only"), default="screen")
    p = sub.add_parser("apply-policy", help="compile matched arms using an inspected scout; no submission")
    account(p); output(p)
    p.add_argument("--manifest", required=True, type=Path, help="collected DD-off scout manifest")
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--shots", type=int, default=256)
    p.add_argument("--control-shots", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    for name in ("submit", "collect", "render", "policy", "analyze", "clone-for-hardware"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        if name in ("submit", "collect"):
            account(p)
        if name == "collect":
            p.add_argument("--download-only", action="store_true",
                help="save existing results without optional job-details requests or HTML rendering")
        if name == "render":
            p.add_argument("--trajectory", type=int, help="render one zero-based saved trajectory index")
            p.add_argument("--basis", choices=raw.circuits.BASES, help="render only this measurement basis")
            p.add_argument("--view", choices=("full", "active", "both"), default="both")
            p.add_argument("--overwrite", action="store_true", help="rebuild already verified HTML views")
        if name == "clone-for-hardware":
            output(p)
        if name == "submit":
            mode = p.add_mutually_exclusive_group(required=True)
            mode.add_argument("--dry-run", action="store_true", help="cloud compiler timing/mock outputs; requires eligible plan and Runtime >=0.51")
            mode.add_argument("--hardware", action="store_true", help="real QPU shots; consumes allocation")
        if name == "policy":
            p.add_argument("--system-sites", type=int, nargs="+", required=True, help="explicit 1-based system sites, never bath indices")
            ratios = p.add_mutually_exclusive_group()
            ratios.add_argument("--min-idle-ratio", type=float, default=2., help="idle duration must exceed this times the two X pulse durations")
            ratios.add_argument("--min-idle-ratios", type=float, nargs=2, help="two increasing ratios for one matched baseline + two-DD-arm comparison; e.g. 2 10")
            p.add_argument("--note", required=True, help="what the server schedule showed and why these sites were chosen")
            p.add_argument("--output", type=Path, required=True)
    a = parser.parse_args(arguments)
    if a.action in ("prepare", "apply-policy"):
        if a.shots < 2 or a.control_shots < 2 or a.batch_size < 12:
            parser.error("shots/control-shots >=2 and batch-size >=12 required")
    if a.action == "prepare":
        if not 2 <= a.trajectory_count <= 16384 or min(a.seed, a.seed_transpiler) < 0:
            parser.error("need 2..16384 paths and nonnegative seeds")
        if a.physical_path is not None and (len(a.physical_path) != 12
                or len(set(a.physical_path)) != 12 or min(a.physical_path) < 0):
            parser.error("physical-path requires twelve distinct nonnegative physical qubits")
        if a.calibration_policy == "record-only" and not a.physical_path:
            parser.error("record-only requires an explicit physical path")
    if a.action == "policy":
        ratios = a.min_idle_ratios or [a.min_idle_ratio]
        if (any(not np.isfinite(r) or r < 2 for r in ratios)
                or any(b <= x for x, b in zip(ratios, ratios[1:]))
                or not a.note.strip() or len(set(a.system_sites)) != len(a.system_sites)
                or min(a.system_sites) < 1 or max(a.system_sites) > 10):
            parser.error("choose distinct system sites 1..10, increasing finite ratios >=2 and a nonempty inspection note")
    if a.action == "render" and a.trajectory is not None and a.trajectory < 0:
        parser.error("trajectory is a nonnegative saved index")
    return a


@contextmanager
def progress(message):
    """Expose blocking API/Plotly stages; a heartbeat means waiting, not progress."""
    start = time.monotonic()
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(20):
            print(f"Still waiting: {message} ({time.monotonic()-start:.0f}s elapsed)", flush=True)
    print(f"{message}...", flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    success = False
    try:
        yield
        success = True
    finally:
        stopped.set(); worker.join()
        print(f"{'Finished' if success else 'Stopped'}: {message} ({time.monotonic()-start:.1f}s)", flush=True)


def versions():
    return {name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime", "numpy")}


def static_plan(manifest):
    return {"experiment": manifest["experiment"], "configuration": manifest["configuration"],
        "files": manifest["files"], "circuits": manifest["circuits"],
        "batches": [{k: b[k] for k in ("start", "stop", "resources", "tag")} for b in manifest["batches"]]}


def checkpoint(folder, manifest):
    manifest["plan_sha256"] = input_data.digest(static_plan(manifest))
    atomic_json(folder/"manifest.json", manifest)


def read_saved(path, *, check_files=True):
    path = Path(path).resolve()
    folder, m = path.parent, json.loads(path.read_text())
    if m.get("experiment") != EXPERIMENT or m.get("plan_sha256") != input_data.digest(static_plan(m)):
        raise ValueError("not an intact Experiment7 plan")
    c = m["configuration"]
    if c["spec_sha256"] != input_data.SPEC_SHA256 or c["sampler_options"] != SAMPLER_OPTIONS:
        raise ValueError("frozen physics or sampler options changed")
    policies = dd_policies(c["policy"]) if c.get("policy") is not None else {}
    if c["arms"] != ["baseline", *policies]:
        raise ValueError("saved arm labels differ from their DD policies")
    if check_files:
        for name, details in m["files"].items():
            if Path(name).name != name or qpy_storage.file_sha256(folder/name) != details["sha256"]:
                raise ValueError(f"experiment asset changed or missing: {name}")
    source = json.loads((folder/"source.json").read_text())
    input_data.load_spec(folder/"spec.json")
    if (source["provenance"]["spec_sha256"] != c["spec_sha256"]
            or source["configuration"]["trajectories"] != c["trajectory_count"]
            or source["configuration"]["seed"] != c["seed"]):
        raise ValueError("source identity differs from plan")
    return folder, m, source


def register_file(folder, manifest, name, **details):
    manifest["files"][name] = dict(sha256=qpy_storage.file_sha256(folder/name), **details)


def save_qpy(folder, manifest, name, programs):
    with (folder/(name+".partial")).open("wb") as handle:
        qpy.dump(programs, handle)
    (folder/(name+".partial")).replace(folder/name)
    register_file(folder, manifest, name, circuit_count=len(programs))


def load_qpy(folder, manifest, name):
    details = manifest["files"][name]
    # Qiskit 2.6 requires the Rust decoder for new QPY versions. Public load is
    # safe here because each file is bounded to one path (<=9 circuits) or its
    # controls. Never hand this reader a monolithic experiment archive.
    if not 1 <= details["circuit_count"] <= MAX_QPY_SHARD_CIRCUITS or (folder/name).stat().st_size > MAX_QPY_SHARD_BYTES:
        raise ValueError("QPY shard exceeds the bounded per-path size/count limit")
    if qpy_storage.file_sha256(folder/name) != details["sha256"]:
        raise ValueError("saved QPY hash mismatch")
    with (folder/name).open("rb") as handle:
        programs = qpy.load(handle)
    if len(programs) != details["circuit_count"]:
        raise ValueError("QPY shard circuit count differs from plan")
    return programs


def get_backend(options, name, *, service=None, backend=None):
    if backend is None:
        service = service or raw.runtime.make_service(options)
        backend = service.backend(name, use_fractional_gates=False)
    if backend.name != name:
        raise ValueError("resolved backend differs from requested backend")
    return service, backend


def new_folder(options, config, source, spec):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = options.output_directory/f"dd44_{config['stage']}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment=EXPERIMENT, status="compiling", configuration=config,
                    files={}, circuits=[], batches=[])
    for name, value in (("source.json", source), ("spec.json", spec)):
        atomic_json(folder/name, value); register_file(folder, manifest, name)
    return folder, manifest


def base_config(options):
    return dict(backend=options.backend, trajectory_count=options.trajectory_count, seed=options.seed,
        seed_transpiler=options.seed_transpiler, optimization_level=options.optimization_level,
        spec_sha256=input_data.SPEC_SHA256, shots=options.shots, control_shots=options.control_shots,
        batch_size=options.batch_size, instruction_fraction=.8, measurement_bases=list(raw.circuits.BASES),
        software_versions=versions(), sampler_options=json.loads(json.dumps(SAMPLER_OPTIONS)),
        run_token=uuid.uuid4().hex[:16], stage="scout", arms=["baseline"], policy=None)


def prepare(options, *, backend=None, service=None):
    service, backend = get_backend(options, options.backend, service=service, backend=backend)
    spec = input_data.load_spec()
    source = input_data.make_source(spec, options.trajectory_count, options.seed)
    config = base_config(options)
    settings = dict(avoid_qubits=[], min_t1_seconds=80e-6, min_t2_seconds=60e-6,
                    max_cz_error=.005, max_readout_error=.02, max_expansions=2_000_000)
    paths = list(range(options.trajectory_count))
    if options.physical_path is None:
        selection = raw.quality_path.select(backend, source, paths, avoid=[], min_t1=80e-6,
            min_t2=60e-6, max_cz_error=.005, max_readout_error=.02, max_expansions=2_000_000)
    elif options.calibration_policy == "screen":
        selection = raw.quality_path.validate(backend, options.physical_path, source, paths, settings)
    else:
        raw.circuits.chain_layout(backend, 10, "boundary", options.physical_path)
        selection = dict(physical_path=options.physical_path, policy="explicit path, calibration record-only")
    path = selection["physical_path"]
    config.update(physical_path=path, timing_model=timing.timing_model(backend, path), calibration_settings=settings)
    folder, manifest = new_folder(options, config, source, spec)
    atomic_json(folder/"selected_path.json", selection); register_file(folder, manifest, "selected_path.json")
    raw.runtime.save_backend_snapshot(backend, path, [], folder)
    register_file(folder, manifest, "backend_snapshot.json")
    compile_experiment(folder, manifest, source, backend)
    return folder, manifest


def require_timing(folder, manifest):
    if not manifest["batches"] or any(b["status"] != "collected" for b in manifest["batches"]):
        raise ValueError("collect every scout job before selecting a DD policy")
    fingerprints = []
    for i, batch in enumerate(manifest["batches"]):
        file = folder/batch["result_file"]
        if qpy_storage.file_sha256(file) != batch["result_sha256"]:
            raise ValueError("collected timing/result archive changed")
        saved = json.loads(file.read_text())
        if len(saved["pubs"]) != batch["stop"]-batch["start"]:
            raise ValueError("collected PUB count differs from scout")
        for pub in saved["pubs"]:
            timing.parse_timing(pub["timing"])
        fingerprints.append(dict(batch=i, job_id=batch["job_id"], result_sha256=batch["result_sha256"]))
    return fingerprints


def create_policy(options):
    folder, m, _ = read_saved(options.manifest)
    if m["configuration"]["stage"] != "scout":
        raise ValueError("policy must reference a DD-off scout")
    evidence = require_timing(folder, m)
    sites = sorted(options.system_sites)
    p = dict(artifact_type="model2_dd_policy_v1", sequence="XX", spacings=[.25, .5, .25],
        scout_plan_sha256=m["plan_sha256"], timing_evidence=evidence, system_sites_1based=sites,
        physical_qubits=[m["configuration"]["physical_path"][s] for s in sites],
        min_idle_ratio=options.min_idle_ratio, skip_reset_qubits=True, insert_multiple_cycles=False,
        inspected_at_utc=datetime.now(timezone.utc).isoformat(), inspection_note=options.note)
    if getattr(options, "min_idle_ratios", None) is not None:
        p.pop("min_idle_ratio")
        p.update(artifact_type="model2_dd_policy_comparison_v1", min_idle_ratios=options.min_idle_ratios)
    dd_policies(p)
    if options.output.exists():
        raise ValueError("policy output already exists; use a new filename")
    atomic_json(options.output, p)
    print(f"DD policy: {options.output.resolve()}")
    return p


def dd_policies(policy):
    """Map saved policy parameters to stable arm names, including legacy runs."""
    kind = policy.get("artifact_type")
    if kind == "model2_dd_policy_v1":
        ratios, names = [policy.get("min_idle_ratio")], ["dd"]
    elif kind == "model2_dd_policy_comparison_v1":
        ratios = policy.get("min_idle_ratios")
        if not isinstance(ratios, list) or len(ratios) != 2:
            raise ValueError("comparison policy requires two increasing DD ratios")
        names = ["dd_ratio_"+format(r, ".12g").replace(".", "p") for r in ratios
                 if isinstance(r, (int, float)) and not isinstance(r, bool)]
    else:
        raise ValueError("unknown DD policy type")
    if (any(not isinstance(r, (int, float)) or isinstance(r, bool) or not np.isfinite(r) or r < 2 for r in ratios)
            or any(b <= a for a, b in zip(ratios, ratios[1:])) or len(set(names)) != len(ratios)):
        raise ValueError("DD ratios must be finite, distinct, increasing and >=2")
    return {name: {**policy, "min_idle_ratio": ratio} for name, ratio in zip(names, ratios)}


def compare_arm_timings(config, pairs, expected, *, require_complete=False):
    policies = dd_policies(config["policy"]) if config.get("policy") else {}
    if require_complete:
        wanted = {(i, b) for i in range(config["trajectory_count"]) for b in config["measurement_bases"]}
        if set(pairs) != wanted or any(set(arms) != set(config["arms"]) for arms in pairs.values()):
            raise ValueError("incomplete matched server timing coverage")
    checks = []
    for (trajectory, basis), arms in sorted(pairs.items()):
        for arm, policy in policies.items():
            if "baseline" not in arms or arm not in arms:
                continue
            checks.append(dict(trajectory=trajectory, basis=basis, arm=arm,
                min_idle_ratio=policy["min_idle_ratio"],
                **timing.server_comparison(arms["baseline"], arms[arm],
                    expected_added_x_by_qubit=expected[trajectory, basis, arm])))
    return checks


def apply_policy(options, *, backend=None, service=None):
    parent, scout, source = read_saved(options.manifest)
    evidence = require_timing(parent, scout)
    c = scout["configuration"]
    p = json.loads(options.policy.read_text())
    policies = dd_policies(p)
    sites = p.get("system_sites_1based", [])
    if (c["stage"] != "scout"
            or p.get("scout_plan_sha256") != scout["plan_sha256"] or p.get("timing_evidence") != evidence
            or p.get("sequence") != "XX" or p.get("spacings") != [.25, .5, .25]
            or not sites or sites != sorted(set(sites)) or min(sites) < 1 or max(sites) > 10
            or p.get("physical_qubits") != [c["physical_path"][s] for s in sites]
            or p.get("skip_reset_qubits") is not True or p.get("insert_multiple_cycles") is not False
            or not p.get("inspection_note", "").strip()):
        raise ValueError("policy is not a valid inspected system-only XX policy for this scout")
    service, backend = get_backend(options, c["backend"], service=service, backend=backend)
    if timing.timing_model(backend, c["physical_path"]) != c["timing_model"]:
        raise ValueError("target durations/constraints changed since scout; prepare and inspect a fresh scout")
    if options.batch_size < 3*len(policies)+3+(len(policies)+1)*len(c["measurement_bases"]):
        raise ValueError("batch-size must fit all arms/bases of one path and all arm-local controls")
    config = {**c, "stage": "paired", "arms": ["baseline", *policies], "policy": p,
        "shots": options.shots, "control_shots": options.control_shots, "batch_size": options.batch_size,
        "run_token": uuid.uuid4().hex[:16], "software_versions": versions(),
        "parent_scout_plan_sha256": scout["plan_sha256"]}
    folder, m = new_folder(options, config, source, json.loads((parent/"spec.json").read_text()))
    atomic_json(folder/"dd_policy.json", p); register_file(folder, m, "dd_policy.json")
    raw.runtime.save_backend_snapshot(backend, c["physical_path"], [], folder)
    register_file(folder, m, "backend_snapshot.json")
    compile_experiment(folder, m, source, backend, parent=(parent, scout))
    if m["status"] == "no_eligible_dd_windows":
        raise ValueError(f"no eligible DD windows for at least one policy; nothing should be submitted ({folder})")
    return folder, m


def clone_for_hardware(options):
    """New job ledger, identical tested QPY; never mix mock and physical counts."""
    parent, m, _ = read_saved(options.manifest)
    c = m["configuration"]
    evidence = require_timing(parent, m)
    if c["stage"] != "paired" or m.get("submission_mode") != "dry_run":
        raise ValueError("clone-for-hardware needs a completely collected paired cloud dry-run")
    pairs, expected = {}, {}
    for row, pub in saved_pubs(parent, m):
        if row["kind"] == "evolution":
            pairs.setdefault((row["trajectory"], row["basis"]), {})[row["arm"]] = timing.parse_timing(pub["timing"])
            if row["arm"] != "baseline":
                expected[row["trajectory"], row["basis"], row["arm"]] = row["dd_check"]["added_x_by_qubit"]
    for check in compare_arm_timings(c, pairs, expected, require_complete=True):
        if (not check["non_x_anchor_timing_equal"] or check["baseline_end_dt"] != check["dd_end_dt"]
                or not check["added_x_counts_match"]):
            raise ValueError("paired service timing changed; inspect the schedules before designing a new hardware comparison")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = options.output_directory/f"dd44_hardware_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    for name in m["files"]:
        shutil.copy2(parent/name, folder/name)
    c.update(run_token=uuid.uuid4().hex[:16], verified_cloud_timing=dict(
        plan_sha256=m["plan_sha256"], jobs=evidence))
    m.pop("submission_mode", None)
    for i, batch in enumerate(m["batches"]):
        m["batches"][i] = {k: batch[k] for k in ("start", "stop", "resources")}
        m["batches"][i].update(status="not_submitted", job_id=None, tag=f"m2dd-{c['run_token']}-{i:04d}")
    m["status"] = "compiled"
    checkpoint(folder, m); write_plan(folder, m)
    print(f"New hardware manifest (no jobs submitted): {(folder/'manifest.json').resolve()}")
    return folder, m


def compile_experiment(folder, m, source, backend, *, parent=None):
    c = m["configuration"]
    layout, path = raw.circuits.chain_layout(backend, 10, "boundary", c["physical_path"])
    controls, groups, group_vectors = [], [], []
    policies = dd_policies(c["policy"]) if c.get("policy") else {}
    def compile_body(path_index=0, population=None):
        logical = raw.build_body(source, path_index, steps=0 if population is not None else None,
                                 populations=[population]*10 if population is not None else None)
        native = transpile(logical, backend=backend, initial_layout=layout,
                           optimization_level=c["optimization_level"], seed_transpiler=c["seed_transpiler"])
        if native.layout.final_index_layout(filter_ancillas=True) != layout:
            raise ValueError("transpiler changed the frozen physical mapping")
        return native
    def group(programs, filename, rows):
        save_qpy(folder, m, filename, programs)
        for j, (program, row) in enumerate(zip(programs, rows)):
            row.update(isa_file=filename, isa_index=j, metrics=raw.circuits.metrics(program))
            validate_isa_circuits([program], backend.target)
        return sum((instruction_audit.footprint(p) for p in programs),
                   np.zeros(backend.num_qubits, dtype=np.int64))
    control_programs = []
    for name, value in CONTROLS.items():
        native = raw.circuits.append_readout(compile_body(population=value), layout[:10], "z")
        scheduled, _ = timing.schedule(native, backend.target)
        for arm in c["arms"]:
            control_programs.append(scheduled.copy())
            controls.append(dict(kind="control", control=name, arm=arm, basis="z", shots=c["control_shots"]))
    control_vector = group(control_programs, "controls.qpy", controls)
    for index in range(c["trajectory_count"]):
        raw_name = f"native_{index:05d}.qpy"
        if parent is None:
            body = compile_body(index)
            natives = [raw.circuits.append_readout(body, layout[:10], basis) for basis in c["measurement_bases"]]
            save_qpy(folder, m, raw_name, natives)
        else:
            originals = load_qpy(parent[0], parent[1], raw_name)
            shutil.copy2(parent[0]/raw_name, folder/raw_name)
            register_file(folder, m, raw_name, circuit_count=len(originals))
            natives = originals
        programs, rows, checks = [], [], []
        for basis_index, (basis, native) in enumerate(zip(c["measurement_bases"], natives)):
            with progress(f"Scheduling path {index}, {basis}, baseline"):
                baseline, baseline_events = timing.schedule(native, backend.target)
            variants, variant_checks = {"baseline": baseline}, {}
            for arm, policy in policies.items():
                with progress(f"Scheduling path {index}, {basis}, {arm}"):
                    decoupled, dd_events = timing.schedule(native, backend.target, policy=policy)
                    check = timing.verify_pair(baseline, decoupled, baseline_events, dd_events, policy,
                                               alignment_dt=int(backend.target.pulse_alignment or 1))
                checks.append(dict(basis=basis, arm=arm, min_idle_ratio=policy["min_idle_ratio"], **check))
                variants[arm], variant_checks[arm] = decoupled, check
            if len(c["arms"]) <= 2:
                arms = c["arms"] if (index+sum(map(ord, basis))) % 2 else c["arms"][::-1]
            else:
                # Six permutations across consecutive path/basis groups balance
                # each condition's requested position in a three-arm comparison.
                position = index*len(c["measurement_bases"])+basis_index
                offset = position % len(c["arms"])
                arms = c["arms"][offset:]+c["arms"][:offset]
                if (position//len(c["arms"])) % 2:
                    arms = arms[::-1]
            for arm in arms:
                program = variants[arm]
                if program.count_ops().get("reset", 0) != 44:
                    raise ValueError("evolution no longer contains all 44 reservoir resets")
                program.metadata.update(experiment=EXPERIMENT, trajectory=index, arm=arm, readout_basis=basis)
                programs.append(program)
                row = dict(kind="evolution", trajectory=index, basis=basis, arm=arm, shots=c["shots"])
                if arm != "baseline":
                    row["dd_min_idle_ratio"] = policies[arm]["min_idle_ratio"]
                    row["dd_check"] = {k: v for k, v in variant_checks[arm].items() if k != "windows"}
                rows.append(row)
        name = f"scheduled_{index:05d}.qpy"
        group_vectors.append(group(programs, name, rows)); groups.append(rows)
        if checks:
            name = f"dd_windows_{index:05d}.json"
            atomic_json(folder/name, checks); register_file(folder, m, name)
        print(f"Compiled path {index+1}/{c['trajectory_count']}", flush=True)
    # Keep every path's bases/arms together; repeat complete controls in each job.
    pending, load, shots = [], control_vector.copy(), sum(r["shots"] for r in controls)
    limit = int(c["instruction_fraction"]*instruction_audit.INSTRUCTION_LIMIT)
    def finish():
        nonlocal pending, load, shots
        if not pending:
            return
        start = len(m["circuits"])
        # Controls at the beginning, middle and end; same calibration scope per job.
        block = [dict(r) for r in controls[:len(c["arms"])]]
        midpoint = len(pending)//2
        block += [dict(r) for g in pending[:midpoint] for r in g]
        block += [dict(r) for r in controls[len(c["arms"]):2*len(c["arms"])]]
        block += [dict(r) for g in pending[midpoint:] for r in g]
        block += [dict(r) for r in controls[2*len(c["arms"]):]]
        for row in block:
            row.update(pub_index=len(m["circuits"]), batch=len(m["batches"]))
            m["circuits"].append(row)
        m["batches"].append(dict(start=start, stop=len(m["circuits"]), status="not_submitted", job_id=None,
            tag=f"m2dd-{c['run_token']}-{len(m['batches']):04d}",
            resources=dict(circuits=len(block), shots=shots, peak_instructions_per_qubit=int(load.max()),
                           instruction_allowance=limit)))
        pending, load, shots = [], control_vector.copy(), sum(r["shots"] for r in controls)
    for rows, vector in zip(groups, group_vectors):
        extra = sum(r["shots"] for r in rows)
        if (np.any(control_vector+vector > limit) or extra+sum(r["shots"] for r in controls) > instruction_audit.EXECUTION_LIMIT
                or len(rows)+len(controls) > c["batch_size"]):
            raise ValueError("one complete path and controls cannot fit the chosen job limits")
        if pending and (len(controls)+sum(map(len, pending))+len(rows) > c["batch_size"]
                or np.any(load+vector > limit) or shots+extra > instruction_audit.EXECUTION_LIMIT):
            finish()
        pending.append(rows); load += vector; shots += extra
    finish()
    c.update(total_jobs=len(m["batches"]), total_circuits=len(m["circuits"]),
             total_shots=sum(r["shots"] for r in m["circuits"]))
    m["status"] = "compiled"
    if c["stage"] == "paired" and any(not any(r["arm"] == arm and r.get("dd_check", {}).get("added_x_pulses", 0)
            for r in m["circuits"]) for arm in policies):
        m["status"] = "no_eligible_dd_windows"
    checkpoint(folder, m)
    write_plan(folder, m)
    print(f"Manifest: {(folder/'manifest.json').resolve()}")
    print(f"{c['total_jobs']} jobs, {c['total_circuits']:,} circuits, {c['total_shots']:,} shots (including controls).")


def write_plan(folder, m):
    c = m["configuration"]
    text = ["# Model2 Experiment7: schedule-first dynamical decoupling", "",
        f"Stage: **{c['stage']}**; backend `{c['backend']}`; physical path `{c['physical_path']}`.",
        "Frozen Model2: N=10, J=1, Γ/J=2, equal reservoirs p=.25, half-mixed, 44-step warped cosine.",
        f"First {c['trajectory_count']} fixed seed-{c['seed']} trajectories; all three bases z/xy/yx.",
        f"Arms: {c['arms']}; {c['shots']} shots per trajectory/basis/arm; {c['control_shots']} per control/arm/job.",
        f"**{c['total_jobs']} jobs, {c['total_circuits']:,} circuits, {c['total_shots']:,} total shots.**", "",
        "Preparation/measurement/reservoir physics and all 44 resets retained. Runtime automatic DD and twirling disabled.",
        "DD-off is ALAP scheduled with delays. The paired experiment adds inspected system-only XX inside fitting local block idle windows.",
        "Each job includes zero, one and held-out half-mixture Z controls for each arm. Controls diagnose preparation/readout, not full-depth DD fidelity.",
        "Published instruction/execution limits counted at 80% instruction allowance. These counts do not establish compiler acceptance or QPU cost.",
        "The compiler-returned timing is the visualization source; no timeline_drawer or synthetic schedule fallback.",
        "Cloud dry-run outputs are random mock counts and cannot measure DD performance. Hardware counts are analyzed only from an explicit hardware submission.",
        "Small scouts/pilots do not certify the 0.02 target or the full 16384-path experiment. No QPU jobs are submitted by preparation.", ""]
    for arm, policy in (dd_policies(c["policy"]) if c.get("policy") else {}).items():
        x = [i["duration_seconds"] for i in c["timing_model"]["instructions"]
             if i["name"] == "x" and i["qubits"][0] in policy["physical_qubits"]]
        cutoffs = [2*t*policy["min_idle_ratio"]*1e6 for t in x]
        text.append(f"`{arm}`: XX, minimum idle ratio {policy['min_idle_ratio']:g}; physical idle cutoff >{min(cutoffs):g}..{max(cutoffs):g} us on the selected sites.")
    (folder/"Plan.md").write_text("\n".join(text), encoding="utf-8")


def resolve_sampler(*, dry_run):
    from qiskit_ibm_runtime import SamplerV2
    if dry_run:
        numbers = tuple(int(x) for x in re.findall(r"\d+", importlib.metadata.version("qiskit-ibm-runtime"))[:3])
        if numbers < (0, 51, 0) or "dry_run" not in inspect.signature(SamplerV2.run).parameters:
            raise ValueError("cloud dry-run requires qiskit-ibm-runtime >=0.51 with run(..., dry_run=True). "
                "Install Experiment7/requirements.txt in a separate environment. No hardware fallback is performed.")
    return SamplerV2


def submit(options, *, service=None, backend=None, sampler_factory=None):
    folder, m, _ = read_saved(options.manifest)
    c = m["configuration"]
    mode = "dry_run" if options.dry_run else "hardware"
    if m["status"] == "no_eligible_dd_windows":
        raise ValueError("nothing to compare: this policy inserted no DD")
    if m["status"] not in ("compiled", "collected") or not m["batches"]:
        raise ValueError("a completely compiled experiment is required before submission")
    if m.get("submission_mode") not in (None, mode):
        raise ValueError("a run cannot switch between mock and hardware results; create a fresh run")
    # Fail unsupported dry-run locally, before account discovery or any submission.
    sampler_factory = sampler_factory or resolve_sampler(dry_run=options.dry_run)
    service, backend = get_backend(options, c["backend"], service=service, backend=backend)
    if timing.timing_model(backend, c["physical_path"]) != c["timing_model"]:
        raise ValueError("target timing changed since compilation; prepare a fresh scout")
    if not options.dry_run and getattr(backend, "is_mock", False):
        raise ValueError("mock devices cannot produce hardware DD results")
    m["submission_mode"] = mode; checkpoint(folder, m)
    for i, batch in enumerate(m["batches"]):
        if batch["job_id"]:
            print(f"Keeping job {batch['job_id']}; collect retrieves existing jobs, never resubmits.")
            continue
        if batch["status"] == "submitting":
            raise ValueError(f"ambiguous submission for tag {batch['tag']}; reconcile its job ID on IBM before resuming")
        if not options.dry_run:
            raw.backend_readiness(backend, require_ready=True)
        programs, cache = [], {}
        rows = m["circuits"][batch["start"]:batch["stop"]]
        for row in rows:
            name = row["isa_file"]
            if name not in cache:
                cache[name] = load_qpy(folder, m, name)
            program = cache[name][row["isa_index"]]
            validate_isa_circuits([program], backend.target)
            programs.append((program, None, row["shots"]))
        sampler_options = json.loads(json.dumps(SAMPLER_OPTIONS))
        sampler_options["environment"] = {"job_tags": [batch["tag"], "model2-experiment7"]}
        sampler = sampler_factory(mode=backend, options=sampler_options)
        batch["status"] = "submitting"; checkpoint(folder, m)
        kwargs = {"dry_run": True} if options.dry_run else {}
        job = sampler.run(programs, **kwargs)
        job_id = job.job_id()
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("no usable job ID returned; reconcile the tagged job on IBM before resuming")
        batch.update(job_id=job_id, status="submitted"); checkpoint(folder, m)
        print(f"Submitted {mode} job {i+1}/{len(m['batches'])}: {job_id}", flush=True)
    print(f"Collect when complete: python Model2/Experiment7/run_hardware.py collect --manifest \"{options.manifest}\"")
    return m


def collect(options, *, service=None):
    folder, m, _ = read_saved(options.manifest)
    if not any(b["job_id"] for b in m["batches"]):
        raise ValueError("no submitted job IDs to collect")
    download_only = getattr(options, "download_only", False)
    for i, batch in enumerate(m["batches"]):
        if not batch["job_id"]:
            continue
        job = None
        if batch["status"] != "collected":
            service = service or raw.runtime.make_service(options)
            job = service.job(batch["job_id"])
            status = job.status()
            status = getattr(status, "name", str(status)).upper()
            batch["provider_status"] = status
            print(f"Job {i+1}: {batch['job_id']} {status}", flush=True)
            if status != "DONE":
                if status in ("ERROR", "CANCELLED"):
                    batch["error_message"] = str(job.error_message())
                checkpoint(folder, m); continue
            with progress(f"Downloading/decoding results for {batch['job_id']}"):
                result = job.result()
            rows = m["circuits"][batch["start"]:batch["stop"]]
            if len(result) != len(rows):
                raise ValueError("server PUB count differs from saved job")
            pubs = []
            with progress(f"Saving {len(rows)} PUB results for job {i+1}"):
                for pub, row in zip(result, rows):
                    scheduler = pub.metadata.get("compilation", {}).get("scheduler_timing", {})
                    value = scheduler.get("timing")
                    counts = {str(k): int(v) for k, v in pub.data.outcome.get_counts().items()}
                    if sum(counts.values()) != row["shots"]:
                        raise ValueError("server outcome shot count differs from request")
                    pubs.append(dict(pub_index=row["pub_index"], timing=value, outcome_counts=counts,
                                     metadata=json.loads(json.dumps(pub.metadata, default=str))))
                saved = dict(job_id=batch["job_id"], submission_mode=m["submission_mode"], pubs=pubs,
                             collected_at_utc=datetime.now(timezone.utc).isoformat())
                name = f"job_{i:04d}_results.json"
                atomic_json(folder/name, saved)
                batch.update(result_file=name, result_sha256=qpy_storage.file_sha256(folder/name), status="collected")
                # Results survive interruption in optional API calls or plotting.
                if all(b["status"] == "collected" for b in m["batches"]):
                    m["status"] = "collected"
                checkpoint(folder, m)
            print(f"Saved: {folder/name} ({(folder/name).stat().st_size/1024**2:.1f} MiB)", flush=True)
            del result, pubs, saved
        else:
            result_path = folder/batch["result_file"]
            if qpy_storage.file_sha256(result_path) != batch["result_sha256"]:
                raise ValueError("saved results changed")
            print(f"Using saved results for {batch['job_id']}", flush=True)
        if download_only:
            continue
        if batch.get("details_file"):
            if qpy_storage.file_sha256(folder/batch["details_file"]) != batch["details_sha256"]:
                raise ValueError("saved job details changed")
            continue
        # Keep optional job details separate from immutable timing/count archives:
        # adding usage later must not invalidate an inspected policy's evidence.
        service = service or raw.runtime.make_service(options)
        job = job or service.job(batch["job_id"])
        details = dict(job_id=batch["job_id"], recorded_at_utc=datetime.now(timezone.utc).isoformat())
        for key in ("metrics", "usage"):
            try:
                with progress(f"Retrieving optional {key} for {batch['job_id']}"):
                    details[key] = json.loads(json.dumps(getattr(job, key)(), default=str))
            except Exception as error:
                details[key+"_unavailable"] = str(error)
                print(f"Optional {key} unavailable; timing/count results are already saved.", flush=True)
        name = f"job_{i:04d}_details.json"
        atomic_json(folder/name, details)
        batch.update(details_file=name, details_sha256=qpy_storage.file_sha256(folder/name))
        checkpoint(folder, m)
    if all(b["status"] == "collected" for b in m["batches"]):
        m["status"] = "collected"; checkpoint(folder, m)
    if download_only:
        print("Download-only complete; no optional job-details requests or HTML rendering. No jobs submitted.", flush=True)
    else:
        render(options)
    return m


def saved_pubs(folder, m):
    for batch in m["batches"]:
        if batch["status"] != "collected":
            continue
        name = batch["result_file"]
        if qpy_storage.file_sha256(folder/name) != batch["result_sha256"]:
            raise ValueError("saved results changed")
        saved = json.loads((folder/name).read_text())
        rows = m["circuits"][batch["start"]:batch["stop"]]
        if saved["job_id"] != batch["job_id"] or saved["submission_mode"] != m["submission_mode"] or len(saved["pubs"]) != len(rows):
            raise ValueError("saved result identity differs from job")
        for row, pub in zip(rows, saved["pubs"]):
            if pub["pub_index"] != row["pub_index"]:
                raise ValueError("saved result PUB identity differs from job")
            yield row, pub


def render(options):
    folder, m, _ = read_saved(options.manifest)
    out = folder/"timing"; out.mkdir(exist_ok=True)
    c = m["configuration"]
    trajectory, basis = getattr(options, "trajectory", None), getattr(options, "basis", None)
    view = getattr(options, "view", "both")
    overwrite = getattr(options, "overwrite", False)
    if trajectory is not None and not 0 <= trajectory < c["trajectory_count"]:
        raise ValueError("trajectory is outside this run's saved paths")
    views = ("full", "active") if view == "both" else (view,)
    cache_path = out/"render_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
        if cache.get("artifact_type") != "model2_timing_render_cache_v1":
            cache = {}
    except (FileNotFoundError, ValueError):
        cache = {}
    entries = cache.setdefault("entries", {})
    cache["artifact_type"] = "model2_timing_render_cache_v1"
    comparisons, paired, expected = [], {}, {}
    selected_count = sum(r["kind"] == "evolution"
        and (trajectory is None or r["trajectory"] == trajectory)
        and (basis is None or r["basis"] == basis) for r in m["circuits"])
    print(f"Rendering up to {selected_count} schedules / {selected_count*len(views)} HTML views; verified completed views are reused.", flush=True)
    def valid_asset(entry, filename):
        path = out/filename
        return (filename in entry.get("assets", {}) and path.is_file()
                and qpy_storage.file_sha256(path) == entry["assets"][filename])
    def update_index():
        links = []
        for label, entry in sorted(entries.items()):
            files = []
            for suffix, caption in ((".html", "all channels"), ("_active.html", "12 active qubits"), ("_gaps.csv", "drive gaps")):
                filename = label+suffix
                if filename in entry.get("assets", {}) and (out/filename).exists():
                    files.append(f'<a href="{filename}">{caption}</a>')
            if files:
                links.append(f'<li>{label}: '+" | ".join(files)+"</li>")
        (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Model2 compiler timing</title>'
            '<h1>IBM compiler-returned timing</h1><p>Completed views appear here as they are written. Drive-channel gaps are candidates for inspection; controller/readout activity may restrict insertion. The HTML plots use draw_circuit_schedule_timing.</p><ul>'
            +"\n".join(links)+"</ul>", encoding="utf-8")
    position = 0
    for row, pub in saved_pubs(folder, m):
        if (row["kind"] != "evolution" or (trajectory is not None and row["trajectory"] != trajectory)
                or (basis is not None and row["basis"] != basis)):
            continue
        position += 1
        label = f"path_{row['trajectory']:05d}_{row['basis']}_{row['arm']}"
        value = timing.timing_text(pub["timing"])
        rows = timing.parse_timing(value)
        print(f"Schedule {position}/{selected_count}: {label}, {len(rows):,} compiler instructions", flush=True)
        identity = input_data.digest(dict(timing=value, physical_path=c["physical_path"],
            dt_seconds=c["timing_model"]["dt_seconds"], render_format=1))
        entry = entries.setdefault(label, {})
        if entry.get("input_sha256") != identity:
            entry.clear(); entry.update(input_sha256=identity, assets={})
        tables = [label+suffix for suffix in (".txt", "_instructions.csv", "_gaps.csv")]
        if overwrite or not all(valid_asset(entry, name) for name in tables):
            with progress(f"Exporting instruction and drive-gap tables for {label}"):
                (out/tables[0]).write_text(value, encoding="utf-8")
                timing.write_csv(out/tables[1], rows)
                timing.write_csv(out/tables[2], timing.gap_table(rows, c["physical_path"], c["timing_model"]["dt_seconds"]))
            for name in tables:
                entry["assets"][name] = qpy_storage.file_sha256(out/name)
            atomic_json(cache_path, cache); update_index()
        for selected_view in views:
            name = label+(".html" if selected_view == "full" else "_active.html")
            if not overwrite and valid_asset(entry, name):
                print(f"Reusing completed view: {name}", flush=True)
                continue
            channels = None if selected_view == "full" else [f"Qubit {q}" for q in c["physical_path"]]
            with progress(f"Drawing/writing {selected_view} view for {label}"):
                timing.draw(value, out/name, channels=channels)
            entry["assets"][name] = qpy_storage.file_sha256(out/name)
            atomic_json(cache_path, cache); update_index()
            print(f"View ready: {out/name}", flush=True)
        paired.setdefault((row["trajectory"], row["basis"]), {})[row["arm"]] = rows
        if row["arm"] != "baseline":
            expected[row["trajectory"], row["basis"], row["arm"]] = row["dd_check"]["added_x_by_qubit"]
    comparisons = compare_arm_timings(c, paired, expected)
    comparison_file = "server_comparison.json" if trajectory is None and basis is None else (
        f"server_comparison_path_{trajectory if trajectory is not None else 'all'}_{basis or 'all'}.json")
    atomic_json(out/comparison_file, comparisons)
    update_index()
    print(f"Timing views: {(out/'index.html').resolve()}", flush=True)
    return comparisons


def main(options):
    actions = dict(prepare=prepare, submit=submit, collect=collect, render=render,
                   policy=create_policy, **{"apply-policy": apply_policy, "clone-for-hardware": clone_for_hardware})
    if options.action == "analyze":
        from Model2.Experiment7.analyze_hardware import analyze
        return analyze(options.manifest)
    return actions[options.action](options)


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(str(error)) from error
