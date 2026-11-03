"""Frozen Model2 cosine44 hardware experiment; default is an offline plan.

--execute submits, --compile-only contacts the backend without submitting.
--resume reuses the saved source, compilation, jobs and per-job controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np
from qiskit import qpy, transpile

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, qpy_storage
from Model3.Experiment5 import circuit_depth as circuits
from Model3.Experiment5 import quality_path
from Model3.Experiment5 import run_hardware as runtime

EXPERIMENT = "Model2/Experiment5/hardware"


def parse_arguments(arguments=None):
    argv = sys.argv[1:] if arguments is None else list(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_phoenix")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--spec", type=Path, default=input_data.DEFAULT_SPEC)
    parser.add_argument("--trajectory-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--shots", type=int, default=8, help="shots per path per basis")
    parser.add_argument("--bases", choices=circuits.BASES, nargs="+", default=list(circuits.BASES))
    parser.add_argument("--batch-size", type=int, default=512, help="maximum circuits per job including three controls")
    parser.add_argument("--control-shots", type=int, default=2048)
    parser.add_argument("--order-seed", type=int, default=1729)
    parser.add_argument("--physical-path", type=int, nargs="+", help="left bath, ten sites, right bath; otherwise calibration-selected")
    parser.add_argument("--avoid-qubits", type=int, nargs="*", default=[])
    parser.add_argument("--min-t1-us", type=float, default=80.)
    parser.add_argument("--min-t2-us", type=float, default=60.)
    parser.add_argument("--max-cz-error", type=float, default=.005)
    parser.add_argument("--max-readout-error", type=float, default=.02)
    parser.add_argument("--max-path-expansions", type=int, default=2_000_000)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--recover-compiled", action="store_true",
                        help="offline: recover the completed QPY from an interrupted legacy compilation; requires --resume")
    action.add_argument("--inspect-job", metavar="JOB_ID", help="read-only job/error/usage and backend-status inspection; requires --resume")
    parser.add_argument("--retry-failed-job", metavar="JOB_ID", help="explicit retry of a saved terminal ERROR/6056 job; requires --resume and --execute")
    parser.add_argument("--resume", type=Path, metavar="MANIFEST")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    options = parser.parse_args(argv)
    if options.recover_compiled and not options.resume:
        raise ValueError("--recover-compiled requires --resume MANIFEST")
    if options.inspect_job and not options.resume:
        raise ValueError("--inspect-job requires --resume MANIFEST")
    if options.retry_failed_job and not (options.resume and options.execute):
        raise ValueError("--retry-failed-job requires --resume MANIFEST --execute")
    if not 2 <= options.trajectory_count <= 16384 or options.shots < 1 or options.control_shots < 2:
        raise ValueError("need 2..16384 trajectories, positive shots and control-shots>=2")
    if options.bases not in (["z"], list(circuits.BASES)):
        raise ValueError("bases must be z or z xy yx")
    if options.batch_size < len(options.bases)+3:
        raise ValueError("batch-size must fit a complete path's bases and all three controls")
    if min(options.seed, options.order_seed, options.seed_transpiler) < 0 or options.max_path_expansions < 1:
        raise ValueError("seeds must be nonnegative and path search cap positive")
    if (not options.backend.strip() or any(q < 0 for q in options.avoid_qubits)
            or not np.all(np.isfinite([options.min_t1_us,options.min_t2_us,options.max_cz_error,options.max_readout_error]))
            or min(options.min_t1_us, options.min_t2_us) <= 0
            or not 0 < options.max_cz_error < 1 or not 0 < options.max_readout_error < 1):
        raise ValueError("invalid backend or calibration settings")
    if options.physical_path is not None and (len(options.physical_path) != 12
            or len(set(options.physical_path)) != 12 or min(options.physical_path) < 0):
        raise ValueError("physical-path must contain twelve distinct nonnegative qubit indices")
    if options.resume:
        allowed = {"--resume", "--execute", "--compile-only", "--recover-compiled", "--inspect-job", "--retry-failed-job", "--account-file"}
        if any(word.split("=",1)[0] not in allowed for word in argv if word.startswith("--")):
            raise ValueError("resume uses saved settings; only action and account-file may accompany --resume")
    return options


def plan_hash(manifest):
    return input_data.digest({"configuration": manifest["configuration"], "circuits": manifest["circuits"],
        "batch_ranges": [(b["start"], b["stop"]) for b in manifest["batches"]]})


def plan(source, spec, options):
    per_job = (options.batch_size-3)//len(options.bases)
    order = np.random.default_rng(options.order_seed).permutation(options.trajectory_count).tolist()
    records, batches = [], []
    for begin in range(0, len(order), per_job):
        group = order[begin:begin+per_job]
        start = len(records)
        job = len(batches)
        def control(name, value):
            return {"kind": "control", "control": name, "basis": "z", "shots": options.control_shots,
                    "expected_populations": [value]*10, "batch": job}
        evolved = [{"kind": "evolution", "trajectory": path, "basis": basis, "shots": options.shots,
                    "batch": job} for path in group for basis in options.bases]
        midpoint = (len(group)//2)*len(options.bases)
        records.extend([control("all_zero", 0.), *evolved[:midpoint], control("all_one", 1.),
                        *evolved[midpoint:], control("half_mixed", .5)])
        batches.append({"start": start, "stop": len(records), "job_id": None, "status": "not_submitted"})
    for index, row in enumerate(records):
        row["pub_index"] = index
    config = {"backend": options.backend, "physics": spec["physics"], "spec_sha256": input_data.digest(spec),
        "trajectories": options.trajectory_count, "seed": options.seed, "shots": options.shots,
        "measurement_bases": options.bases, "control_shots": options.control_shots,
        "batch_size": options.batch_size, "order_seed": options.order_seed,
        "preparation_steps": 44, "optimization_level": options.optimization_level,
        "seed_transpiler": options.seed_transpiler, "requested_physical_path": options.physical_path,
        "physical_path": None, "total_circuits": len(records), "total_shots": sum(r["shots"] for r in records),
        "total_jobs": len(batches), "ancillas": "boundary", "source_sha256": None, "isa_sha256": None,
        "path_settings": {"avoid_qubits": options.avoid_qubits, "min_t1_seconds": options.min_t1_us*1e-6,
            "min_t2_seconds": options.min_t2_us*1e-6, "max_cz_error": options.max_cz_error,
            "max_readout_error": options.max_readout_error, "max_expansions": options.max_path_expansions}}
    return {"experiment": EXPERIMENT, "status": "planned", "configuration": config,
            "circuits": records, "batches": batches}


def write_plan(manifest, source, folder):
    c = manifest["configuration"]
    a = source["finite_path_assessment"]
    lines = ["# Model2 Experiment5 hardware plan", "",
        "Frozen 44-step warped cosine, N=10, J=1, Gamma=2, p_L=p_R=.25, half-mixed preparation, J/4 exchange, jump-first even/odd layers.",
        f"Backend: `{c['backend']}`. Status: `{manifest['status']}`.",
        f"{c['trajectories']} paths x {c['shots']} shots x {len(c['measurement_bases'])} bases; three {c['control_shots']}-shot Z controls inside each job.",
        f"**{c['total_jobs']} jobs, {c['total_circuits']} circuits, {c['total_shots']} total shots.**", "",
        f"Ideal finite-path target error: {a['ideal_subset_target_error']:.9g}; sampled-minus-mean distance: {a['sampling_distance']:.9g}; maximum descriptive two SE: {a['maximum_two_standard_errors']:.9g}.",
        f"This finite ensemble passes the classical empirical checks: **{a['empirical_classical_checks_passed']}**.",
        "The default is a hardware pilot. The passing full 16384-path classical replay does not certify a smaller subset, measurement precision or device fidelity. Compare hardware to its own saved paths as well as the physical stationary target. No seed/path selection on hardware outcomes.", "",
        "Every path's requested bases stay in one job. Zero, one and held-out half-mixture controls share that job and physical mapping. Requested order is recorded; actual dispatch timing is not guaranteed. No automatic SPAM correction, twirling, ZNE or DD.", "",
        f"Physical path: {c['physical_path']}. Twelve active qubits when evolving: left bath, ten sites, right bath. Preparation records are ignored. Only final `outcome` counts are analyzed."]
    (folder/"Plan.md").write_text("\n".join(lines)+"\n")


def build_body(source, trajectory, steps=None, populations=None):
    if populations is not None:
        source = {**source, "preparation": {"name": "control", "populations": populations}}
    circuit = circuits.build_body(source, trajectory, steps=steps, ancillas="boundary")
    circuit.metadata["experiment"] = "Model2/Experiment5"
    return circuit


def storage_records(manifest):
    return [{key: batch[key] for key in ("start", "stop", "isa_file", "isa_sha256", "isa_byte_range")
             if key in batch} for batch in manifest["batches"]]


def checkpoint(manifest, folder):
    manifest["configuration"]["isa_index_sha256"] = input_data.digest(storage_records(manifest))
    manifest["plan_sha256"] = plan_hash(manifest)
    atomic_json(folder/"manifest.json", manifest)


class Publications:
    """Load just the requested Runtime job; never retain the whole experiment."""
    def __init__(self, manifest, folder, backend=None):
        self.manifest, self.folder, self.backend = manifest, folder, backend
        self.archive = None
        if manifest["configuration"].get("isa_storage") != "batch-qpy-v1":
            self.archive = qpy_storage.Archive(folder/"isa_circuits.qpy", len(manifest["circuits"]))

    def __len__(self):
        return len(self.manifest["circuits"])

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("load publications one complete job slice at a time")
        batch = next((b for b in self.manifest["batches"]
                      if (b["start"], b["stop"]) == (selection.start, selection.stop)), None)
        if batch is None:
            raise ValueError("publication slice is not a complete saved job")
        records = self.manifest["circuits"][batch["start"]:batch["stop"]]
        if self.archive is not None:
            programs = self.archive.load_batch(batch)
        else:
            path = self.folder/batch["isa_file"]
            archive = qpy_storage.Archive(path, len(records))
            programs, _ = next(archive.iter_batches([{"start": 0, "stop": len(records)}]))
        validate_publications(self.manifest, programs, records)
        if self.backend is not None:
            for program in programs:
                circuits.assert_native(program, self.backend)
        return programs

    def validate_all(self):
        for batch in self.manifest["batches"]:
            programs = self[batch["start"]:batch["stop"]]
            del programs


def compile_plan(manifest, source, backend, folder):
    c = manifest["configuration"]
    if backend.name != c["backend"]:
        raise ValueError("resolved backend does not match the saved plan")
    if not {"cz", "reset", "measure", "rz", "sx", "x"} <= set(backend.target.operation_names):
        raise ValueError("backend requires CZ, reset, measure, RZ, SX and X support")
    if (folder/"isa_circuits.qpy").exists() and c.get("isa_storage") != "batch-qpy-v1":
        raise ValueError("a legacy QPY already exists; recover it with --resume MANIFEST --recover-compiled")
    settings = c["path_settings"]
    indices = list(range(c["trajectories"]))
    fixed_path = c.get("physical_path") or c["requested_physical_path"]
    if fixed_path is None:
        selection = quality_path.select(backend, source, indices, avoid=settings["avoid_qubits"],
            min_t1=settings["min_t1_seconds"], min_t2=settings["min_t2_seconds"],
            max_cz_error=settings["max_cz_error"], max_readout_error=settings["max_readout_error"],
            max_expansions=settings["max_expansions"])
    else:
        selection = quality_path.validate(backend, fixed_path, source, indices, settings)
    layout, path = circuits.chain_layout(backend, 10, "boundary", selection["physical_path"])
    atomic_json(folder/"selected_path.json", selection)
    print(f"Selected path: {path}", flush=True)
    c.update(physical_path=path, isa_storage="batch-qpy-v1",
             versions={name: importlib.metadata.version(name) for name in ("qiskit","qiskit-ibm-runtime","numpy")})
    manifest["status"] = "compiling"
    if not (folder/"backend_snapshot.json").exists():
        runtime.save_backend_snapshot(backend, path, manifest["circuits"], folder)
    checkpoint(manifest, folder)
    bodies, controls = {}, {}
    compiled_count = 0
    for batch in manifest["batches"]:
        records = manifest["circuits"][batch["start"]:batch["stop"]]
        if batch.get("isa_sha256"):
            if qpy_storage.file_sha256(folder/batch["isa_file"]) != batch["isa_sha256"]:
                raise ValueError("saved native batch hash mismatch")
            programs = Publications(manifest, folder, backend)[batch["start"]:batch["stop"]]
            del programs
            compiled_count += sum(row["kind"] == "evolution" and row["basis"] == c["measurement_bases"][-1] for row in records)
            continue
        publications = []
        for row in records:
            is_control = row["kind"] == "control"
            key = row["control"] if is_control else row["trajectory"]
            cache = controls if is_control else bodies
            if key not in cache:
                logical = (build_body(source, 0, steps=0, populations=row["expected_populations"])
                           if is_control else build_body(source, key))
                native = transpile(logical, backend=backend, initial_layout=layout,
                    optimization_level=c["optimization_level"], seed_transpiler=c["seed_transpiler"])
                mapping = native.layout.final_index_layout(filter_ancillas=True)
                if mapping != layout:
                    raise ValueError("transpiler changed the fixed chain mapping; no circuits submitted")
                cache[key] = native
            native = circuits.append_readout(cache[key], layout[:10], row["basis"])
            native.metadata.update(experiment="Model2/Experiment5", kind=row["kind"], pub_index=row["pub_index"], batch=row["batch"])
            if is_control:
                native.metadata["control"] = row["control"]
            circuits.assert_native(native, backend)
            row.update(metrics=circuits.metrics(native), final_logical_to_physical=list(layout))
            publications.append(native)
            if not is_control and row["basis"] == c["measurement_bases"][-1]:
                compiled_count += 1
                if compiled_count % max(1,c["trajectories"]//20) == 0:
                    print(f"Compiled {compiled_count}/{c['trajectories']} paths", flush=True)
                del bodies[key]
        validate_publications(manifest, publications, records)
        name = "isa_circuits.qpy" if len(manifest["batches"]) == 1 else f"isa_{batch['start']:06d}_{batch['stop']:06d}.qpy"
        temporary = folder/(name+".partial")
        with temporary.open("wb") as handle:
            qpy.dump(publications, handle)
        del publications, native
        temporary.replace(folder/name)
        batch.update(isa_file=name, isa_sha256=qpy_storage.file_sha256(folder/name))
        checkpoint(manifest, folder)
    c["isa_sha256"] = input_data.digest(storage_records(manifest))
    manifest["status"] = "compiled"
    checkpoint(manifest, folder)
    write_plan(manifest, source, folder)
    return Publications(manifest, folder, backend)


def validate_publications(manifest, publications, records=None):
    records = manifest["circuits"] if records is None else records
    if len(publications) != len(records):
        raise ValueError("native circuit count differs from saved manifest")
    for circuit,row in zip(publications,records):
        metadata = circuit.metadata or {}
        if (metadata.get("experiment") != "Model2/Experiment5" or metadata.get("pub_index") != row["pub_index"]
                or metadata.get("kind") != row["kind"] or metadata.get("batch") != row["batch"]
                or metadata.get("readout_basis") != row["basis"]):
            raise ValueError("native circuit identity differs from its manifest record")
        if row["kind"] == "evolution" and (metadata.get("trajectory") != row["trajectory"] or metadata.get("steps") != 44):
            raise ValueError("native trajectory or preparation length changed")
        if row["kind"] == "control" and (metadata.get("control") != row["control"] or metadata.get("steps") != 0):
            raise ValueError("native preparation control identity changed")
        registers = [reg for reg in circuit.cregs if reg.name == "outcome"]
        if len(registers) != 1 or len(registers[0]) != 10:
            raise ValueError("native circuit needs a ten-bit final outcome register")
        bits = {bit:index for index,bit in enumerate(registers[0])}
        measured = {}
        for instruction in circuit.data:
            if instruction.operation.name == "measure" and instruction.clbits[0] in bits:
                index = bits[instruction.clbits[0]]
                if index in measured:
                    raise ValueError("final outcome bit measured more than once")
                measured[index] = circuit.find_bit(instruction.qubits[0]).index
        if [measured.get(index) for index in range(10)] != row["final_logical_to_physical"][:10]:
            raise ValueError("native final measurements differ from the saved system mapping")
        expected = 44 if row["kind"] == "evolution" else 0
        if circuit.count_ops().get("reset",0) != expected:
            raise ValueError("native bath reset count differs from the frozen protocol")


def index_legacy_archive(manifest, source, folder, recovering=False):
    """Read one legacy job at a time; recover metadata lost after qpy.dump."""
    c = manifest["configuration"]
    if recovering and (manifest["status"] != "planned" or c.get("isa_storage")
            or any(b.get("job_id") or b.get("observations") is not None
                   or b["status"] != "not_submitted" for b in manifest["batches"])):
        raise ValueError("legacy recovery requires an unsubmitted, uncompiled plan")
    archive = qpy_storage.Archive(folder/"isa_circuits.qpy", len(manifest["circuits"]))
    selection = json.loads((folder/"selected_path.json").read_text()) if recovering else None
    path = selection["physical_path"] if recovering else c["physical_path"]
    if len(path) != 12 or len(set(path)) != 12 or min(path) < 0:
        raise ValueError("invalid saved physical chain")
    layout = path[1:-1]+[path[0], path[-1]]
    active = set(path)
    edges = {frozenset(pair) for pair in zip(path, path[1:])}
    # Mutate a copy until every program has been decoded and checked. A failed
    # recovery never changes the existing manifest or its circuit archive.
    recovered = json.loads(json.dumps(manifest))
    for number, (batch, (programs, byte_range)) in enumerate(zip(recovered["batches"], archive.iter_batches(recovered["batches"])), 1):
        records = recovered["circuits"][batch["start"]:batch["stop"]]
        for program, row in zip(programs, records):
            if recovering:
                row.update(metrics=circuits.metrics(program), final_logical_to_physical=list(layout))
            if (program.metadata or {}).get("physical_sites") != layout[:10]:
                raise ValueError("legacy readout metadata differs from the saved path")
            for instruction in program.data:
                args = [program.find_bit(q).index for q in instruction.qubits]
                if (instruction.operation.name not in {"cz", "rz", "sx", "x", "measure", "reset", "barrier", "delay", "id"}
                        or not set(args) <= active
                        or (instruction.operation.name == "cz" and frozenset(args) not in edges)):
                    raise ValueError("legacy program uses gates outside the saved native chain")
            if program.count_ops().get("cz", 0) != (880 if row["kind"] == "evolution" else 0):
                raise ValueError("legacy CZ resources differ from the frozen protocol")
        validate_publications(recovered, programs, records)
        batch.update(isa_file="isa_circuits.qpy", isa_byte_range=byte_range)
        del programs
        print(f"Verified saved compilation: job {number}/{len(recovered['batches'])}", flush=True)
    file_hash = qpy_storage.file_sha256(folder/"isa_circuits.qpy")
    if c["isa_sha256"] is not None and c["isa_sha256"] != file_hash:
        raise ValueError("saved native circuit hash mismatch")
    recovered["configuration"].update(physical_path=path, isa_sha256=file_hash, isa_storage="legacy-indexed-qpy-v1")
    if recovering:
        recovered["compilation_recovery"] = {
            "scope": "Recovered completed legacy archive before any submission. Circuit identities, mapping, native chain and resources verified; original archive retained unchanged.",
            "qpy_writer_version": f"{archive.header.major_version}.{archive.header.minor_version}.{archive.header.patch_version}",
            "reader_qiskit_version": importlib.metadata.version("qiskit"),
            "compilation_backend_snapshot_available": (folder/"backend_snapshot.json").exists()}
    recovered["status"] = "compiled" if recovering else recovered["status"]
    checkpoint(recovered, folder)
    manifest.clear()
    manifest.update(recovered)
    write_plan(manifest, source, folder)
    return Publications(manifest, folder)


def read_saved(path):
    folder = path.resolve().parent
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != EXPERIMENT or manifest.get("plan_sha256") != plan_hash(manifest):
        raise ValueError("not an intact Model2 Experiment5 manifest")
    c = manifest["configuration"]
    if qpy_storage.file_sha256(folder/"source.json") != c["source_sha256"]:
        raise ValueError("saved source hash mismatch")
    spec = input_data.load_spec(folder/"spec.json")
    if input_data.digest(spec) != c["spec_sha256"]:
        raise ValueError("saved specification changed")
    descriptors = storage_records(manifest)
    if c.get("isa_index_sha256") and c["isa_index_sha256"] != input_data.digest(descriptors):
        raise ValueError("saved native circuit index hash mismatch")
    if c.get("isa_storage") == "batch-qpy-v1":
        for batch in manifest["batches"]:
            if batch.get("isa_sha256") and qpy_storage.file_sha256(folder/batch["isa_file"]) != batch["isa_sha256"]:
                raise ValueError("saved native batch hash mismatch")
        if c["isa_sha256"] is not None and (any(not b.get("isa_sha256") for b in manifest["batches"])
                or c["isa_sha256"] != input_data.digest(descriptors)):
            raise ValueError("saved native circuit storage hash mismatch")
    elif c["isa_sha256"] is not None and qpy_storage.file_sha256(folder/"isa_circuits.qpy") != c["isa_sha256"]:
        raise ValueError("saved native circuit hash mismatch")
    with (folder/"source.json").open() as handle:
        return folder, manifest, json.load(handle)


def bundle(folder):
    names = ["manifest.json","source.json","spec.json","Plan.md","isa_circuits.qpy","backend_snapshot.json",
             "selected_path.json","execution_backend_snapshot.json","hardware_results.json",
             "Analysis.md","observables.csv","observables.png"]
    manifest = json.loads((folder/"manifest.json").read_text())
    names += [b["isa_file"] for b in manifest["batches"] if b.get("isa_file") and b["isa_file"] not in names]
    names = list(dict.fromkeys(names))
    with zipfile.ZipFile(folder/"results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if (folder/name).exists():
                archive.write(folder/name, arcname=name)


def backend_readiness(backend, require_ready=False):
    getter = getattr(backend, "status", None)
    if not callable(getter):
        return {"checked": False, "scope": "local target without a live backend status API"}
    status = getter()
    details = {"checked": True, "operational": status.operational, "status_msg": status.status_msg}
    if require_ready and (status.operational is not True or status.status_msg.strip().lower() != "active"):
        raise ValueError(f"backend {backend.name} is not ready for new submissions: {details}; saved jobs/circuits retained")
    return details


def job_diagnostic(manifest, service, job_id):
    batch = next((b for b in manifest["batches"] if b.get("job_id") == job_id), None)
    if batch is None:
        raise ValueError("job ID is not a current saved job in this manifest")
    job = service.job(job_id)
    state = job.status()
    details = {"job_id": job_id, "status": str(getattr(state, "name", state)).upper(),
               "error_message": job.error_message(), "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        details["metrics"] = json.loads(json.dumps(job.metrics(), default=str))
    except Exception as error:
        details["metrics_unavailable"] = str(error)
    return batch, details


def inspect_saved_job(options, service=None, backend=None):
    # No circuit archive/source load is needed for this read-only diagnosis.
    folder = options.resume.resolve().parent
    manifest = json.loads(options.resume.read_text())
    if manifest.get("experiment") != EXPERIMENT or manifest.get("plan_sha256") != plan_hash(manifest):
        raise ValueError("not an intact Model2 Experiment5 manifest")
    if service is None:
        service = runtime.make_service(options)
    _, details = job_diagnostic(manifest, service, options.inspect_job)
    if backend is None:
        backend = service.backend(manifest["configuration"]["backend"], use_fractional_gates=False)
    details["backend_status"] = backend_readiness(backend)
    atomic_json(folder/"last_job_diagnostic.json", details)
    print(json.dumps(details, indent=2, default=str), flush=True)
    print("Read-only inspection; no jobs submitted or reset for retry.", flush=True)
    return details


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.inspect_job:
        return inspect_saved_job(options, service, backend)
    if options.resume:
        folder, manifest, source = read_saved(options.resume)
    else:
        spec = input_data.load_spec(options.spec)
        source = input_data.make_source(spec, options.trajectory_count, options.seed)
        manifest = plan(source, spec, options)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder = options.output_directory/f"cosine44_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder/"source.json", source)
        atomic_json(folder/"spec.json", spec)
        manifest["configuration"]["source_sha256"] = qpy_storage.file_sha256(folder/"source.json")
        manifest["plan_sha256"] = plan_hash(manifest)
        atomic_json(folder/"manifest.json", manifest)
        write_plan(manifest, source, folder)
    c = manifest["configuration"]
    print(f"Plan: {folder/'manifest.json'}", flush=True)
    print(f"Workload: {c['total_jobs']} jobs, {c['total_circuits']} circuits, {c['total_shots']} shots", flush=True)
    print(f"Exact finite-path assessment: {source['finite_path_assessment']}", flush=True)
    if options.recover_compiled:
        if c["isa_sha256"] is None:
            index_legacy_archive(manifest, source, folder, recovering=True)
        elif not c.get("isa_index_sha256") and c.get("isa_storage") != "batch-qpy-v1":
            index_legacy_archive(manifest, source, folder)
        print("Saved compilation recovered/indexed offline; no account loaded or jobs submitted.", flush=True)
        return manifest
    if not options.execute and not options.compile_only:
        print("Offline plan only; no account loaded, backend queried or jobs submitted.", flush=True)
        return manifest
    if manifest["status"] == "collected":
        if options.retry_failed_job:
            raise ValueError("collected runs cannot be retried")
        from Model2.Experiment5.analyze import main as analyze
        result = analyze(folder)
        bundle(folder)
        return result
    if c["isa_sha256"] is None and (folder/"isa_circuits.qpy").exists() and c.get("isa_storage") != "batch-qpy-v1":
        raise ValueError("completed legacy QPY found; first use --resume MANIFEST --recover-compiled (offline), then --resume MANIFEST --execute")
    if c["isa_sha256"] is not None and not c.get("isa_index_sha256") and c.get("isa_storage") != "batch-qpy-v1":
        index_legacy_archive(manifest, source, folder)
        c = manifest["configuration"]
    retry = None
    if options.retry_failed_job:
        if service is None:
            service = runtime.make_service(options)
        retry = job_diagnostic(manifest, service, options.retry_failed_job)
        batch, details = retry
        if (batch.get("observations") is not None or details["status"] != "ERROR"
                or not re.search(r"\b6056\b", str(details["error_message"]))):
            raise ValueError("retry refused: job must be a saved terminal ERROR with compiler code 6056 and no collected results")
    pending = c["isa_sha256"] is None or retry is not None or any(not b.get("job_id") and b.get("observations") is None for b in manifest["batches"])
    if backend is None and pending:
        if service is None:
            service = runtime.make_service(options)
        backend = service.backend(c["backend"], use_fractional_gates=False)
    if pending and backend is not None:
        backend_readiness(backend, require_ready=True)
    if c["isa_sha256"] is None:
        publications = compile_plan(manifest, source, backend, folder)
    else:
        if backend is not None:
            if backend.name != c["backend"]:
                raise ValueError("backend differs from saved compilation")
        publications = Publications(manifest, folder, backend)
        publications.validate_all()
    if options.compile_only:
        print(f"Compiled only; no jobs submitted. Resume with --resume {folder/'manifest.json'} --execute", flush=True)
        return manifest
    if service is None and sampler_factory is None:
        service = runtime.make_service(options)
    if backend is not None:
        with tempfile.TemporaryDirectory(dir=folder, prefix="execution_calibration_") as temporary:
            runtime.save_backend_snapshot(backend, c["physical_path"], manifest["circuits"], Path(temporary))
            atomic_json(folder/"execution_backend_snapshot.json", json.loads((Path(temporary)/"backend_snapshot.json").read_text()))
    if retry is not None:
        backend_readiness(backend, require_ready=True)
        batch, details = retry
        batch.setdefault("failed_attempts", []).append(details)
        batch.update(job_id=None, status="not_submitted")
        atomic_json(folder/"manifest.json", manifest)
        print(f"Explicit retry enabled for failed job {options.retry_failed_job}; original ID/error/usage retained.", flush=True)
    runtime.collect_and_submit(manifest, folder, publications, service, backend, sampler_factory,
                               before_submit=lambda: backend_readiness(backend, require_ready=True))
    from Model2.Experiment5.analyze import main as analyze
    result = analyze(folder)
    bundle(folder)
    print(f"Complete result bundle: {folder/'results_bundle.zip'}", flush=True)
    return result


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
