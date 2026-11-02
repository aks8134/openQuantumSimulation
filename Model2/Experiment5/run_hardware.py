"""Frozen Model2 cosine44 hardware experiment; default is an offline plan.

--execute submits, --compile-only contacts the backend without submitting.
--resume reuses the saved source, compilation, jobs and per-job controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
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
from Model2.Experiment5 import input_data
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
    parser.add_argument("--resume", type=Path, metavar="MANIFEST")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    options = parser.parse_args(argv)
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
        allowed = {"--resume", "--execute", "--compile-only", "--account-file"}
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


def compile_plan(manifest, source, backend, folder):
    c = manifest["configuration"]
    if backend.name != c["backend"]:
        raise ValueError("resolved backend does not match the saved plan")
    if not {"cz", "reset", "measure", "rz", "sx", "x"} <= set(backend.target.operation_names):
        raise ValueError("backend requires CZ, reset, measure, RZ, SX and X support")
    settings = c["path_settings"]
    indices = list(range(c["trajectories"]))
    if c["requested_physical_path"] is None:
        selection = quality_path.select(backend, source, indices, avoid=settings["avoid_qubits"],
            min_t1=settings["min_t1_seconds"], min_t2=settings["min_t2_seconds"],
            max_cz_error=settings["max_cz_error"], max_readout_error=settings["max_readout_error"],
            max_expansions=settings["max_expansions"])
    else:
        selection = quality_path.validate(backend, c["requested_physical_path"], source, indices, settings)
    layout, path = circuits.chain_layout(backend, 10, "boundary", selection["physical_path"])
    atomic_json(folder/"selected_path.json", selection)
    print(f"Selected path: {path}", flush=True)
    bodies, controls, publications = {}, {}, []
    compiled_count = 0
    for row in manifest["circuits"]:
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
            # Each path is used only once; release its body after all bases.
            del bodies[key]
    with (folder/"isa_circuits.qpy").open("wb") as handle:
        qpy.dump(publications, handle)
    c.update(physical_path=path, isa_sha256=hashlib.sha256((folder/"isa_circuits.qpy").read_bytes()).hexdigest(),
             versions={name: importlib.metadata.version(name) for name in ("qiskit","qiskit-ibm-runtime","numpy")})
    runtime.save_backend_snapshot(backend, path, manifest["circuits"], folder)
    manifest.update(status="compiled", plan_sha256=plan_hash(manifest))
    atomic_json(folder/"manifest.json", manifest)
    write_plan(manifest, source, folder)
    validate_publications(manifest,publications)
    return publications


def validate_publications(manifest, publications):
    if len(publications) != len(manifest["circuits"]):
        raise ValueError("native circuit count differs from saved manifest")
    for circuit,row in zip(publications,manifest["circuits"]):
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


def read_saved(path):
    folder = path.resolve().parent
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != EXPERIMENT or manifest.get("plan_sha256") != plan_hash(manifest):
        raise ValueError("not an intact Model2 Experiment5 manifest")
    c = manifest["configuration"]
    raw = (folder/"source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != c["source_sha256"]:
        raise ValueError("saved source hash mismatch")
    spec = input_data.load_spec(folder/"spec.json")
    if input_data.digest(spec) != c["spec_sha256"]:
        raise ValueError("saved specification changed")
    if c["isa_sha256"] is not None and hashlib.sha256((folder/"isa_circuits.qpy").read_bytes()).hexdigest() != c["isa_sha256"]:
        raise ValueError("saved native circuit hash mismatch")
    return folder, manifest, json.loads(raw)


def bundle(folder):
    names = ["manifest.json","source.json","spec.json","Plan.md","isa_circuits.qpy","backend_snapshot.json",
             "selected_path.json","execution_backend_snapshot.json","hardware_results.json",
             "Analysis.md","observables.csv","observables.png"]
    with zipfile.ZipFile(folder/"results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if (folder/name).exists():
                archive.write(folder/name, arcname=name)


def main(options, *, backend=None, service=None, sampler_factory=None):
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
        manifest["configuration"]["source_sha256"] = hashlib.sha256((folder/"source.json").read_bytes()).hexdigest()
        manifest["plan_sha256"] = plan_hash(manifest)
        atomic_json(folder/"manifest.json", manifest)
        write_plan(manifest, source, folder)
    c = manifest["configuration"]
    print(f"Plan: {folder/'manifest.json'}", flush=True)
    print(f"Workload: {c['total_jobs']} jobs, {c['total_circuits']} circuits, {c['total_shots']} shots", flush=True)
    print(f"Exact finite-path assessment: {source['finite_path_assessment']}", flush=True)
    if not options.execute and not options.compile_only:
        print("Offline plan only; no account loaded, backend queried or jobs submitted.", flush=True)
        return manifest
    if manifest["status"] == "collected":
        from Model2.Experiment5.analyze import main as analyze
        result = analyze(folder)
        bundle(folder)
        return result
    if backend is None and (c["isa_sha256"] is None or any(not b.get("job_id") and b.get("observations") is None for b in manifest["batches"])):
        if service is None:
            service = runtime.make_service(options)
        backend = service.backend(c["backend"], use_fractional_gates=False)
    if c["isa_sha256"] is None:
        publications = compile_plan(manifest, source, backend, folder)
    else:
        with (folder/"isa_circuits.qpy").open("rb") as handle:
            publications = qpy.load(handle)
        validate_publications(manifest,publications)
        if backend is not None:
            if backend.name != c["backend"]:
                raise ValueError("backend differs from saved compilation")
            for circuit in publications:
                circuits.assert_native(circuit, backend)
    if options.compile_only:
        print(f"Compiled only; no jobs submitted. Resume with --resume {folder/'manifest.json'} --execute", flush=True)
        return manifest
    if service is None and sampler_factory is None:
        service = runtime.make_service(options)
    if backend is not None:
        with tempfile.TemporaryDirectory(dir=folder, prefix="execution_calibration_") as temporary:
            runtime.save_backend_snapshot(backend, c["physical_path"], manifest["circuits"], Path(temporary))
            atomic_json(folder/"execution_backend_snapshot.json", json.loads((Path(temporary)/"backend_snapshot.json").read_text()))
    runtime.collect_and_submit(manifest, folder, publications, service, backend, sampler_factory)
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
