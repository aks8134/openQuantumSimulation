"""Compare saved Model3 trajectory prefixes on one fixed physical chain.

Defaults: 32 trajectories x prefixes 0/12/24/48/72 x 128 Z shots.
Only explicit CLI execution connects to IBM; tests can inject a local backend.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from qiskit import qpy, transpile

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import reservoir_scaling as physics
from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import circuit_depth as circuits, quality_path, run_hardware as hardware
from Model3.Experiment5.analyze_hardware import source_integrity

EXPERIMENT = "Model3/Experiment5/prefix_diagnostic"
DEFAULT_REFERENCE = DIRECTORY/"hardware_runs/ibm_phoenix_20261024T042722_287349Z"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE,
                        help="collected full run providing source and physical path")
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[0, 12, 24, 48, 72])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trajectory-count", type=int, help="first N saved paths; default 32")
    selection.add_argument("--trajectory-indices", type=int, nargs="+")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--order-seed", type=int, default=29, help="shuffle all prefix/path pairs before submission")
    parser.add_argument("--compile-only", action="store_true", help="resolve target and save circuits; submit no jobs")
    parser.add_argument("--resume", type=Path, help="resume saved prefix manifest with its original circuits and shots")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    parser.add_argument("--analysis-output-directory", type=Path, help="optional separate destination for plots and report")
    options = parser.parse_args(arguments)
    if options.trajectory_indices is None:
        count = 32 if options.trajectory_count is None else options.trajectory_count
        if count < 1:
            raise ValueError("trajectory-count must be positive")
        options.trajectory_indices = list(range(count))
    for name in ("prefix_steps", "trajectory_indices"):
        values = getattr(options, name)
        if not values or min(values) < 0 or len(values) != len(set(values)):
            raise ValueError(f"{name} must contain distinct nonnegative integers")
    options.prefix_steps.sort()
    if options.shots < 2 or options.batch_size < 1 or min(options.seed_transpiler, options.order_seed) < 0:
        raise ValueError("shots must be >=2, batch-size positive, and seeds nonnegative")
    if options.resume and options.compile_only:
        raise ValueError("resume and compile-only cannot be combined")
    return options


def load_reference(folder, indices, prefixes):
    raw_manifest = (folder/"manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    config = manifest["configuration"]
    if (manifest.get("experiment") != "Model3/Experiment5/hardware" or manifest.get("status") != "collected"
            or config.get("controls_only") or config.get("layout") != "chain"):
        raise ValueError("reference-run must be a collected full run with a fixed chain")
    raw = (folder/"source.json").read_bytes()
    integrity = source_integrity(raw, config["source_sha256"])
    source = json.loads(raw)
    hardware.require_validated_source(source)
    if max(prefixes) > len(source["schedule"]["step_dts"]):
        raise ValueError("prefix exceeds saved schedule")
    if not set(indices) <= set(config["trajectory_indices"]):
        raise ValueError("trajectory selection must be a subset of the reference hardware run")
    n = source["configuration"]["physics"]["n_qubits"]
    path = config.get("physical_path")
    selection = config.get("path_selection")
    if not path or not selection or selection["physical_path"] != path or len(path) != n+2:
        raise ValueError("reference needs its calibrated boundary-ancilla path")
    records = [row for row in manifest["circuits"] if row.get("kind") != "control"]
    if not records or any(row["final_logical_to_physical"][:n] != path[1:-1] for row in records):
        raise ValueError("reference readout mapping disagrees with physical path")
    provenance = dict(reference_run=folder.name, manifest_sha256=digest(raw_manifest),
                      original_source_sha256=config["source_sha256"], source_integrity=integrity,
                      backend=config["backend"], physical_path=path,
                      reference_job_ids=[b["job_id"] for b in manifest["batches"]])
    return raw, source, config, provenance


def ideal_references(source, indices, prefixes):
    """Exact finite-step covariance channels, with saved choices and signed cosines.

    No shots or density-matrix simulator: each retained path uses its own
    deterministic collision/reset channel, including ignored bath outcomes.
    """
    circuits.validate_source(source)
    config, p = source["configuration"], source["configuration"]["physics"]
    model = physics.Model(p["n_qubits"], p["gamma"]/p["J"], p["p_left"], p["p_right"], p["exchange_prefactor"])
    initial = np.diag(source["preparation"]["populations"])
    cov = np.repeat(initial[None, None, :, :], len(indices), axis=1)
    mean = initial.copy()
    choices = np.asarray(source["schedule"]["jump_choices"])[indices]
    rows = []
    def retain(step):
        rows.append(dict(prefix_steps=step, physical_time=float(sum(source["schedule"]["step_dts"][:step])),
                         per_trajectory_populations=np.diagonal(cov[0], axis1=-2, axis2=-1).tolist(),
                         mean_channel_populations=np.diag(mean).tolist()))
    if 0 in prefixes:
        retain(0)
    for step, dt in enumerate(source["schedule"]["step_dts"][:max(prefixes)], 1):
        branch = physics.channel(model, p["J"]*dt, config["sampling_policy"], config["order"], config["system_formula"])
        cov = physics.sampled_step(cov, choices[:, step-1], branch)
        mean = physics.mean_step(mean, branch)
        if step in prefixes:
            retain(step)
    full = len(source["schedule"]["step_dts"])
    if max(prefixes) == full and "final_covariances_real_gauge" in source:
        saved = np.asarray(source["final_covariances_real_gauge"]).reshape(-1, model.n, model.n)[indices]
        np.testing.assert_allclose(cov[0], saved, atol=2e-12, rtol=2e-12,
                                   err_msg="recomputed full prefix disagrees with saved ideal covariance")
    return dict(artifact_type="model3_prefix_ideal_v1", trajectory_indices=indices, prefixes=rows,
                physical_stationary_populations=np.diag(physics.exact_stationary(model)).tolist(),
                full_schedule_steps=full,
                interpretation="Ideal predictions use each prefix's saved timesteps and branch choices; early prefixes are transient, not steady-state tests.")


def prepare(source, backend, options, path):
    n = source["configuration"]["physics"]["n_qubits"]
    layout, path = circuits.chain_layout(backend, n, "boundary", path)
    if not {"reset", "measure"} <= backend.target.operation_names:
        raise ValueError("backend must support reset and measure")
    pairs = [(step, index) for step in options.prefix_steps for index in options.trajectory_indices]
    order = np.random.default_rng(options.order_seed).permutation(len(pairs))
    publications, records = [], []
    for position in order:
        step, index = pairs[position]
        logical = circuits.build_body(source, index, steps=step, ancillas="boundary")
        native = transpile(logical, backend=backend, initial_layout=layout,
                           optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler)
        mapping = native.layout.final_index_layout(filter_ancillas=True)
        if mapping[:n] != path[1:-1]:
            raise ValueError("prefix readout mapping changed from reference chain")
        circuit = circuits.append_readout(native, mapping[:n], "z")
        circuits.assert_native(circuit, backend)
        publications.append(circuit)
        records.append(dict(pub_index=len(records), trajectory=index, prefix_steps=step,
                            physical_time=float(sum(source["schedule"]["step_dts"][:step])), basis="z", shots=options.shots,
                            final_logical_to_physical=mapping, metrics=circuits.metrics(circuit)))
        print(f"Compiled {len(records)}/{len(pairs)}: path {index}, prefix {step}", flush=True)
    return publications, records


def bundle(directory, analysis=None):
    with zipfile.ZipFile(directory/"results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("source.json", "ideal_prefixes.json", "reference_run.json", "manifest.json", "isa_circuits.qpy",
                     "backend_snapshot.json", "selected_path.json", "hardware_results.json"):
            if (directory/name).exists():
                archive.write(directory/name, name)
        if analysis:
            for path in sorted(analysis.iterdir()):
                if path.is_file():
                    archive.write(path, "analysis/"+path.name)


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.resume:
        directory = options.resume.resolve().parent
        manifest = json.loads(options.resume.read_bytes())
        if manifest.get("experiment") != EXPERIMENT:
            raise ValueError("resume requires a prefix diagnostic manifest")
        config = manifest["configuration"]
        for name, key in (("source.json", "source_sha256"), ("ideal_prefixes.json", "ideal_sha256")):
            source_integrity((directory/name).read_bytes(), config[key])
        raw = (directory/"source.json").read_bytes()
        source = json.loads(raw)
        hardware.require_validated_source(source)
        from Model3.Experiment5.analyze_prefix_diagnostic import validate_records
        validate_records(manifest, source)
        if digest((directory/"isa_circuits.qpy").read_bytes()) != config["isa_sha256"]:
            raise ValueError("saved compiled circuits changed")
        with (directory/"isa_circuits.qpy").open("rb") as stream:
            publications = qpy.load(stream)
        if len(publications) != len(manifest["circuits"]):
            raise ValueError("compiled circuit count mismatch")
        for circuit, record in zip(publications, manifest["circuits"]):
            if (circuit.metadata.get("trajectory") != record["trajectory"]
                    or circuit.metadata.get("steps") != record["prefix_steps"]):
                raise ValueError("compiled prefix metadata disagrees with manifest")
        pending = any(b.get("observations") is None for b in manifest["batches"])
        if pending:
            if backend is None:
                service = service or hardware.make_service(options)
                backend = service.backend(config["backend"], use_fractional_gates=False)
            if backend.name != config["backend"]:
                raise ValueError("resume backend differs from saved backend")
            for circuit in publications:
                circuits.assert_native(circuit, backend)
    else:
        raw, source, original, provenance = load_reference(options.reference_run, options.trajectory_indices, options.prefix_steps)
        ideal = ideal_references(source, options.trajectory_indices, options.prefix_steps)
        if backend is None:
            service = service or hardware.make_service(options)
            backend = service.backend(original["backend"], use_fractional_gates=False)
        if backend.name != original["backend"]:
            raise ValueError("backend must match reference run")
        refresh = getattr(backend, "refresh", None)
        if callable(refresh):
            refresh()
        selected = quality_path.validate(backend, original["physical_path"], source,
                                         options.trajectory_indices, original["path_selection"]["settings"])
        selected["mode"] = "reuse_prefix_reference_path"
        publications, records = prepare(source, backend, options, selected["physical_path"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = options.output_directory/f"{original['backend']}_prefix_{stamp}"
        directory.mkdir(parents=True, exist_ok=False)
        (directory/"source.json").write_bytes(raw)
        atomic_json(directory/"ideal_prefixes.json", ideal)
        atomic_json(directory/"reference_run.json", provenance)
        atomic_json(directory/"selected_path.json", selected)
        with (directory/"isa_circuits.qpy").open("wb") as stream:
            qpy.dump(publications, stream)
        hardware.save_backend_snapshot(backend, selected["physical_path"], records, directory)
        config = dict(backend=original["backend"], physical_path=selected["physical_path"], path_selection=selected,
                      source_sha256=digest(raw), ideal_sha256=digest((directory/"ideal_prefixes.json").read_bytes()),
                      isa_sha256=digest((directory/"isa_circuits.qpy").read_bytes()),
                      trajectory_indices=options.trajectory_indices, prefix_steps=options.prefix_steps,
                      shots=options.shots, batch_size=options.batch_size, measurement_bases=["z"],
                      total_circuits=len(records), total_shots=len(records)*options.shots,
                      optimization_level=options.optimization_level, seed_transpiler=options.seed_transpiler,
                      order_seed=options.order_seed, execution_order="seeded permutation of all prefix/path pairs",
                      versions={name: importlib.metadata.version(name) for name in ("qiskit", "qiskit-ibm-runtime")})
        manifest = dict(experiment=EXPERIMENT, status="compiled", configuration=config, circuits=records,
                        batches=[dict(start=start, stop=min(start+options.batch_size, len(records)), job_id=None, status="not_submitted")
                                 for start in range(0, len(records), options.batch_size)])
        atomic_json(directory/"manifest.json", manifest)
        print(f"Saved {len(records)} circuits, {config['total_shots']} shots, {len(manifest['batches'])} batch(es): {directory}", flush=True)
        if options.compile_only:
            bundle(directory)
            print("Compile only: no jobs submitted.", flush=True)
            return manifest
    if any(b.get("observations") is None for b in manifest["batches"]) and service is None and sampler_factory is None:
        raise ValueError("Runtime service required for execution")
    hardware.collect_and_submit(manifest, directory, publications, service, backend, sampler_factory)
    # The analyzer independently reconstructs raw counts and verifies references.
    from Model3.Experiment5 import analyze_prefix_diagnostic as analysis
    report = analysis.analyze(directory)
    atomic_json(directory/"hardware_results.json", analysis.serializable(report))
    output = options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/directory.name
    analysis.save(report, output)
    bundle(directory, output)
    print(f"Prefix analysis: {output/'Analysis.md'}", flush=True)
    print(f"Results including QPY: {directory/'results_bundle.zip'}", flush=True)
    return report


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
