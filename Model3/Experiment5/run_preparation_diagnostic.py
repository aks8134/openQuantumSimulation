"""Fresh zero/one/mixture controls on the chain of a saved population run.

Default: 4096 shots for each of three controls, no Trotter evolution.
Hardware execution occurs only when the user invokes the runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model3.Experiment5 import run_hardware as hardware
from Model3.Experiment5.analyze_hardware import source_integrity

DEFAULT_REFERENCE = DIRECTORY/"hardware_runs/ibm_phoenix_prefix_20261024T162033_596311Z"
MODE = "fixed_chain_preparation_followup"


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--shots", "--control-shots", type=int, default=4096)
    parser.add_argument("--focus-qubits", type=int, nargs="+", default=[56, 76], help="physical system qubits to highlight; all sites are measured")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--max-control-error", type=float, default=.03)
    parser.add_argument("--control-alpha", type=float, default=.05)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    parser.add_argument("--analysis-output-directory", type=Path)
    options = parser.parse_args(arguments)
    if options.shots < 2 or options.batch_size < 1 or options.seed_transpiler < 0:
        raise ValueError("shots must be >=2, batch-size positive and transpiler seed nonnegative")
    if len(set(options.focus_qubits)) != len(options.focus_qubits) or min(options.focus_qubits) < 0:
        raise ValueError("focus-qubits must be distinct nonnegative physical qubits")
    if not 0 < options.max_control_error < 1 or not 0 < options.control_alpha < 1:
        raise ValueError("invalid control screening thresholds")
    if options.resume and options.compile_only:
        raise ValueError("resume and compile-only cannot be combined")
    return options


def reference(folder, focus):
    manifest_raw = (folder/"manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if (manifest.get("experiment") not in ("Model3/Experiment5/hardware", "Model3/Experiment5/prefix_diagnostic")
            or manifest.get("status") != "collected"):
        raise ValueError("reference-run must be a collected Model3 population or prefix run")
    config = manifest["configuration"]
    source_file = folder/"source.json"
    raw = source_file.read_bytes()
    integrity = source_integrity(raw, config["source_sha256"])
    source = json.loads(raw)
    hardware.require_validated_source(source)
    n = source["configuration"]["physics"]["n_qubits"]
    path = config.get("physical_path")
    if not path or len(path) != n+2 or len(set(path)) != len(path):
        raise ValueError("reference must contain a fixed boundary-ancilla chain")
    if not set(focus) <= set(path[1:-1]):
        raise ValueError("focus-qubits must be system sites on the saved chain")
    if not manifest["circuits"] or any(r["final_logical_to_physical"][:n] != path[1:-1] for r in manifest["circuits"]):
        raise ValueError("reference measurements disagree with the saved chain")
    metadata = dict(mode=MODE, reference_run=folder.name, reference_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        reference_source_sha256=config["source_sha256"], source_integrity=integrity,
        reference_job_ids=[b["job_id"] for b in manifest["batches"]], physical_path=path, focus_qubits=focus,
        interpretation="Fixed-chain preparation/readout followup. Current calibration is captured; no reranking or full-evolution calibration filter is applied to these zero-evolution controls.")
    return source_file, config["backend"], metadata


def main(options, *, backend=None, service=None, sampler_factory=None):
    if options.resume:
        manifest = json.loads(options.resume.read_bytes())
        config = manifest["configuration"]
        if (manifest.get("experiment") != "Model3/Experiment5/hardware" or not config.get("controls_only")
                or (config.get("path_selection") or {}).get("mode") != MODE):
            raise ValueError("resume requires a preparation-followup manifest")
        args = ["--resume", str(options.resume)]
        backend_name = config["backend"]
    else:
        source_file, backend_name, metadata = reference(options.reference_run, options.focus_qubits)
        args = ["--source", str(source_file), "--backend", backend_name, "--controls-only", "--bases", "z",
                "--trajectory-count", "1", "--shots", str(options.shots), "--control-shots", str(options.shots),
                "--batch-size", str(options.batch_size), "--max-control-error", str(options.max_control_error),
                "--control-alpha", str(options.control_alpha), "--optimization-level", str(options.optimization_level),
                "--seed-transpiler", str(options.seed_transpiler), "--physical-path", *map(str, metadata["physical_path"]),
                "--output-directory", str(options.output_directory)]
        if options.compile_only:
            args.append("--compile-only")
    if options.account_file:
        args.extend(["--account-file", str(options.account_file)])
    request = hardware.parse_arguments(args)
    if backend is None:
        service = service or hardware.make_service(request)
        backend = service.backend(backend_name, use_fractional_gates=False)
    if backend.name != backend_name:
        raise ValueError("backend must match the reference run")
    if not options.resume:
        refresh = getattr(backend, "refresh", None)
        if callable(refresh):
            refresh()
        request.path_selection = metadata
        request.required_system_sites = metadata["physical_path"][1:-1]
        print(f"Fixed physical path: {metadata['physical_path']}; focus qubits: {options.focus_qubits}", flush=True)
        print(f"Three preparation controls, {options.shots} shots each, {3*options.shots} total shots.", flush=True)
    result = hardware.main(request, backend=backend, service=service, sampler_factory=sampler_factory)
    if options.compile_only:
        return result
    from Model3.Experiment5 import analyze_preparation_diagnostic as analysis
    directory = Path(result["run_directory"])
    report = analysis.analyze(directory)
    output = options.analysis_output_directory or DIRECTORY/"analysis"/"hardware"/directory.name/"preparation"
    analysis.save(report, output)
    with zipfile.ZipFile(directory/"results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("source.json", "manifest.json", "hardware_results.json", "isa_circuits.qpy", "backend_snapshot.json", "selected_path.json", "ControlReport.md"):
            if (directory/name).exists():
                archive.write(directory/name, name)
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, "analysis/"+path.name)
    print(f"Preparation analysis: {output/'Analysis.md'}", flush=True)
    print(f"Bring back: {directory/'results_bundle.zip'}", flush=True)
    return result


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
