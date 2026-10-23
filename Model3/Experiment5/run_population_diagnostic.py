"""Two-stage population workflow: calibrated controls, then reuse their path.

Invoking this script accesses IBM only when no backend is explicitly injected.
The controls stage never submits evolved circuits; the full stage is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model3.Experiment5 import run_hardware as hardware
from Model3.Experiment5 import quality_path
from Model3.Experiment5.analyze_hardware import source_integrity

DEFAULT_SOURCE = DIRECTORY/"inputs/cosine72_seed29_paths0-127.json"
DEFAULT_SELECTION = DIRECTORY/"hardware_runs/latest_population_controls.json"


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("controls", "full"))
    parser.add_argument("--backend", help="controls default: ibm_phoenix; full must match saved controls")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="controls input; full reuses the saved source")
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION,
                        help="controls write this pointer; full reads it to locate collected controls")
    parser.add_argument("--trajectory-count", type=int, default=128)
    parser.add_argument("--shots", type=int, default=64, help="shots per full-run trajectory")
    parser.add_argument("--control-shots", type=int, default=2048)
    parser.add_argument("--avoid-qubits", nargs="*", type=int, default=[7])
    parser.add_argument("--min-t1-us", type=float, default=80.)
    parser.add_argument("--min-t2-us", type=float, default=60.)
    parser.add_argument("--max-cz-error", type=float, default=.005)
    parser.add_argument("--max-readout-error", type=float, default=.02)
    parser.add_argument("--max-path-expansions", type=int, default=2_000_000)
    parser.add_argument("--max-control-error", type=float, default=.03)
    parser.add_argument("--control-alpha", type=float, default=.05)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"hardware_runs")
    options = parser.parse_args(arguments)
    if min(options.trajectory_count, options.shots, options.batch_size, options.max_path_expansions) < 1:
        raise ValueError("counts, shots, batch size and search limit must be positive")
    if options.control_shots < 2 or options.seed_transpiler < 0:
        raise ValueError("need at least two control shots and a nonnegative transpiler seed")
    if not (0 < options.max_control_error < 1 and 0 < options.control_alpha < 1
            and 0 < options.max_cz_error < 1 and 0 < options.max_readout_error < 1
            and options.min_t1_us > 0 and options.min_t2_us > 0):
        raise ValueError("invalid quality thresholds")
    if options.backend is not None and not options.backend.strip():
        raise ValueError("backend cannot be empty")
    if any(q < 0 for q in options.avoid_qubits):
        raise ValueError("avoid-qubits must be nonnegative")
    return options


def load_controls(selection_file):
    pointer = json.loads(selection_file.read_text())
    if pointer.get("artifact_type") != "model3_population_controls_v1":
        raise ValueError("selection-file is not a population control pointer")
    manifest_path = Path(pointer["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = selection_file.resolve().parent/manifest_path
    folder = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest["configuration"]
    if (manifest.get("experiment") != "Model3/Experiment5/hardware" or manifest.get("status") != "collected"
            or not cfg.get("controls_only")):
        raise ValueError("full run requires completely collected controls-only results")
    source_file = folder/"source.json"
    raw = source_file.read_bytes()
    source_integrity(raw, cfg["source_sha256"])
    source = json.loads(raw)
    hardware.require_validated_source(source)
    if hashlib.sha256((folder/"isa_circuits.qpy").read_bytes()).hexdigest() != cfg["isa_sha256"]:
        raise ValueError("saved controls QPY hash mismatch")
    records, observations, stop = manifest["circuits"], [], 0
    for batch in manifest["batches"]:
        if batch["status"] != "collected" or batch["start"] != stop or not batch.get("job_id"):
            raise ValueError("incomplete control batches")
        if len(batch["observations"]) != batch["stop"]-batch["start"]:
            raise ValueError("control batch range mismatch")
        stop = batch["stop"]; observations.extend(batch["observations"])
    if stop != len(records) or len(records) != 3:
        raise ValueError("expected three collected controls")
    expected_targets = {"all_zero": [0.]*source["configuration"]["physics"]["n_qubits"],
                        "all_one": [1.]*source["configuration"]["physics"]["n_qubits"],
                        "initial_mixture": source["preparation"]["populations"]}
    for index, (record, row) in enumerate(zip(records, observations)):
        if record.get("kind") != "control" or record["pub_index"] != index or record["basis"] != "z":
            raise ValueError("unexpected circuit in controls-only run")
        if record["expected_populations"] != expected_targets.get(record["control"]):
            raise ValueError("control targets disagree with source")
        if sum(row["outcome_counts"].values()) != record["shots"]:
            raise ValueError("control shot count mismatch")
    summary = hardware.summarize(source, records, observations,
        max_control_error=cfg.get("max_control_error", .03), control_alpha=cfg.get("control_alpha", .05))
    if not summary["control_assessment"]["passed"]:
        raise ValueError(f"control screening failed at sites {summary['control_assessment']['failed_sites_1based']}; no full run submitted")
    selection = cfg.get("path_selection")
    if not selection or selection["physical_path"] != cfg["physical_path"]:
        raise ValueError("missing or inconsistent calibrated path selection")
    if records[0]["final_logical_to_physical"][:-2] != cfg["physical_path"][1:-1]:
        raise ValueError("control measurement mapping differs from selected path")
    return manifest_path, manifest, source_file, source, summary


def main(options, *, backend=None, service=None, sampler_factory=None):
    indices = list(range(options.trajectory_count))
    reused = None
    if options.stage == "controls":
        source_file = options.source
        source = json.loads(source_file.read_bytes())
        hardware.require_validated_source(source)
        backend_name = options.backend or "ibm_phoenix"
    else:
        path, manifest, source_file, source, summary = load_controls(options.selection_file)
        backend_name = manifest["configuration"]["backend"]
        if options.backend is not None and options.backend != backend_name:
            raise ValueError("backend must match the collected control run")
        reused = (path, manifest, summary)
    if options.trajectory_count > source["configuration"]["trajectories"]:
        raise ValueError("trajectory-count exceeds saved source")
    if backend is None:
        if service is None:
            service = hardware.make_service(options)
        backend = service.backend(backend_name, use_fractional_gates=False)
    refresh = getattr(backend, "refresh", None)
    if callable(refresh):
        refresh()
    if reused is None:
        selection = quality_path.select(backend, source, indices, avoid=options.avoid_qubits,
            min_t1=options.min_t1_us*1e-6, min_t2=options.min_t2_us*1e-6,
            max_cz_error=options.max_cz_error, max_readout_error=options.max_readout_error,
            max_expansions=options.max_path_expansions)
        selection["planned_trajectory_indices"] = indices
    else:
        path, manifest, summary = reused
        original = manifest["configuration"]["path_selection"]
        if indices != original["planned_trajectory_indices"]:
            raise ValueError("full trajectory selection must match the planned control-stage selection")
        selection = quality_path.validate(backend, original["physical_path"], source, indices, original["settings"])
        selection.update(control_manifest=str(path.resolve()), control_assessment=summary["control_assessment"],
                         planned_trajectory_indices=indices)
    print(f"Selected physical path: {selection['physical_path']}", flush=True)
    print(f"Path calibration metrics: {selection['metrics']}", flush=True)
    args = ["--source", str(source_file), "--backend", backend_name, "--trajectory-count", str(options.trajectory_count),
            "--bases", "z", "--shots", str(options.control_shots if options.stage == "controls" else options.shots),
            "--control-shots", str(options.control_shots), "--max-control-error", str(options.max_control_error),
            "--control-alpha", str(options.control_alpha), "--batch-size", str(options.batch_size),
            "--physical-path", *map(str, selection["physical_path"]), "--optimization-level", str(options.optimization_level),
            "--seed-transpiler", str(options.seed_transpiler), "--output-directory", str(options.output_directory)]
    if options.stage == "controls":
        args.append("--controls-only")
    if options.compile_only:
        args.append("--compile-only")
    request = hardware.parse_arguments(args)
    request.path_selection = selection
    request.required_system_sites = selection["physical_path"][1:-1]
    if options.stage == "controls":
        request.control_pointer_file = options.selection_file
    result = hardware.main(request, service=service, backend=backend, sampler_factory=sampler_factory)
    if options.stage == "controls" and not options.compile_only:
        hardware.save_control_pointer(options.selection_file, Path(result["run_directory"]),
                                      result["configuration"], result["control_assessment"]["passed"])
        print(f"Saved control-run pointer: {options.selection_file}", flush=True)
        print("Controls stage finished. No full evolution circuits were submitted.", flush=True)
    return result


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
