"""Export a small, portable hardware input from a validated full replay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.reservoir_scaling import atomic_json
from Model3.Experiment5.circuit_depth import validate_source


def export(source, indices, source_sha256):
    validate_source(source)
    if source.get("run_status") != "passed":
        raise ValueError("export requires a passing full replay")
    if (not indices or len(set(indices)) != len(indices) or min(indices) < 0 or
            max(indices) >= source["configuration"]["trajectories"]):
        raise ValueError("trajectory indices must be distinct and inside the full replay")
    result = {key: copy.deepcopy(source[key]) for key in (
        "experiment", "configuration", "preparation", "times", "schedule")}
    result.update(artifact_type="model3_hardware_input_v1", run_status="hardware_input",
        assessment={"validation_scope": "original_full_replay_only",
                    "interpretation": "the compact subset is not independently sampling-validated"},
        provenance={"full_replay_sha256": source_sha256,
            "full_replay_run_status": source["run_status"],
            "full_replay_trajectories": source["configuration"]["trajectories"],
            "saved_trajectory_indices": list(indices),
            "full_replay_assessment": copy.deepcopy(source.get("assessment", {})),
            "interpretation": "validation describes the original full replay; this subset has not independently passed its sampling budget"})
    result["configuration"]["trajectories"] = len(indices)
    result["schedule"]["jump_choices"] = [copy.deepcopy(source["schedule"]["jump_choices"][i]) for i in indices]
    for key in ("mean_channel_observables", "stationary_observables", "covariance_gauge", "schedule_method"):
        if key in source:
            result[key] = copy.deepcopy(source[key])
    if "final_covariances_real_gauge" in source:
        result["final_covariances_real_gauge"] = [[copy.deepcopy(preparation[i]) for i in indices]
            for preparation in source["final_covariances_real_gauge"]]
    validate_source(result)
    return result


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    if options.output.resolve() == options.source.resolve():
        raise ValueError("export must not overwrite the full replay")
    raw = options.source.read_bytes()
    result = export(json.loads(raw), options.trajectory_indices, hashlib.sha256(raw).hexdigest())
    options.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(options.output, result)
    print(f"Saved {len(options.trajectory_indices)} paths: {options.output} ({options.output.stat().st_size:,} bytes)")
    return result


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error
