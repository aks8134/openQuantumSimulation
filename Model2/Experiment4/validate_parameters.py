"""Validate and replay the best saved candidate at each selected physical rate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library import reservoir_scaling as physics
from Model2.Experiment4 import cosine_search as experiment
from Model2.Experiment4 import analyze_results as independent


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--ratios", type=float, nargs="+", default=[np.pi/8, .5, 2.])
    parser.add_argument("--trajectories", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.trajectories) < 2:
        raise ValueError("trajectories must be >=2")
    sources = [(path, json.loads(path.read_text())) for path in args.sources]
    folders = []
    for ratio in args.ratios:
        candidates = [(path, key, case) for path, source in sources for key, case in source["cases"].items()
                      if case.get("complete") and case.get("selected") is not None
                      and np.isclose(case["configuration"]["gamma"], ratio, rtol=1e-12, atol=1e-14)]
        if not candidates:
            raise ValueError(f"no completed candidate for ratio {ratio}")
        source, key, case = min(candidates, key=lambda row: (row[2]["selected"]["steps"],
                                                          row[2]["selected"]["forward"]["maximum_error"]))
        config = {**case["configuration"], "trajectories": sorted(args.trajectories)}
        selected = case["selected"]
        config["families"] = [selected["family"]]
        spec = {"configuration": config, "step_dts": selected["step_dts"],
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_case": key}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        folder = source.parent.parent/f"parameter_validation_{digest}"
        folder.mkdir(exist_ok=True)
        folders.append(folder)
        file = folder/"study.json"
        model = physics.Model(config["n_qubits"], config["gamma"], config["p"], config["p"], .25)
        if args.resume and file.exists():
            study = json.loads(file.read_text())
        else:
            fresh = experiment.evaluate(model, np.asarray(selected["step_dts"]), config)
            if not fresh["passed"]:
                raise ValueError("candidate failed fresh affine check")
            fresh.update(family=selected["family"], parameters=selected["parameters"])
            study = {"experiment": "Model2/Experiment4", "configuration": config,
                     "source": spec, "families": {selected["family"]: {
                         "selected": fresh, "replays": [], "sampling_passed": False}}}
        record = study["families"][selected["family"]]
        for R in sorted(args.trajectories):
            rows = []
            for seed in config["sampling_seeds"]:
                old = [r for r in record["replays"] if r["seed"] == seed and r["trajectories"] == R]
                if old:
                    row = old[0]
                else:
                    row = experiment.sampled_replay(model, record["selected"], config, seed, R)
                    record["replays"].append(row)
                    physics.atomic_json(file, study)
                rows.append(row)
            print(f"ratio={ratio:.12g}, steps={record['selected']['steps']}, R={R}: {[row['passed'] for row in rows]}", flush=True)
            if all(r["passed"] for r in rows):
                record["sampling_passed"] = True
                break
        study["complete"] = True
        physics.atomic_json(file, study)
        experiment.write_report(study, folder)
        experiment.plot(study, folder)
        independent.main(folder)
        print(f"Saved {folder/'Study.md'}", flush=True)
    return folders


if __name__ == "__main__":
    main()
