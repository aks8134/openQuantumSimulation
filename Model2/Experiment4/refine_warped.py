"""Checkpointed shorter-count challenges starting from a saved cosine.

Ordinary cosine is exactly the q=1 member of the warped family. Reusing
that feasible schedule is legitimate; each shorter count is independently
optimized and directly validated, with no seed/schedule selection on replays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library import reservoir_scaling as physics
from Model2.Experiment4 import cosine_search as experiment


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--counts-below", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    if args.iterations < 1 or args.counts_below < 1:
        raise ValueError("iterations/counts must be positive")
    raw = (args.source/"study.json").read_bytes()
    previous = json.loads(raw)
    config = previous["configuration"]
    baseline = previous["families"]["cosine"]["selected"]
    if not baseline["passed"]:
        raise ValueError("source cosine must pass direct mean validation")
    configuration = {**config, "families": ["warped_cosine"],
        "refinement": {"source_sha256": hashlib.sha256(raw).hexdigest(),
                       "iterations": args.iterations, "counts_below": args.counts_below,
                       "initial_schedule": "saved ordinary cosine, q=1"}}
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
    directory = args.source.parent/f"warped_refinement_{digest}"
    directory.mkdir(exist_ok=True)
    model = physics.Model(config["n_qubits"], config["gamma"]/config["J"],
                          config["p"], config["p"], config["exchange_prefactor"])
    if args.resume and (directory/"study.json").exists():
        study = json.loads((directory/"study.json").read_text())
    else:
        selected = experiment.evaluate(model, np.asarray(baseline["step_dts"]), config)
        selected.update(family="warped_cosine", parameters=baseline["parameters"]+[0.])
        study = {"experiment": "Model2/Experiment4", "configuration": configuration,
                 "families": {"warped_cosine": {"family": "warped_cosine",
                     "status": "shorter_count_search_running", "selected": selected,
                     "trials": [], "replays": [], "wall_seconds": 0.}}}
    record = study["families"]["warped_cosine"]
    bounds = experiment.parameter_bounds("warped_cosine", config)
    target = config["p"]*np.eye(model.n)
    for count in range(baseline["steps"]-1, max(0, baseline["steps"]-args.counts_below-1), -1):
        for seed in config["optimizer_seeds"]:
            if any(row["fixed_steps"] == count and row["seed"] == seed for row in record["trials"]):
                continue
            started = time.perf_counter()
            best = None
            evaluations = 0
            def objective(parameters):
                nonlocal best, evaluations
                evaluations += 1
                dts = experiment.schedule("warped_cosine", count, parameters)
                if dts.sum() > config["maximum_time"]:
                    return 1.+float(dts.sum()/config["maximum_time"])
                state, _ = experiment.evolve(model, dts, config)
                error = experiment.local_error(state, target, model)
                if error <= config["mean_tolerance"]:
                    error = experiment.continuation(model, state, float(dts[-1]), config).get("maximum_error", np.inf)
                if best is None or error < best["error"]:
                    best = {"parameters": list(map(float, parameters)), "error": float(error)}
                return error
            optimum = differential_evolution(objective, bounds, seed=seed, maxiter=args.iterations,
                popsize=config["population_multiplier"], polish=False, tol=1e-6)
            for start in (optimum.x, record["selected"]["parameters"]):
                minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                         options={"maxiter": 100, "xatol": 1e-7, "fatol": 1e-10})
            row = {"fixed_steps": count, "seed": seed, "evaluations": evaluations,
                   "optimizer_converged": bool(optimum.success), "best": best,
                   "wall_seconds": time.perf_counter()-started}
            record["trials"].append(row)
            record["wall_seconds"] += row["wall_seconds"]
            if best["error"] <= config["mean_tolerance"]:
                candidate = experiment.evaluate(model, experiment.schedule("warped_cosine", count, best["parameters"]), config)
                if candidate["passed"] and count < record["selected"]["steps"]:
                    candidate.update(family="warped_cosine", parameters=best["parameters"])
                    record["selected"] = candidate
            physics.atomic_json(directory/"study.json", study)
            experiment.write_report(study, directory)
            print(f"warped count={count}, seed={seed}, error={best['error']:.10g}, evaluations={evaluations}", flush=True)
    record["status"] = "mean_target_met"
    if not config["plan_only"]:
        record["sampling_passed"] = False
        for R in config["trajectories"]:
            existing = {(row["seed"], row["trajectories"]): row for row in record["replays"]}
            rows = []
            for seed in config["sampling_seeds"]:
                row = existing.get((seed, R))
                if row is None:
                    row = experiment.sampled_replay(model, record["selected"], config, seed, R)
                    record["replays"].append(row)
                    physics.atomic_json(directory/"study.json", study)
                rows.append(row)
            print(f"warped R={R}, checks={[row['passed'] for row in rows]}", flush=True)
            if all(row["passed"] for row in rows):
                record["sampling_passed"] = True
                break
    study["complete"] = True
    physics.atomic_json(directory/"study.json", study)
    experiment.write_report(study, directory)
    experiment.plot(study, directory)
    print(f"Saved {directory/'Study.md'}", flush=True)
    return study


if __name__ == "__main__":
    main()
