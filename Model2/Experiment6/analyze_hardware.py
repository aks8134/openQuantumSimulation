"""Analyze Model2 mitigation counts with independent calibration for each job."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model2.Experiment6 import run_hardware as experiment
from Model3.Experiment5 import mitigation as mit, run_mitigation as protocol


def analyze_counts(manifest, source):
    experiment.verify_manifest(manifest, source)
    if manifest["status"] != "collected":
        raise ValueError("analysis needs all mitigation jobs collected; saved counts are retained for resume")
    config = manifest["configuration"]
    R, L, K = config["trajectory_count"], len(config["fold_factors"]), config["randomizations"]
    controls = {arm: {name: [] for name in experiment.CONTROLS} for arm in ("raw", "twirled")}
    paths, groups, controls_by_job, job_ids = {}, [], [], []
    total = 0
    for batch in manifest["batches"]:
        records = manifest["circuits"][batch["start"]:batch["stop"]]
        observations = batch.get("observations", [])
        if batch["status"] != "collected" or not batch.get("job_id") or not batch["compiled"] or len(observations) != len(records):
            raise ValueError("incomplete mitigation batch or result length")
        local = {arm: {name: [] for name in experiment.CONTROLS} for arm in ("raw", "twirled")}
        groups.append([r["trajectory"] for r in records if r["kind"] == "trajectory" and r["arm"] == "raw"])
        job_ids.append(batch["job_id"])
        for row, observation in zip(records, observations):
            counts = mit.undo_measurement_twirl(observation["outcome_counts"], 10, row["measurement_mask"])
            if sum(counts.values()) != row["shots"]:
                raise ValueError("observed shot total differs from the plan")
            total += row["shots"]
            if row["kind"] == "control":
                controls[row["arm"]][row["control"]].append(counts)
                local[row["arm"]][row["control"]].append(counts)
            else:
                key = row["trajectory"], 0 if row["arm"] == "raw" else row["factor_index"]+1
                paths.setdefault(key, []).append(counts)
        for arm in ("raw", "twirled"):
            for name in experiment.CONTROLS:
                if len(local[arm][name]) != (1 if arm == "raw" else K):
                    raise ValueError("missing job-local preparation control variants")
        controls_by_job.append(local)
    if total != config["total_shots"] or len(paths) != R*(L+1):
        raise ValueError("missing matched trajectory/factor results")
    values = np.zeros((R, L+1, 10))
    covariances = np.zeros((R, L+1, 10, 10))
    for key, counts in paths.items():
        if len(counts) != (1 if key[1] == 0 else K):
            raise ValueError("missing matched randomization")
        values[key], covariances[key] = mit.group_moments(counts, 10)
    # Model2 covariances are (R,N,N); Model3's leading experiment axis is absent.
    ideal = np.asarray(source["final_covariances_real_gauge"]).diagonal(axis1=-2, axis2=-1)
    if ideal.shape != (R, 10):
        raise ValueError("same-path ideal covariance shape differs from Model2 input")
    target = np.asarray(source["stationary_observables"]["populations"])
    factors = [1+2*mit.fold_count(f, 880)/880 for f in config["fold_factors"]]
    statistics = mit.estimates(values, covariances, controls, factors, ideal, target,
        replicates=config["bootstrap"], seed=config["seed"], groups=groups)
    for metric in ("ideal_mae", "ideal_maximum"):
        statistics["comparisons"].append(dict(method="zne", reference="raw", metric=metric,
                                              difference=0., bootstrap_ci95=[0., 0.]))
    statistics = protocol.combine_job_calibrations(statistics, values, covariances, controls_by_job,
        groups, factors, ideal, target, replicates=config["bootstrap"], seed=config["seed"], job_ids=job_ids)
    return dict(experiment=experiment.EXPERIMENT, configuration=config, job_ids=job_ids,
        statistics=statistics, ideal_same_paths=ideal.mean(axis=0), physical_target=target,
        per_path_populations=values, per_path_ideal_populations=ideal,
        scope="Model2 frozen cosine44, half-mixed preparation, populations only. Descriptive paired-job/path and shared-control uncertainty; no currents or 0.02 certification.")


def analyze(folder):
    folder = Path(folder).resolve()
    _, manifest, source = experiment.read_saved(folder/"manifest.json")
    verified, missing = [], []
    store = experiment.Publications(manifest, folder)
    for batch in manifest["batches"]:
        path = folder/batch.get("isa_file", "missing.qpy")
        if path.exists():
            programs = store[batch["start"]:batch["stop"]]
            del programs
            verified.append(path.name)
        else:
            missing.append(path.name)
    report = analyze_counts(manifest, source)
    report.update(run_directory=str(folder), verification={"source_integrity": "exact_bytes",
        "compiled_batches_verified": verified, "compiled_batches_not_available": missing,
        "raw_counts_reconstructed": True, "calibration": "independent job-local correction before trajectory-weighted averaging"})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args()
    try:
        _, manifest, _ = experiment.read_saved(options.run_directory/"manifest.json")
        experiment.finish_analysis(options.run_directory, manifest, options.output_directory)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
