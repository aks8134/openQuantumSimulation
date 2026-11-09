"""Recover analysis from saved Aer shots without importing or running Aer.

Reads one verified sample checkpoint at a time. Input files are read-only.
Multiple shot counts use prefixes of the same actual measurement outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def chunk_observables(samples, shots):
    """Per-path populations and currents/J, using only one small checkpoint."""
    values = np.empty((len(samples), 19), dtype=float)
    for site in range(10):
        values[:, site] = ((samples[:, 0, :shots] >> site) & 1).mean(axis=1)
    for bond in range(9):
        correlators = []
        for basis in (1, 2):
            packed = samples[:, basis, :shots]
            parity = ((packed >> bond) ^ (packed >> (bond + 1))) & 1
            correlators.append(1 - 2 * parity.mean(axis=1))
        values[:, 10 + bond] = .25 * (1 if bond % 2 == 0 else -1) * (correlators[1] - correlators[0])
    return values


def bounded_estimate(samples, chunk_paths=16):
    """Equivalent to the original estimator with bounded temporary arrays."""
    if samples.ndim != 3 or samples.shape[1] != 3 or chunk_paths < 1:
        raise ValueError("Expected paths × three bases × shots and a positive chunk size")
    measured = np.empty((len(samples), 19), dtype=float)
    for begin in range(0, len(samples), chunk_paths):
        end = min(begin + chunk_paths, len(samples))
        measured[begin:end] = chunk_observables(samples[begin:end], samples.shape[2])
    return measured


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def analyze(study_path, shot_counts=(32, 64), output_directory=None, plot=False):
    started = time.perf_counter()
    study_path = Path(study_path).resolve()
    original_study_hash = file_sha256(study_path)
    with study_path.open() as stream:
        study = json.load(stream)
    config = study["configuration"]
    paths, stored_shots, chunk_size = config["paths"], config["shots"], config["chunk"]
    shot_counts = sorted(set(shot_counts))
    if (paths < 2 or chunk_size < 1 or not shot_counts or
            any(not isinstance(s, int) or s < 1 or s > stored_shots for s in shot_counts)):
        raise ValueError("Requested shot counts must be positive and no larger than the saved shots per basis")
    print(f"Analysis only: {paths:,} saved trajectories, {stored_shots} shots/basis; "
          f"analyzing {shot_counts}. No simulation will run.", flush=True)
    print("Verifying source and loading the small per-path ideal reference...", flush=True)
    source_path = study_path.parent / "source.json"
    if file_sha256(source_path) != study["source_sha256"]:
        raise ValueError("Saved source hash mismatch")
    with source_path.open() as stream:
        source = json.load(stream)
    if (source["configuration"]["trajectories"] != paths or
            source["configuration"]["seed"] != config["seed"] or
            source["configuration"]["physics"]["n_qubits"] != 10 or
            source["configuration"]["physics"]["J"] != 1 or
            source["provenance"]["spec_sha256"] != config["spec_sha256"]):
        raise ValueError("Source configuration does not match this frozen ten-qubit J=1 study")
    covariances = np.asarray(source.pop("final_covariances_real_gauge"), dtype=float)
    if covariances.shape != (paths, 10, 10) or not np.isfinite(covariances).all():
        raise ValueError("Invalid saved ideal covariance reference")
    populations = np.diagonal(covariances, axis1=-2, axis2=-1)
    off = np.diagonal(covariances, offset=1, axis1=-2, axis2=-1)
    ideal = np.r_[populations.mean(axis=0), off.mean(axis=0)]
    variance_numerator = np.r_[(populations * (1 - populations)).mean(axis=0),
                              (.125 * (1 - 4 * off**2)).mean(axis=0)]
    infinity = np.r_[source["mean_channel_observables"]["populations"][0][-1],
                     source["mean_channel_observables"]["excitation_flows"][0][-1]]
    target = np.r_[source["stationary_observables"]["populations"],
                   source["stationary_observables"]["excitation_flows"]]
    del populations, off, covariances, source

    # 2.375 MiB per requested shot count at 16384 paths; no full shot/bit tensor.
    observed = {s: np.empty((paths, 19), dtype=float) for s in shot_counts}
    stop = 0
    rows = study["chunks"]
    expected_chunks = (paths + chunk_size - 1) // chunk_size
    if len(rows) != expected_chunks:
        raise ValueError(f"Incomplete checkpoints: expected {expected_chunks}, found {len(rows)}; no simulation was run")
    for index, row in enumerate(rows, 1):
        begin, end = row["begin"], row["end"]
        if begin != stop or end != min(begin + chunk_size, paths) or row["shots"] != stored_shots:
            raise ValueError("Checkpoint path coverage or shot count is inconsistent")
        archive = study_path.parent / f"samples_{begin:05d}_{end:05d}.npz"
        if file_sha256(archive) != row["sha256"]:
            raise ValueError(f"Sample hash mismatch: {archive.name}")
        with archive.with_suffix(".json").open() as stream:
            if json.load(stream) != row:
                raise ValueError(f"Checkpoint metadata mismatch: {archive.name}")
        with np.load(archive, allow_pickle=False) as data:
            samples = data["samples"]
        if (samples.shape != (end - begin, 3, stored_shots) or samples.dtype != np.uint16 or
                np.any(samples >= 1024)):
            raise ValueError(f"Invalid packed samples: {archive.name}")
        for shots in shot_counts:
            observed[shots][begin:end] = chunk_observables(samples, shots)
        del samples
        stop = end
        if index == 1 or index % 64 == 0 or index == len(rows):
            print(f"Verified/analyzed checkpoints {index}/{len(rows)}: paths {end:,}/{paths:,} "
                  f"({time.perf_counter()-started:.1f}s elapsed)", flush=True)
    if stop != paths or file_sha256(study_path) != original_study_hash:
        raise ValueError("Checkpoint coverage is incomplete or the input study changed during analysis")

    profiles = {}
    for shots, measured in observed.items():
        mean = measured.mean(axis=0)
        se = measured.std(axis=0, ddof=1) / np.sqrt(paths)
        target_error = float(np.max(abs(mean - target)))
        mean_error = float(np.max(abs(mean - infinity)))
        two_se = float(2 * np.max(se))
        profiles[str(shots)] = {
            "shots_per_basis": shots, "actual_outcomes_used": paths * 3 * shots,
            "observables": mean.tolist(), "across_path_standard_errors": se.tolist(),
            "conditional_shot_standard_errors": np.sqrt(np.maximum(variance_numerator, 0) / (paths * shots)).tolist(),
            "maximum_target_error": target_error, "maximum_mean_channel_error": mean_error,
            "maximum_same_path_error": float(np.max(abs(mean - ideal))), "maximum_two_SE": two_se,
            "maximum_population_target_error": float(np.max(abs(mean[:10] - target[:10]))),
            "maximum_current_target_error": float(np.max(abs(mean[10:] - target[10:]))),
            "empirical_checks_passed": bool(target_error <= .02 and mean_error <= .004 and two_se <= .004),
        }
    del observed
    result = {
        "analysis_complete": True, "simulation_rerun": False,
        "study_file": str(study_path), "study_sha256": original_study_hash,
        "source_sha256": study["source_sha256"], "original_study_complete_flag": study.get("complete", False),
        "configuration": config, "original_simulator_versions": study["versions"],
        "verification": {"checkpoint_count": len(rows), "paths": paths,
                         "stored_outcomes": paths * 3 * stored_shots,
                         "all_hashes_metadata_shapes_and_coverage_passed": True},
        "same_path_ideal": ideal.tolist(), "mean_channel": infinity.tolist(), "physical_target": target.tolist(),
        "total_saved_simulation_seconds": sum(row["seconds"] for row in rows), "profiles": profiles,
        "scope": "Actual saved Aer final-measurement outcomes. Smaller shot counts use prefixes of the same paths/bases. "
                 "Across-path two SE is descriptive; one realization does not estimate failure frequency. "
                 "No hardware noise model, controls, mitigation or new simulator executions.",
    }
    output = Path(output_directory).resolve() if output_directory else Path(__file__).parent / "analysis" / study_path.parent.name
    output.mkdir(parents=True, exist_ok=True)
    if output == study_path.parent:
        raise ValueError("Use an analysis output directory separate from the original checkpoints")
    with (output / "shot_comparison.csv").open("w", newline="") as stream:
        fields = ["shots_per_basis", "actual_outcomes_used", "maximum_target_error", "maximum_mean_channel_error",
                  "maximum_same_path_error", "maximum_two_SE", "empirical_checks_passed"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(profiles.values())
    with (output / "observables.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shots_per_basis", "kind", "site_or_bond", "observed", "across_path_SE",
                         "conditional_shot_SE", "same_path_ideal", "mean_channel", "physical_target"])
        for shots, profile in profiles.items():
            for j in range(19):
                writer.writerow([shots, "population" if j < 10 else "current_over_J", j + 1 if j < 10 else j - 9,
                                 profile["observables"][j], profile["across_path_standard_errors"][j],
                                 profile["conditional_shot_standard_errors"][j], ideal[j], infinity[j], target[j]])
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for ax, segment, label in [(axes[0], slice(0, 10), "Population"),
                                   (axes[1], slice(10, 19), "Current / J")]:
            x = np.arange(1, len(ideal[segment]) + 1)
            for shots, profile in profiles.items():
                ax.errorbar(x, np.asarray(profile["observables"])[segment],
                            yerr=2*np.asarray(profile["across_path_standard_errors"])[segment],
                            fmt="o-", capsize=3, label=f"Aer {shots} shots: descriptive ±2 across-path SE")
            ax.plot(x, ideal[segment], "k--", label="Exact same-path ideal")
            ax.plot(x, infinity[segment], "-.", color="gray", label="Infinite-path mean")
            ax.plot(x, target[segment], ":", color="green", label="Stationary physical target")
            ax.axhspan(target[segment][0] - .02, target[segment][0] + .02, color="green", alpha=.08)
            ax.set(xlabel="Site / bond (one based)", ylabel=label, xticks=x)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"shot_comparison.{suffix}", dpi=160)
        plt.close(fig)
    result["analysis_wall_seconds"] = time.perf_counter() - started
    result["analysis_process_peak_rss_mib"] = peak_rss_mib()
    atomic_json(output / "analysis.json", result)
    lines = ["# Recovered Aer shot analysis", "",
             f"**Verified all {len(rows):,} checkpoints: {paths:,} trajectories and {paths*3*stored_shots:,} actual saved outcomes.** No simulation was rerun.", "",
             "The original study and source files, checkpoint metadata and NPZ samples remain unchanged. The original complete flag describes the interrupted analysis; this separate report records successful recovery.", "",
             "| Shots per basis | Outcomes used | Max target error | Max mean-channel error | Max same-path error | Max descriptive 2 SE | Empirical checks passed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for shots, p in profiles.items():
        lines.append(f"| {shots} | {p['actual_outcomes_used']:,} | {p['maximum_target_error']:.10f} | "
                     f"{p['maximum_mean_channel_error']:.10f} | {p['maximum_same_path_error']:.10f} | "
                     f"{p['maximum_two_SE']:.10f} | {p['empirical_checks_passed']} |")
    lines += ["", "Original criteria: maximum local-observable error to the stationary target ≤0.02, distance to the infinite-path mean ≤0.004, and maximum descriptive across-path two SE ≤0.004. Observables are ten populations and nine currents/J.", "",
              "The smaller-shot profiles use the first requested outcomes from every trajectory and basis in the saved run. They are matched analyses of actual Aer shots, rather than new simulator executions. These realizations do not estimate a repeated-experiment failure frequency. Across-path error bars include both path and shot fluctuations and are descriptive, rather than simultaneous coverage guarantees.", "",
              f"Exact same-path target error: {float(max(abs(ideal-target))):.10f}; same-path distance to the infinite-path mean: {float(max(abs(ideal-infinity))):.10f}.", "",
              f"Analyzer wall time (excluding imports): {result['analysis_wall_seconds']:.2f} s. Peak process RSS: {result['analysis_process_peak_rss_mib']} MiB (when supported).", "",
              f"Input study: `{study_path}`. SHA256: `{original_study_hash}`.", "",
              "[Detailed JSON](analysis.json), [comparison CSV](shot_comparison.csv), [site and bond values](observables.csv)."]
    if plot:
        lines += ["", "![Actual Aer shot comparison](shot_comparison.png)"]
    (output / "Analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"analysis_complete": True, "output_directory": str(output),
                      "profiles": {s: {k: p[k] for k in ("maximum_target_error", "maximum_mean_channel_error",
                                  "maximum_same_path_error", "maximum_two_SE", "empirical_checks_passed")}
                                   for s, p in profiles.items()},
                      "analysis_wall_seconds": result["analysis_wall_seconds"],
                      "peak_rss_mib": result["analysis_process_peak_rss_mib"]}), flush=True)
    return result


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Existing study.json, including an interrupted final analysis")
    parser.add_argument("--shots", type=int, nargs="+", default=[32, 64], help="Prefixes of the saved outcomes to analyze")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    options = parser.parse_args(arguments)
    return analyze(options.study, options.shots, options.output_directory, options.plot)


if __name__ == "__main__":
    main()
