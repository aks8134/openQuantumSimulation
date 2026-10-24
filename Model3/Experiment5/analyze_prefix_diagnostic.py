"""Reconstruct and plot collected Model3 prefix populations from saved files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model3.Experiment5.analyze_hardware import moments, serializable, source_integrity
from Model3.Experiment5.analyze_populations import population_error, write_csv
from Model3.Experiment5.run_prefix_diagnostic import EXPERIMENT, digest, ideal_references


def validate_records(manifest, source):
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("need a Model3 prefix diagnostic manifest")
    config, records = manifest["configuration"], manifest["circuits"]
    indices, prefixes = config["trajectory_indices"], config["prefix_steps"]
    n = source["configuration"]["physics"]["n_qubits"]
    if (not indices or not prefixes or len(set(indices)) != len(indices) or len(set(prefixes)) != len(prefixes)
            or min(indices) < 0 or max(indices) >= source["configuration"]["trajectories"]
            or min(prefixes) < 0 or max(prefixes) > len(source["schedule"]["step_dts"])
            or config["shots"] < 2):
        raise ValueError("invalid saved trajectory, prefix or shot selection")
    path = config["physical_path"]
    if len(path) != n+2 or len(set(path)) != n+2:
        raise ValueError("invalid physical chain")
    expected = {(step, index) for step in prefixes for index in indices}
    found = []
    for position, row in enumerate(records):
        if (row["pub_index"] != position or row["basis"] != "z" or row["shots"] != config["shots"]
                or row["final_logical_to_physical"] != path[1:-1]+[path[0], path[-1]]):
            raise ValueError("invalid publication index, basis, shots or physical mapping")
        step = row["prefix_steps"]
        if not np.isclose(row["physical_time"], sum(source["schedule"]["step_dts"][:step]), atol=1e-12, rtol=1e-12):
            raise ValueError("prefix physical time mismatch")
        found.append((step, row["trajectory"]))
    if set(found) != expected or len(found) != len(expected):
        raise ValueError("missing or duplicate prefix/trajectory records")
    if config["total_circuits"] != len(records) or config["total_shots"] != len(records)*config["shots"]:
        raise ValueError("manifest circuit or shot totals mismatch")
    stop = 0
    for batch in manifest["batches"]:
        if batch["start"] != stop or not stop < batch["stop"] <= len(records):
            raise ValueError("unordered or incomplete batch coverage")
        stop = batch["stop"]
    if stop != len(records):
        raise ValueError("missing batch coverage")


def analyze(folder):
    raw = {name: (folder/name).read_bytes() for name in ("manifest.json", "source.json", "ideal_prefixes.json")}
    manifest, source, saved_ideal = [json.loads(raw[name]) for name in raw]
    if manifest.get("status") != "collected":
        raise ValueError("analysis requires completely collected results")
    config = manifest["configuration"]
    validate_records(manifest, source)
    integrity = {name: source_integrity(raw[name], config[key]) for name, key in
                 (("source.json", "source_sha256"), ("ideal_prefixes.json", "ideal_sha256"))}
    ideal = ideal_references(source, config["trajectory_indices"], config["prefix_steps"])
    if (saved_ideal["trajectory_indices"] != ideal["trajectory_indices"]
            or len(saved_ideal["prefixes"]) != len(ideal["prefixes"])):
        raise ValueError("ideal prefix selection mismatch")
    for saved, actual in zip(saved_ideal["prefixes"], ideal["prefixes"]):
        if saved["prefix_steps"] != actual["prefix_steps"]:
            raise ValueError("ideal prefix order mismatch")
        for key in ("physical_time", "per_trajectory_populations", "mean_channel_populations"):
            np.testing.assert_allclose(saved[key], actual[key], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(saved_ideal["physical_stationary_populations"], ideal["physical_stationary_populations"], atol=2e-12)
    observations, jobs = [], []
    for batch in manifest["batches"]:
        if batch["status"] != "collected" or not batch.get("job_id"):
            raise ValueError("uncollected prefix batch")
        if len(batch.get("observations", [])) != batch["stop"]-batch["start"]:
            raise ValueError("missing observations")
        observations.extend(batch["observations"])
        jobs.append(batch["job_id"])
    n = source["configuration"]["physics"]["n_qubits"]
    measured = {}
    for row, observation in zip(manifest["circuits"], observations):
        mean, covariance, shots = moments(observation["outcome_counts"], n)
        if shots != row["shots"]:
            raise ValueError("raw shot count mismatch")
        measured[row["prefix_steps"], row["trajectory"]] = (mean, covariance)
    results = []
    count = len(config["trajectory_indices"])
    for reference in ideal["prefixes"]:
        step = reference["prefix_steps"]
        values = np.array([measured[step, i][0] for i in config["trajectory_indices"]])
        ideal_paths = np.asarray(reference["per_trajectory_populations"])
        population = values.mean(axis=0)
        shot_cov = sum(measured[step, i][1] for i in config["trajectory_indices"])/count**2
        paired_cov = np.cov(values-ideal_paths, rowvar=False, ddof=1)/count if count > 1 else None
        across_cov = np.cov(values, rowvar=False, ddof=1)/count if count > 1 else None
        resources = [r["metrics"] for r in manifest["circuits"] if r["prefix_steps"] == step]
        results.append(dict(prefix_steps=step, physical_time=reference["physical_time"],
            hardware_populations=population, ideal_populations=ideal_paths.mean(axis=0),
            mean_channel_populations=reference["mean_channel_populations"],
            per_trajectory_hardware=values, per_trajectory_ideal=ideal_paths,
            shot_covariance_fixed_paths=shot_cov, shot_se_fixed_paths=np.sqrt(np.maximum(0, np.diag(shot_cov))),
            across_path_covariance=across_cov, paired_residual_covariance=paired_cov,
            comparison_to_ideal=population_error(population, ideal_paths.mean(axis=0)),
            distance_to_physical_stationary=population_error(population, np.asarray(ideal["physical_stationary_populations"])),
            ideal_distance_to_physical_stationary=population_error(ideal_paths.mean(axis=0), np.asarray(ideal["physical_stationary_populations"])),
            resources=dict(depth_range=[min(r["depth"] for r in resources), max(r["depth"] for r in resources)],
                           cz_counts=sorted({r["operation_counts"].get("cz", 0) for r in resources}),
                           reset_counts=sorted({r["operation_counts"].get("reset", 0) for r in resources}))))
    qpy_file = folder/"isa_circuits.qpy"
    if qpy_file.exists():
        if digest(qpy_file.read_bytes()) != config["isa_sha256"]:
            raise ValueError("ISA QPY hash mismatch")
        qpy_status = "verified original QPY SHA-256"
    elif (folder/"results_bundle.zip").exists():
        with zipfile.ZipFile(folder/"results_bundle.zip") as archive:
            if digest(archive.read("isa_circuits.qpy")) != config["isa_sha256"]:
                raise ValueError("archived ISA QPY hash mismatch")
        qpy_status = "verified original QPY SHA-256 inside bundle"
    else:
        qpy_status = "QPY unavailable; original compilation cannot be hash-verified"
    return dict(experiment=EXPERIMENT, run_directory=str(folder.resolve()), configuration=config,
                job_ids=jobs, prefixes=results, physical_stationary_populations=ideal["physical_stationary_populations"],
                physical_sites=config["physical_path"][1:-1], verification=dict(source_integrity=integrity,
                    isa_qpy=qpy_status, ideal_prefixes_recomputed=True, raw_counts_reconstructed=True,
                    input_sha256={name: digest(value) for name, value in raw.items()}),
                scope="Matched-prefix population diagnostic; transient prefixes are not steady-state accuracy tests. No mitigation.")


def save(report, output):
    output = Path(output)
    if output.resolve().is_relative_to(Path(report["run_directory"]).resolve()):
        raise ValueError("analysis output must be outside original run directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    rows = report["prefixes"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3*len(rows)), squeeze=False, layout="constrained")
    sites = np.arange(1, len(report["physical_sites"])+1)
    for ax, row in zip(axes[:, 0], rows):
        ax.plot(sites, report["physical_stationary_populations"], "k:", label="Physical stationary target")
        ax.plot(sites, row["ideal_populations"], "s-", color="teal", label="Ideal: same prefix and paths")
        ax.errorbar(sites, row["hardware_populations"], yerr=2*row["shot_se_fixed_paths"],
                    fmt="o-", color="firebrick", capsize=3, label="Hardware ±2 shot SE (fixed paths)")
        ax.set(xlabel="Site", ylabel="Population", xticks=sites,
               title=f"{row['prefix_steps']} steps; physical time {row['physical_time']:.4f}")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"populations_by_prefix.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    selected_sites = list(dict.fromkeys([0, (len(sites)-1)//2, len(sites)-1]))
    for site in selected_sites:
        color = plt.get_cmap("tab10")(site % 10)
        step = [r["prefix_steps"] for r in rows]
        axes[0].plot(step, [r["hardware_populations"][site] for r in rows], "o-", color=color, label=f"Site {site+1}")
        axes[0].plot(step, [r["ideal_populations"][site] for r in rows], "--", color=color, alpha=.7)
    axes[0].set(xlabel="Trotter steps", ylabel="Population", title="Selected sites; solid: hardware, dashed: ideal")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot([r["prefix_steps"] for r in rows], [r["comparison_to_ideal"]["maximum"] for r in rows], "o-", label="Maximum over sites")
    axes[1].plot([r["prefix_steps"] for r in rows], [r["comparison_to_ideal"]["mae"] for r in rows], "s-", label="Mean over sites")
    axes[1].set(xlabel="Trotter steps", ylabel="Absolute population difference", title="Hardware disagreement with each prefix's ideal")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"prefix_progression.{ext}", dpi=160)
    plt.close(fig)
    csv = [dict(prefix_steps=r["prefix_steps"], physical_time=r["physical_time"], site_1based=j+1,
                physical_qubit=report["physical_sites"][j], hardware=r["hardware_populations"][j],
                ideal_same_paths=r["ideal_populations"][j], shot_se_fixed_paths=r["shot_se_fixed_paths"][j],
                physical_stationary=report["physical_stationary_populations"][j]) for r in rows for j in range(len(sites))]
    write_csv(output/"populations.csv", csv)
    config = report["configuration"]
    text = ["# Model3 population prefixes", "", f"{len(config['trajectory_indices'])} saved trajectories × {config['shots']} shots × {len(rows)} prefixes: {config['total_circuits']} circuits and {config['total_shots']} shots.", "",
        f"Physical path (baths included): `{config['physical_path']}`. Jobs: {', '.join(report['job_ids'])}.", "",
        "![Actual populations at each prefix](populations_by_prefix.png)", "", "![Progression](prefix_progression.png)", "",
        "| Steps | Physical time | MAE vs matched ideal | Max vs matched ideal | Max vs stationary target | CZ counts | Resets |",
        "|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        text.append(f"| {r['prefix_steps']} | {r['physical_time']:.6f} | {r['comparison_to_ideal']['mae']:.6f} | {r['comparison_to_ideal']['maximum']:.6f} | {r['distance_to_physical_stationary']['maximum']:.6f} | {r['resources']['cz_counts']} | {r['resources']['reset_counts']} |")
    text += ["", "Compare hardware against the matching prefix ideal. Early prefixes are transient and need not match the stationary target. The selected subset is not independently certified to meet the original 0.02 sampling budget.", "",
        "The 0-step circuits repeat the initial-mixture preparation, including ignored preparation measurements, with no evolution. They screen the preparation at this execution occasion but do not replace all-zero/all-one or bath-reset controls.", "",
        "Error bars are ±2 shot SE conditional on the selected fixed paths. Across-path and paired hardware-minus-ideal covariance matrices are saved separately in analysis.json. Do not add these uncertainty estimates. They assume independent sampling and omit device drift; they are not simultaneous accuracy guarantees.", "",
        "All prefix/path pairs were shuffled with a saved seed. Submission order is not a calibrated execution timeline. Gate, idle and reset errors can accumulate together; this diagnostic alone does not isolate a noise mechanism.", "",
        f"Integrity: {report['verification']['isa_qpy']}. Source and ideal hashes checked; ideal prefix references independently recomputed; shot counts, pair coverage, mappings and count bit ordering verified. No original run files were changed by this analyzer.", ""]
    (output/"Analysis.md").write_text("\n".join(text))
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n")


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)
    report = analyze(options.run_directory)
    output = options.output_directory or Path(__file__).resolve().parent/"analysis"/"hardware"/options.run_directory.name
    save(report, output)
    print(f"Analysis: {output/'Analysis.md'}")
    return report


if __name__ == "__main__":
    main()
