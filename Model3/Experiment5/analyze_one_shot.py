"""Analyze saved one-shot hardware trajectories; no Runtime calls or evolution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model3.Experiment5.analyze_hardware import as_observables, metric, serializable, source_integrity


def reconstruct(records, observations, n, prefactor, J):
    """Pair three one-shot settings for each trajectory, including bond parity."""
    if len(records) != len(observations):
        raise ValueError("circuit/result lengths disagree")
    grouped = {}
    for record, observation in zip(records, observations):
        counts = observation["outcome_counts"]
        if (not counts or any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts.values()) != 1):
            raise ValueError("each circuit must contain exactly one shot")
        signs = None
        for key, count in counts.items():
            bits = key.replace(" ", "")
            if len(bits) != n or set(bits)-{"0", "1"}:
                raise ValueError("invalid final-register bitstring")
            if count:
                signs = 1-2*np.array([int(bit) for bit in bits[::-1]], float)
        path, basis = record["trajectory"], record["basis"]
        if basis not in ("z", "xy", "yx") or basis in grouped.get(path, {}):
            raise ValueError("unknown or duplicate trajectory/basis")
        grouped.setdefault(path, {})[basis] = (1-signs)/2 if basis == "z" else signs[:-1]*signs[1:]
    if not grouped:
        raise ValueError("no trajectories")
    parity = np.where(np.arange(n-1) % 2 == 0, 1., -1.)
    populations, currents = [], []
    for path in grouped.values():
        if set(path) != {"z", "xy", "yx"}:
            raise ValueError("each trajectory needs all three bases")
        populations.append(path["z"])
        currents.append(prefactor*J*parity*(path["yx"]-path["xy"]))
    return list(grouped), np.column_stack((populations, currents))


def statistics(samples):
    """Nominal covariance of the pooled mean across independent trajectories."""
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("at least two independent trajectory rows are required")
    centered = samples-samples.mean(axis=0)
    covariance = centered.T @ centered/(len(samples)*(len(samples)-1))
    return samples.mean(axis=0), covariance


def batch_diagnostics(values, ideal_values, indices, records, batches, n):
    """Use only paths with all three settings in one job for job clustering."""
    if len(batches) < 2:
        raise ValueError("job clustering requires at least two jobs")
    memberships = {}
    for g, batch in enumerate(batches):
        for record in records[batch["start"]:batch["stop"]]:
            memberships.setdefault(record["trajectory"], set()).add(g)
    row_for_path = {path: i for i, path in enumerate(indices)}
    groups, rows = [], []
    for g, batch in enumerate(batches):
        selected = [row_for_path[path] for path, jobs in memberships.items() if jobs == {g}]
        if len(selected) < 2:
            raise ValueError("each job needs at least two complete trajectories for diagnostics")
        x, target = values[selected], ideal_values[selected]
        hw_current, ideal_current = x[:, n:].mean(axis=1), target[:, n:].mean(axis=1)
        delta = hw_current-ideal_current
        groups.append(selected)
        rows.append({"job_order": g+1, "job_id": batch["job_id"], "complete_paths": len(selected),
            "mean_current": float(hw_current.mean()), "same_path_ideal_current": float(ideal_current.mean()),
            "current_discrepancy": float(delta.mean()),
            "within_job_mean_current_se": float(hw_current.std(ddof=1)/np.sqrt(len(x))),
            "within_job_discrepancy_se": float(delta.std(ddof=1)/np.sqrt(len(x))),
            "populations": x[:, :n].mean(axis=0), "currents": x[:, n:].mean(axis=0)})
    kept = np.concatenate(groups)

    def cluster_covariance(samples):
        center = samples[kept].mean(axis=0)
        sums = np.array([(samples[group]-center).sum(axis=0) for group in groups])
        return len(groups)/(len(groups)-1)*(sums.T @ sums)/len(kept)**2

    covariance = cluster_covariance(values)
    difference_covariance = cluster_covariance(values-ideal_values)
    _, subset_nominal_covariance = statistics(values[kept])
    averaging = np.r_[np.zeros(n), np.full(n-1, 1/(n-1))]
    observed_spread = np.std([row["current_discrepancy"] for row in rows], ddof=1)
    typical_se = np.sqrt(np.mean([row["within_job_discrepancy_se"]**2 for row in rows]))
    return {"jobs": rows, "complete_paths": len(kept), "excluded_split_paths": len(indices)-len(kept),
        "complete_path_hardware_mean": values[kept].mean(axis=0),
        "complete_path_paired_difference": (values-ideal_values)[kept].mean(axis=0),
        "cluster_covariance_of_hardware_mean": covariance,
        "cluster_covariance_of_paired_difference": difference_covariance,
        "mean_current_cluster_se": float(np.sqrt(max(0., averaging @ covariance @ averaging))),
        "mean_current_subset_nominal_se": float(np.sqrt(max(0., averaging @ subset_nominal_covariance @ averaging))),
        "paired_mean_current_cluster_se": float(np.sqrt(max(0., averaging @ difference_covariance @ averaging))),
        "job_discrepancy_sd": float(observed_spread), "typical_within_job_discrepancy_se": float(typical_se),
        "job_spread_to_within_job_se_ratio": float(observed_spread/typical_se) if typical_se > 0 else None,
        "interpretation": "Diagnostic on complete within-job triples only. Cluster SE assumes independent jobs; not a calibration-drift test. Split triples are retained in the primary pooled estimate."}


def analyze(folder):
    raw = {name: (folder/name).read_bytes() for name in ("manifest.json", "hardware_results.json", "source.json")}
    manifest, saved, source = (json.loads(raw[name]) for name in raw)
    if manifest.get("experiment") != "Model3/Experiment5/hardware" or manifest.get("status") != "collected":
        raise ValueError("a fully collected Model3 hardware run is required")
    config, p = manifest["configuration"], source["configuration"]["physics"]
    if config["shots"] != 1 or p.get("delta", 0) != 0 or min(p["J"], p["gamma"]) <= 0:
        raise ValueError("analysis requires one shot, an XX chain, and positive J/gamma")
    integrity = source_integrity(raw["source.json"], config["source_sha256"])
    records, batches = manifest["circuits"], manifest["batches"]
    observations, stop = [], 0
    for batch in batches:
        if (batch["status"] != "collected" or batch["start"] != stop
                or len(batch["observations"]) != batch["stop"]-batch["start"] or not batch["job_id"]):
            raise ValueError("missing, unordered, or incomplete job")
        stop = batch["stop"]
        observations.extend(batch["observations"])
    if stop != len(records) or [r["pub_index"] for r in records] != list(range(stop)):
        raise ValueError("publication indices or job coverage disagree")
    if len({b["job_id"] for b in batches}) != len(batches):
        raise ValueError("duplicate job IDs")
    n, J, prefactor = p["n_qubits"], p["J"], p["exchange_prefactor"]
    indices, values = reconstruct(records, observations, n, prefactor, J)
    if indices != config["trajectory_indices"] or len(indices) != len(set(indices)):
        raise ValueError("trajectory selection mismatch")
    pooled, covariance = statistics(values)
    hardware = as_observables(pooled[:n], pooled[n:])
    reference_cov = np.asarray(source["final_covariances_real_gauge"], float)
    if reference_cov.shape != (1, source["configuration"]["trajectories"], n, n):
        raise ValueError("a single-preparation covariance reference is required")
    reference_cov = reference_cov[0, indices]
    if not np.allclose(reference_cov, reference_cov.transpose(0, 2, 1), atol=1e-12):
        raise ValueError("asymmetric reference covariance")
    eigenvalues = np.linalg.eigvalsh(reference_cov)
    if eigenvalues.min() < -1e-7 or eigenvalues.max() > 1+1e-7:
        raise ValueError("unphysical ideal reference covariance")
    ideal_values = np.column_stack((np.diagonal(reference_cov, axis1=-2, axis2=-1),
        4*prefactor*J*np.diagonal(reference_cov, offset=1, axis1=-2, axis2=-1)))
    ideal_mean, ideal_covariance = statistics(ideal_values)
    paired_difference, paired_covariance = statistics(values-ideal_values)
    same_paths = as_observables(ideal_mean[:n], ideal_mean[n:])
    mean = as_observables(np.asarray(source["mean_channel_observables"]["populations"])[0, -1],
        np.asarray(source["mean_channel_observables"]["excitation_flows"])[0, -1])
    a, gamma = 2*prefactor*J, p["gamma"]
    current = 2*a*a*gamma/(gamma*gamma+4*a*a)*(p["p_left"]-p["p_right"])
    populations = np.full(n, (p["p_left"]+p["p_right"])/2)
    populations[0], populations[-1] = p["p_left"]-current/gamma, p["p_right"]+current/gamma
    physical = as_observables(populations, np.full(n-1, current))
    references = {"physical_stationary": physical, "ideal_mean_channel": mean, "ideal_same_saved_paths": same_paths}
    se = np.sqrt(np.maximum(0., np.diag(covariance)))
    errors = as_observables(se[:n], se[n:]); errors["magnetizations"] = 2*se[:n]
    for family, start, end in (("populations", 0, n), ("excitation_flows", n, 2*n-1)):
        np.testing.assert_allclose(hardware[family], saved["observables"][family], atol=1e-13)
        np.testing.assert_allclose(values[:, start:end], saved["per_trajectory"][family], atol=1e-13)
        np.testing.assert_allclose(errors[family], saved["across_path_standard_errors"][family], atol=1e-13)
        for name, reference in references.items():
            np.testing.assert_allclose(reference[family], saved["references"][name][family], atol=1e-12)
    if saved["job_ids"] != [b["job_id"] for b in batches]:
        raise ValueError("saved job IDs mismatch")
    for name, reference in references.items():
        np.testing.assert_allclose(metric(hardware, reference, J)["maximum"], saved["local_observable_errors"][name], atol=1e-12)
    comparisons = {"hardware_vs_physical": metric(hardware, physical, J),
        "hardware_vs_same_ideal_paths": metric(hardware, same_paths, J),
        "same_ideal_paths_vs_physical": metric(same_paths, physical, J),
        "finite_paths_vs_ideal_mean": metric(same_paths, mean, J),
        "ideal_mean_vs_physical": metric(mean, physical, J)}
    averaging = np.r_[np.zeros(n), np.full(n-1, 1/(n-1))]
    currents = {name: float(obs["excitation_flows"].mean()) for name, obs in {"hardware": hardware, **references}.items()}
    diagnostics = batch_diagnostics(values, ideal_values, indices, records, batches, n)
    balance = []
    for label, gradient, constant in (("left", np.r_[-gamma, np.zeros(n-1), -1., np.zeros(n-2)], gamma*p["p_left"]),
            ("right", np.r_[np.zeros(n-1), gamma, np.zeros(n-2), -1.], -gamma*p["p_right"])):
        balance.append({"boundary": label, "residual": float(constant+gradient @ pooled),
            "nominal_se": float(np.sqrt(max(0., gradient @ covariance @ gradient)))})
    return {"run_directory": str(folder.resolve()), "backend": config["backend"], "physics": p,
        "paths": len(indices), "circuits": len(records), "shots_per_circuit": 1, "total_shots": len(observations),
        "jobs": len(batches), "job_ids": [b["job_id"] for b in batches],
        "preparation": source["preparation"]["name"], "preparation_steps": len(source["schedule"]["step_dts"]),
        "physical_output_time": source["times"][-1], "tolerance": source["configuration"]["steady_tolerance"],
        "hardware": hardware, "references": references, "comparisons": comparisons,
        "nominal_across_path_se_including_shots": errors,
        "covariance_of_pooled_mean": covariance, "ideal_path_covariance_of_mean": ideal_covariance,
        "paired_difference_covariance_of_mean": paired_covariance, "paired_difference": paired_difference,
        "mean_bond_currents": currents, "mean_current_se": float(np.sqrt(averaging @ covariance @ averaging)),
        "paired_mean_current_difference_se": float(np.sqrt(averaging @ paired_covariance @ averaging)),
        "current_retention_same_ideal_paths": currents["hardware"]/currents["ideal_same_saved_paths"],
        "steady_boundary_balance_diagnostics": balance, "batch_diagnostics": diagnostics,
        "resource_metrics": {"depth_range": [min(r["metrics"]["depth"] for r in records), max(r["metrics"]["depth"] for r in records)],
            "cz_gates_per_circuit": sorted({r["metrics"]["operation_counts"].get("cz", 0) for r in records}),
            "resets_per_circuit": sorted({r["metrics"]["operation_counts"].get("reset", 0) for r in records}),
            "physical_path": config["physical_path"], "versions": config["versions"]},
        "verification": {"source_integrity": integrity, "saved_values_reconstructed": True,
            "source_reference_covariance_physical": True, "isa_qpy_available": (folder/"isa_circuits.qpy").exists(),
            "original_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}},
        "interpretation": "One-shot uncertainty is estimated across trajectories and includes preparation and measurement randomness. It cannot be split empirically into fixed-path shot variance and trajectory variance. Nominal SE assumes independent samples; job-cluster sensitivity is reported separately. Final output only, with no hardware stationarity or noisy accuracy guarantee."}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11))
    fig.subplots_adjust(left=.10, right=.98, bottom=.06, top=.84, hspace=.36)
    labels = {"populations": r"Population $\langle n_j\rangle$",
        "magnetizations": r"Magnetization $\langle Z_j\rangle$", "excitation_flows": r"Excitation current $I_j/J$"}
    for axis, (family, label) in zip(axes, labels.items()):
        divisor = report["physics"]["J"] if family == "excitation_flows" else 1.
        x = np.arange(1, len(report["hardware"][family])+1)
        target = report["references"]["physical_stationary"][family]/divisor
        allowance = report["tolerance"]*(2 if family == "magnetizations" else 1)
        axis.fill_between(x, target-allowance, target+allowance, color="gray", alpha=.13, label="Accuracy band")
        axis.plot(x, target, "--", color="black", lw=1.7, label="Physical steady state")
        axis.plot(x, report["references"]["ideal_mean_channel"][family]/divisor,
                  ":", color="#00897b", lw=2, label="Ideal mean channel")
        axis.plot(x, report["references"]["ideal_same_saved_paths"][family]/divisor,
                  "s-", color="#1565c0", ms=4, lw=1.3, label=f"Ideal same {report['paths']} paths")
        axis.errorbar(x, report["hardware"][family]/divisor,
            yerr=2*report["nominal_across_path_se_including_shots"][family]/divisor,
            fmt="o-", color="#c62828", capsize=3, lw=1.5, ms=5, label="Phoenix ±2 pooled SE (paths + shots)")
        axis.set(ylabel=label, xlabel="Bond (j, j+1)" if family == "excitation_flows" else "Site j", xticks=x)
        axis.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, fontsize=9)
    fig.suptitle(f"IBM Phoenix: final observables | {report['preparation_steps']} cosine steps\n"
        f"{report['paths']} trajectories × 1 shot/basis | {report['jobs']} jobs | output t={report['physical_output_time']:.5f}", y=.99, fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"observables.{suffix}", dpi=180)
    plt.close(fig)
    jobs = report["batch_diagnostics"]["jobs"]
    x = np.arange(1, len(jobs)+1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained", height_ratios=[1, 1.15])
    axes[0].errorbar(x, [r["mean_current"] for r in jobs],
        yerr=[2*r["within_job_mean_current_se"] for r in jobs], fmt="o-", ms=3, lw=1,
        color="#c62828", label="Hardware per-job mean ±2 within-job SE")
    axes[0].plot(x, [r["same_path_ideal_current"] for r in jobs], "s-", ms=3, lw=1,
                  color="#1565c0", label="Ideal same complete paths")
    axes[0].axhline(report["mean_bond_currents"]["physical_stationary"], color="black", ls="--", label="Physical steady current")
    axes[0].set(ylabel="Mean bond current", xlabel="Submission/job order", title=f"All {len(jobs)} jobs: complete trajectory triples within each job")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=.2)
    heat = axes[1].imshow(np.array([r["currents"] for r in jobs]).T, origin="lower", aspect="auto",
                         extent=[.5, len(jobs)+.5, .5, report["physics"]["n_qubits"]-.5], cmap="viridis")
    axes[1].set(xlabel="Submission/job order", ylabel="Left site of bond (1-based)", title="Measured current on each bond, averaged within each job")
    fig.colorbar(heat, ax=axes[1], label="Excitation current")
    fig.savefig(output/"batch_currents.png", dpi=180)
    plt.close(fig)


def save(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n")
    for family in ("populations", "magnetizations", "excitation_flows"):
        with (output/f"{family}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["site_or_left_bond_site_1based", "hardware", "nominal_se_paths_and_shots",
                             "physical_stationary", "ideal_mean_channel", "ideal_same_saved_paths"])
            for i, value in enumerate(report["hardware"][family]):
                writer.writerow([i+1, value, report["nominal_across_path_se_including_shots"][family][i],
                    *[report["references"][name][family][i] for name in
                      ("physical_stationary", "ideal_mean_channel", "ideal_same_saved_paths")]])
    jobs = report["batch_diagnostics"]["jobs"]
    with (output/"per_job.csv").open("w", newline="") as handle:
        keys = [k for k in jobs[0] if k not in ("populations", "currents")]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows({k: row[k] for k in keys} for row in jobs)
    c, means, d = report["comparisons"], report["mean_bond_currents"], report["batch_diagnostics"]
    spread_ratio = f"{d['job_spread_to_within_job_se_ratio']:.3f}" if d["job_spread_to_within_job_se_ratio"] is not None else "undefined: zero within-job SE"
    physical_error = c["hardware_vs_physical"]["maximum"]
    lines = ["# IBM Phoenix: 8192 trajectories with one shot per setting", "",
        f"Run: `{Path(report['run_directory']).name}`. All {report['jobs']} jobs and {report['circuits']:,} circuits collected; {report['paths']:,} complete trajectories, 3 settings/path, one shot/setting, {report['total_shots']:,} total shots.",
        f"Model3 is the unequal-reservoir XX model of writeup Section 4.9. This run uses the ten-site extension: J=gamma=1, reservoirs (0.9,0.1), half-mixed preparation, {report['preparation_steps']} cosine steps, output time {report['physical_output_time']:.9g}. It does not use the PDF's three-site, gamma=2 target.", "",
        f"**The hardware output {'passes' if physical_error <= report['tolerance'] else 'fails'} the {report['tolerance']:g} tolerance:** maximum local error {physical_error:.9g}.", "",
        "## Raw-count verification and ideal references", "",
        "Every outcome histogram was checked for width, binary labels, and exactly one shot. The three measurement settings were paired by trajectory, current signs were reconstructed for both bond parities, all publication ranges and job IDs were checked, and saved observables/uncertainties/references were independently reproduced. Final-register counts alone were used.", "",
        "| Comparison | Max population error | Max current error / J | Combined metric |",
        "|---|---:|---:|---:|"]
    for label, key in (("Ideal mean vs physical", "ideal_mean_vs_physical"),
            ("Ideal same 8192 paths vs mean", "finite_paths_vs_ideal_mean"),
            ("Ideal same paths vs physical", "same_ideal_paths_vs_physical"),
            ("Hardware vs same ideal paths", "hardware_vs_same_ideal_paths"),
            ("Hardware vs physical", "hardware_vs_physical")):
        row = c[key]
        lines.append(f"| {label} | {row['population_max']:.9g} | {row['current_divided_by_J_max']:.9g} | {row['maximum']:.9g} |")
    lines += ["", "The ideal same-path error fits 0.02, and its distance from the ideal mean is below 0.004. The hardware discrepancy persists after matching all 8192 ideal branch sequences. Increasing trajectory diversity resolved the old four-path reference limitation; it did not remove the hardware mismatch. Maxima in different rows cannot be added as scalar error contributions.", "",
        "## Actual observable values", "",
        "[Populations, magnetizations and currents](observables.png) ([PDF](observables.pdf)). All sites/bonds are included, using one-based indices. CSV files contain every hardware/reference value and its nominal standard error.", "",
        "| Bond | Hardware current / J | Twice nominal SE / J | Ideal same paths / J | Physical target / J |",
        "|---|---:|---:|---:|---:|"]
    J = report["physics"]["J"]
    for i, value in enumerate(report["hardware"]["excitation_flows"]):
        lines.append(f"| {i+1}–{i+2} | {value/J:.7f} | {2*report['nominal_across_path_se_including_shots']['excitation_flows'][i]/J:.7f} | {report['references']['ideal_same_saved_paths']['excitation_flows'][i]/J:.7f} | {means['physical_stationary']/J:.7f} |")
    lines += ["", "| Spatial mean current | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.9g} |" for key, value in means.items())
    lines += ["", f"Hardware retains {100*report['current_retention_same_ideal_paths']:.2f}% of the matched ideal mean current. The nominal SE of its spatial mean is {report['mean_current_se']:.7g}, retaining within-trajectory correlations between bonds.",
        f"The worst current discrepancy is on bond {c['hardware_vs_physical']['worst_bond_1based']}; the worst population discrepancy is at site {c['hardware_vs_physical']['worst_site_1based']}. All nine measured currents are below their physical target.", "",
        "## What uncertainty is estimable with one shot?", "",
        "A single trajectory has no empirically estimable within-circuit shot variance. Instead, each trajectory contributes a vector of population/current samples, and the covariance of the overall mean is the sample covariance of those vectors divided by R. This includes branch variation, mixture-preparation randomness and measurement fluctuations; these components are not separately identified from one shot. Do not add a separate shot-error estimate to the pooled SE.",
        f"The maximum nominal two-SE values are {2*max(report['nominal_across_path_se_including_shots']['populations']):.7g} for populations and {2*max(report['nominal_across_path_se_including_shots']['excitation_flows'])/J:.7g} for current/J. These exceed the original 0.004 sampling allowance, but remain far smaller than the observed current mismatch. The 0.004 ideal-trajectory validation excluded hardware measurement noise.",
        "Nominal SE assumes independent trajectories. Twice-SE bars are descriptive per-observable uncertainty estimates, not simultaneous confidence bands, and do not include systematic hardware bias.", "",
        "## Sensitivity to jobs and submission order", "",
        f"For job diagnostics, {d['complete_paths']} trajectories have all three settings inside a single job, {jobs[0]['complete_paths']} per job. The {d['excluded_split_paths']} paths crossing a job boundary are excluded only from this diagnostic; all are retained in the primary estimate.",
        f"The per-job spatial mean current ranges from {min(r['mean_current'] for r in jobs):.7g} to {max(r['mean_current'] for r in jobs):.7g}, below the target {means['physical_stationary']:.7g} in every job. [Job means and current heatmap](batch_currents.png); [per-job CSV](per_job.csv).",
        f"Clustering the complete-path subset by job gives mean-current SE {d['mean_current_cluster_se']:.7g}, compared with nominal SE {d['mean_current_subset_nominal_se']:.7g} on that same subset and {report['mean_current_se']:.7g} on all paths. The job-level SE of the paired hardware-minus-ideal current is {d['paired_mean_current_cluster_se']:.7g}. The observed SD of per-job paired discrepancies is {d['job_discrepancy_sd']:.7g}, versus typical within-job SE {d['typical_within_job_discrepancy_se']:.7g} (ratio {spread_ratio}).",
        "These diagnostics make the uncertainty somewhat larger but do not account for the large suppression. They assume independent jobs and use a slightly smaller subset; they do not establish time drift or identify a device mechanism. Submission order is not a calibrated time axis, and job timing/calibration data were not fetched.", "",
        "## Steady-state balance diagnostics", "",
        "The measured output fails the target steady boundary balances as well. Left residual = gamma*(p_left-n_1)-I_1; right residual = gamma*(n_N-p_right)-I_(N-1). These diagnose agreement with the intended stationary equations; they are not measurements of a hardware time derivative:", "",
        "| Boundary | Steady-balance residual | Nominal SE |", "|---|---:|---:|"]
    lines.extend(f"| {row['boundary']} | {row['residual']:.8g} | {row['nominal_se']:.8g} |" for row in report["steady_boundary_balance_diagnostics"])
    if "previous_run" in report:
        old = report["previous_run"]
        lines += ["", "## Comparison with the earlier four-trajectory run", "",
            "| Saved run | Trajectories | Shots/basis/path | Jobs | Total shots | Mean hardware current | Matched ideal mean current | Hardware physical error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Earlier | {old['paths']} | {old['shots_per_circuit']} | {old['jobs']} | {old['total_shots']} | {old['mean_current']:.8g} | {old['ideal_mean_current']:.8g} | {old['physical_error']:.8g} |",
            f"| One-shot | {report['paths']} | 1 | {report['jobs']} | {report['total_shots']} | {means['hardware']:.8g} | {means['ideal_same_saved_paths']:.8g} | {physical_error:.8g} |",
            "Both show strong current suppression. The different path sets, shot allocation, and execution occasions prevent treating this as a controlled test of whether changing shots itself caused a change in hardware fidelity. The observed mean current is slightly larger in the new run, while the maximum physical error is also larger."]
    lines += ["", "## Scope and resource audit", "",
        f"Native depth range {report['resource_metrics']['depth_range']}; CZ counts/circuit {report['resource_metrics']['cz_gates_per_circuit']}; resets/circuit {report['resource_metrics']['resets_per_circuit']}; physical chain `{report['resource_metrics']['physical_path']}`.",
        f"Source integrity: `{report['verification']['source_integrity']}`. The only accepted source-byte conversion is an LF/CRLF transformation that exactly reproduces the original saved hash. ISA QPY available here: {report['verification']['isa_qpy_available']}; its hash could not be checked when absent. Original file hashes and all job IDs are in analysis.json.",
        "This is one final-time measurement, so hardware stationarity is not established. Populations sloping through the chain do not demonstrate diffusion in the physical homogeneous XX model, which has a flat interior target. Gate/reset/preparation/readout effects and calibration changes cannot be separated with these data. No hardware jobs, new evolution, mitigation, or backend queries were performed; original inputs were preserved."]
    (output/"Analysis.md").write_text("\n".join(lines)+"\n")
    plot(report, output)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--compare-run", type=Path)
    options = parser.parse_args(arguments)
    output = options.output_directory or DIRECTORY/"analysis/hardware"/options.run_directory.name
    if output.resolve() == options.run_directory.resolve():
        raise ValueError("output must be separate from input run")
    report = analyze(options.run_directory)
    if options.compare_run:
        from Model3.Experiment5.analyze_hardware import analyze as analyze_previous
        previous, _, _ = analyze_previous(options.compare_run)
        if previous["physics"] != report["physics"] or previous["preparation_steps"] != report["preparation_steps"]:
            raise ValueError("previous run has different model/step count")
        report["previous_run"] = {"directory": str(options.compare_run.resolve()), "paths": previous["paths"],
            "shots_per_circuit": previous["shots_per_circuit"], "total_shots": previous["total_shots"],
            "jobs": len(previous["job_ids"]), "mean_current": previous["mean_bond_currents"]["hardware"],
            "ideal_mean_current": previous["mean_bond_currents"]["ideal_same_saved_paths"],
            "physical_error": previous["comparisons"]["hardware_vs_physical"]["maximum"],
            "original_sha256": previous["verification"]["original_files_sha256"]}
    save(report, output)
    for name, digest in report["verification"]["original_sha256"].items():
        if hashlib.sha256((options.run_directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError("an original run file changed")
    if options.compare_run:
        for name, digest in report["previous_run"]["original_sha256"].items():
            if hashlib.sha256((options.compare_run/name).read_bytes()).hexdigest() != digest:
                raise ValueError("a previous-run input changed")
    print(f"Saved: {output/'Analysis.md'}")
    print(json.dumps(serializable({key: report[key] for key in ("comparisons", "mean_bond_currents", "mean_current_se")} ), indent=2))
    return report


if __name__ == "__main__":
    main()
