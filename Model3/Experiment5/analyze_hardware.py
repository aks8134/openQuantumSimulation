"""Analyze saved Model3 hardware counts and plot final spatial observables.

Read-only with respect to the run: no Runtime imports, job submission,
trajectory evolution, mitigation, or changes to original result files.
"""
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


def source_integrity(raw, expected):
    """Accept exact bytes or a proven LF/CRLF-only Git checkout conversion."""
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if digest(raw) == expected:
        return "exact_bytes"
    lf = raw.replace(b"\r\n", b"\n")
    for name, candidate in (("LF", lf), ("CRLF", lf.replace(b"\n", b"\r\n"))):
        if digest(candidate) == expected:
            return f"matches_original_{name}_line_endings"
    raise ValueError("source hash mismatch is not explained by LF/CRLF conversion")


def moments(counts, n, *, pairs=False):
    """Feature means and unbiased covariance of their shot means.

    C_mean = (E[ff^T]-E[f]E[f]^T)/(S-1). This retains correlations
    between sites/bonds within each measurement setting.
    """
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("counts must be nonnegative integers")
    shots = sum(counts.values())
    if shots < 2:
        raise ValueError("at least two shots are required for uncertainty")
    values, weights = [], []
    for key, weight in counts.items():
        bits = key.replace(" ", "")
        if len(bits) != n or set(bits)-{"0", "1"}:
            raise ValueError("invalid final-register bitstring")
        signs = 1-2*np.array([int(bit) for bit in bits[::-1]])
        values.append(signs[:-1]*signs[1:] if pairs else (1-signs)/2)
        weights.append(weight)
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    mean = weights @ values/shots
    covariance = (values.T @ (weights[:, None]*values)/shots-np.outer(mean, mean))/(shots-1)
    return mean, covariance, shots


def reconstruct(records, observations, n, prefactor, J):
    if len(records) != len(observations):
        raise ValueError("circuit and result counts disagree")
    grouped = {}
    for record, row in zip(records, observations):
        path, basis = record["trajectory"], record["basis"]
        if basis not in ("z", "xy", "yx") or basis in grouped.get(path, {}):
            raise ValueError("unknown or duplicate trajectory/basis record")
        grouped.setdefault(path, {})[basis] = moments(row["outcome_counts"], n, pairs=basis != "z")
    populations, currents, pop_cov, current_cov, shots = [], [], [], [], []
    parity = np.where(np.arange(n-1) % 2 == 0, 1., -1.)
    for path in grouped.values():
        if set(path) != {"z", "xy", "yx"}:
            raise ValueError("each trajectory needs all three bases")
        populations.append(path["z"][0]); pop_cov.append(path["z"][1])
        currents.append(prefactor*J*parity*(path["yx"][0]-path["xy"][0]))
        current_cov.append((prefactor*J)**2*parity[:, None]*parity[None, :]*(path["yx"][1]+path["xy"][1]))
        shots.extend(path[basis][2] for basis in ("z", "xy", "yx"))
    paths = len(grouped)
    if not paths:
        raise ValueError("no trajectories found")
    return {"indices": list(grouped), "populations": np.asarray(populations), "excitation_flows": np.asarray(currents),
        "population_shot_covariance": np.sum(pop_cov, axis=0)/paths**2,
        "current_shot_covariance": np.sum(current_cov, axis=0)/paths**2,
        "per_path_population_shot_covariance": np.asarray(pop_cov),
        "per_path_current_shot_covariance": np.asarray(current_cov), "shots_per_circuit": shots}


def metric(left, right, J):
    population = np.asarray(left["populations"])-right["populations"]
    current = (np.asarray(left["excitation_flows"])-right["excitation_flows"])/J
    return {"population_max": float(np.abs(population).max()), "current_divided_by_J_max": float(np.abs(current).max()),
        "maximum": float(max(np.abs(population).max(), np.abs(current).max())),
        "population_rmse": float(np.sqrt(np.mean(population**2))), "current_divided_by_J_rmse": float(np.sqrt(np.mean(current**2))),
        "worst_site_1based": int(np.argmax(np.abs(population))+1),
        "worst_bond_1based": [int(np.argmax(np.abs(current))+1), int(np.argmax(np.abs(current))+2)]}


def as_observables(populations, currents):
    return {"populations": np.asarray(populations), "magnetizations": 1-2*np.asarray(populations),
            "excitation_flows": np.asarray(currents)}


def analyze(folder):
    paths = {name: folder/name for name in ("manifest.json", "hardware_results.json", "source.json")}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    manifest, saved, source = (json.loads(raw[name]) for name in paths)
    if manifest.get("experiment") != "Model3/Experiment5/hardware" or manifest["status"] != "collected":
        raise ValueError("a completely collected Model3 hardware run is required")
    config, p = manifest["configuration"], source["configuration"]["physics"]
    integrity = source_integrity(raw["source.json"], config["source_sha256"])
    n, J, gamma, prefactor = p["n_qubits"], p["J"], p["gamma"], p["exchange_prefactor"]
    if p.get("delta", 0) != 0 or J <= 0 or gamma <= 0:
        raise ValueError("analysis requires the XX model with positive J and gamma")
    observations, job_ids, expected_start = [], [], 0
    for batch in manifest["batches"]:
        if batch["status"] != "collected" or batch["start"] != expected_start:
            raise ValueError("missing or unordered batch results")
        if len(batch["observations"]) != batch["stop"]-batch["start"]:
            raise ValueError("batch result length disagrees with circuit range")
        observations.extend(batch["observations"]); job_ids.append(batch["job_id"])
        expected_start = batch["stop"]
    records = manifest["circuits"]
    if [row["pub_index"] for row in records] != list(range(len(records))):
        raise ValueError("circuit publication indices are not contiguous")
    measured = reconstruct(records, observations, n, prefactor, J)
    if measured["indices"] != config["trajectory_indices"] or set(measured["shots_per_circuit"]) != {config["shots"]}:
        raise ValueError("measured paths or shots disagree with requested configuration")
    hardware = as_observables(measured["populations"].mean(axis=0), measured["excitation_flows"].mean(axis=0))
    # Independently reconstruct the physical stationary reference analytically.
    a = 2*prefactor*J
    stationary_current = 2*a*a*gamma/(gamma*gamma+4*a*a)*(p["p_left"]-p["p_right"])
    stationary_pop = np.full(n, (p["p_left"]+p["p_right"])/2)
    stationary_pop[0], stationary_pop[-1] = p["p_left"]-stationary_current/gamma, p["p_right"]+stationary_current/gamma
    physical = as_observables(stationary_pop, np.full(n-1, stationary_current))
    mean = as_observables(np.asarray(source["mean_channel_observables"]["populations"])[0, -1],
                          np.asarray(source["mean_channel_observables"]["excitation_flows"])[0, -1])
    covariances = np.asarray(source["final_covariances_real_gauge"]).reshape(-1, n, n)[measured["indices"]]
    if not np.allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-12):
        raise ValueError("ideal reference covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariances)
    if eigenvalues.min() < -1e-7 or eigenvalues.max() > 1+1e-7:
        raise ValueError("ideal reference covariance is not physical")
    ideal_paths = as_observables(np.diagonal(covariances, axis1=-2, axis2=-1),
                                4*prefactor*J*np.diagonal(covariances, offset=1, axis1=-2, axis2=-1))
    ideal = {family: values.mean(axis=0) for family, values in ideal_paths.items()}
    references = {"physical_stationary": physical, "ideal_mean_channel": mean, "ideal_same_saved_paths": ideal}
    # Check original reporting against independently reconstructed raw counts.
    for family in ("populations", "excitation_flows"):
        np.testing.assert_allclose(hardware[family], saved["observables"][family], atol=1e-13)
        np.testing.assert_allclose(measured[family], saved["per_trajectory"][family], atol=1e-13)
        for name, reference in references.items():
            np.testing.assert_allclose(reference[family], saved["references"][name][family], atol=1e-12)
    if saved["job_ids"] != job_ids:
        raise ValueError("saved job IDs disagree with manifest")
    shot_se = as_observables(np.sqrt(np.maximum(0, np.diag(measured["population_shot_covariance"]))),
                              np.sqrt(np.maximum(0, np.diag(measured["current_shot_covariance"]))))
    # Magnetization uncertainty transforms as sigma_Z = 2 sigma_n.
    shot_se["magnetizations"] = 2*shot_se["populations"]
    R = len(measured["indices"])
    across = {family: measured[family].std(axis=0, ddof=1)/np.sqrt(R) if R > 1 else None
              for family in ("populations", "excitation_flows")}
    comparisons = {"hardware_vs_physical": metric(hardware, physical, J),
        "hardware_vs_same_ideal_paths": metric(hardware, ideal, J),
        "finite_paths_vs_ideal_mean": metric(ideal, mean, J),
        "same_ideal_paths_vs_physical": metric(ideal, physical, J),
        "ideal_mean_vs_physical": metric(mean, physical, J)}
    for name, key in (("physical_stationary", "hardware_vs_physical"), ("ideal_same_saved_paths", "hardware_vs_same_ideal_paths")):
        np.testing.assert_allclose(saved["local_observable_errors"][name], comparisons[key]["maximum"], atol=1e-13)
    tolerance = source["configuration"]["steady_tolerance"]
    currents = {name: float(values["excitation_flows"].mean()) for name, values in {"hardware": hardware, **references}.items()}
    current_mean_se = float(np.sqrt(max(0., measured["current_shot_covariance"].sum()))/(n-1))
    ratio = currents["hardware"]/currents["ideal_same_saved_paths"] if currents["ideal_same_saved_paths"] else None
    report = {"run_directory": str(folder.resolve()), "backend": config["backend"], "physics": p,
        "job_ids": job_ids, "paths": R, "circuits": len(records), "shots_per_circuit": config["shots"],
        "total_shots": sum(measured["shots_per_circuit"]), "preparation_steps": config["preparation_steps"],
        "preparation": source["preparation"]["name"],
        "physical_output_time": source["times"][-1], "tolerance": tolerance,
        "hardware_output_within_tolerance": comparisons["hardware_vs_physical"]["maximum"] <= tolerance,
        "hardware": hardware, "references": references, "shot_standard_errors_fixed_paths": shot_se,
        "across_path_standard_errors_including_shots": across, "comparisons": comparisons,
        "mean_bond_currents": currents, "mean_hardware_bond_current_shot_se": current_mean_se,
        "mean_current_fraction_of_same_ideal_paths": ratio,
        "resource_metrics": {"native_depth_range": [min(row["metrics"]["depth"] for row in records), max(row["metrics"]["depth"] for row in records)],
            "cz_gates_per_circuit": sorted({row["metrics"]["operation_counts"].get("cz", 0) for row in records}),
            "resets_per_circuit": sorted({row["metrics"]["operation_counts"].get("reset", 0) for row in records}),
            "physical_path": config["physical_path"]},
        "verification": {"source_integrity": integrity, "saved_observables_reproduced_from_counts": True,
            "physical_reference_reconstructed": True, "original_files_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            "isa_file_available": (folder/"isa_circuits.qpy").exists()},
        "interpretation": "Final output only; no hardware time series or stationarity proof. Shot SE assumes independent shots within fixed paths; across-path SE includes shots and must not be added to shot SE in quadrature. Max-norm comparison terms do not add as scalars. Hardware discrepancy includes gate, reset, preparation and readout effects, which this run cannot isolate."}
    return report, measured, ideal_paths


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def plot(report, measured, ideal_paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hw, refs, se = report["hardware"], report["references"], report["shot_standard_errors_fixed_paths"]
    names = {"populations": r"Population $\langle n_j\rangle$", "magnetizations": r"Magnetization $\langle Z_j\rangle$",
             "excitation_flows": r"Excitation current $I_j/J$"}
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11))
    fig.subplots_adjust(left=.095, right=.98, bottom=.06, top=.86, hspace=.36)
    for family, axis in zip(names, axes):
        divisor = report["physics"]["J"] if family == "excitation_flows" else 1.
        x = np.arange(1, len(hw[family])+1)
        target = refs["physical_stationary"][family]/divisor
        allowance = report["tolerance"]*(2 if family == "magnetizations" else 1)
        axis.fill_between(x, target-allowance, target+allowance, color="gray", alpha=.13, label="Accuracy band")
        axis.plot(x, target, color="black", linestyle="--", linewidth=1.8, label="Physical steady state")
        axis.plot(x, refs["ideal_mean_channel"][family]/divisor, color="#00897b", linestyle=":", linewidth=2.1, label="Ideal mean channel")
        axis.plot(x, refs["ideal_same_saved_paths"][family]/divisor, "s-", color="#1565c0", markersize=4, linewidth=1.4, label=f"Ideal same {report['paths']} paths")
        axis.errorbar(x, hw[family]/divisor, yerr=2*se[family]/divisor, fmt="o-", color="#c62828", capsize=3,
                      markersize=5, linewidth=1.6, label="Phoenix, ±2 shot SE")
        axis.set(ylabel=names[family], xlabel="Bond (j, j+1)" if family == "excitation_flows" else "Site j", xticks=x)
        axis.grid(alpha=.2)
        if family == "excitation_flows":
            axis.set_ylim(bottom=0.)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, fontsize=9)
    fig.suptitle(f"IBM Phoenix: final spatial observables | {report['preparation_steps']} cosine steps\n"
                 f"{report['paths']} paths × {report['shots_per_circuit']} shots/basis | output t={report['physical_output_time']:.5f}", fontsize=13, y=.99)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"observables.{extension}", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    for index, (path, axis) in enumerate(zip(measured["indices"], axes.flat)):
        x = np.arange(1, len(hw["excitation_flows"])+1)
        error = 2*np.sqrt(np.maximum(0, np.diag(measured["per_path_current_shot_covariance"][index])))/report["physics"]["J"]
        axis.plot(x, ideal_paths["excitation_flows"][index]/report["physics"]["J"], "s-", color="#1565c0", label="Ideal same path")
        axis.errorbar(x, measured["excitation_flows"][index]/report["physics"]["J"], yerr=error, fmt="o-", color="#c62828", capsize=3, label="Phoenix ±2 shot SE")
        axis.axhline(report["references"]["physical_stationary"]["excitation_flows"][0]/report["physics"]["J"], color="black", linestyle="--", label="Physical steady state")
        axis.set(title=f"Saved trajectory {path}", xlabel="Bond (j, j+1)", ylabel="Excitation current / J", xticks=x)
        axis.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    fig.savefig(output/"currents_per_trajectory.png", dpi=170)
    plt.close(fig)


def save(report, measured, ideal_paths, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n")
    for family in ("populations", "magnetizations", "excitation_flows"):
        with (output/f"{family}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["site_or_left_bond_site_1based", "hardware", "shot_se_fixed_paths", "physical_stationary", "ideal_mean_channel", "ideal_same_4_paths"])
            for i, value in enumerate(report["hardware"][family]):
                writer.writerow([i+1, value, report["shot_standard_errors_fixed_paths"][family][i],
                                 *[report["references"][name][family][i] for name in ("physical_stationary", "ideal_mean_channel", "ideal_same_saved_paths")]])
    comparisons, currents = report["comparisons"], report["mean_bond_currents"]
    lines = ["# IBM Phoenix: saved cosine-run analysis", "",
        f"Job: `{', '.join(report['job_ids'])}`. {report['paths']} trajectories, {report['circuits']} circuits, {report['shots_per_circuit']} shots/circuit, {report['total_shots']:,} total shots.",
        f"N={report['physics']['n_qubits']}, J={report['physics']['J']:g}, gamma={report['physics']['gamma']:g}, reservoirs ({report['physics']['p_left']:g},{report['physics']['p_right']:g}), {report['preparation']} preparation.",
        f"Cosine preparation: {report['preparation_steps']} steps, output time {report['physical_output_time']:.9g}. All measurements are of this final output, not a time series.", "",
        f"**The hardware output fails the {report['tolerance']:g} accuracy target:** maximum local error {comparisons['hardware_vs_physical']['maximum']:.9g}.", "",
        "| Comparison | Max population error | Max current error / J | Combined metric |", "|---|---:|---:|---:|"]
    for label, key in (("Ideal mean output vs physical steady state", "ideal_mean_vs_physical"),
                       ("Ideal same 4 paths vs ideal mean output", "finite_paths_vs_ideal_mean"),
                       ("Ideal same 4 paths vs physical steady state", "same_ideal_paths_vs_physical"),
                       ("Hardware vs ideal same 4 paths", "hardware_vs_same_ideal_paths"),
                       ("Hardware vs physical steady state", "hardware_vs_physical")):
        row = comparisons[key]
        lines.append(f"| {label} | {row['population_max']:.9g} | {row['current_divided_by_J_max']:.9g} | {row['maximum']:.9g} |")
    lines += ["", "The signed observable differences decompose exactly as hardware minus same-path ideal, plus same-path ideal minus mean-channel output, plus mean-channel output minus the physical target. These maximum norms do not add as scalar error contributions.",
        "", "| Spatial mean bond current | Value |", "|---|---:|"]
    for name, value in currents.items():
        lines.append(f"| {name} | {value:.9g} |")
    fraction = report["mean_current_fraction_of_same_ideal_paths"]
    if fraction is not None:
        lines += ["", f"The measured mean bond current retains {100*fraction:.2f}% of the ideal mean current for the same four paths. Its nominal fixed-path shot SE is {report['mean_hardware_bond_current_shot_se']:.6g}, computed with within-setting bond correlations retained."]
    lines += ["", "The four-path ideal average already differs appreciably from the target; increasing shots on these four paths cannot remove this trajectory-selection effect. Hardware currents are further suppressed compared with these exact same ideal paths. Extra shots alone would not remove that systematic discrepancy. This run cannot identify how much comes from two-qubit errors, reset errors, mixture preparation, readout, drift, or other device effects.",
        "", "The physical XX model has a flat interior steady-state population. A sloping measured population profile here is not evidence of diffusive transport. One output measurement also cannot establish that hardware evolution has become stationary.",
        "", "## Plots and uncertainty", "",
        "[Final populations, magnetizations and currents](observables.png) ([PDF](observables.pdf)). [Currents for each trajectory](currents_per_trajectory.png). CSV files contain every site/bond value.",
        "", "Error bars are ±2 nominal shot standard errors conditional on the four selected paths. They include outcome fluctuations from preparation and final measurement, assuming independent shots, and are not simultaneous confidence intervals. Across-path standard errors, which already include shot fluctuations, are stored separately and must not be combined in quadrature with shot SE. The physical tolerance band is ±0.02 in population and current/J, and ±0.04 in magnetization.",
        "", "## Verification", "",
        f"Source integrity: `{report['verification']['source_integrity']}`. A Windows CRLF to Linux LF conversion is accepted only when reconstructing the original line endings reproduces the exact saved SHA-256.",
        f"All {report['circuits']} raw count histograms have the expected width and shot count. Observables, current parity/sign, saved references, and saved assessments were reconstructed and checked. The physical stationary reference was independently evaluated from the XX formula. Only the final `outcome` register is used; preparation records are ignored.",
        f"Recorded native depths: {report['resource_metrics']['native_depth_range']}; CZ gates/circuit: {report['resource_metrics']['cz_gates_per_circuit']}; bath resets/circuit: {report['resource_metrics']['resets_per_circuit']}. Physical path: `{report['resource_metrics']['physical_path']}`.",
        f"ISA QPY file present in this copy: {report['verification']['isa_file_available']}. No circuits were recompiled or submitted, and no calibration data was fetched. Original run files were left unchanged."]
    (output/"Analysis.md").write_text("\n".join(lines)+"\n")
    plot(report, measured, ideal_paths, output)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)
    output = options.output_directory or DIRECTORY/"analysis/hardware"/options.run_directory.name
    if output.resolve() == options.run_directory.resolve():
        raise ValueError("analysis output must be separate from original run files")
    report, measured, ideal_paths = analyze(options.run_directory)
    save(report, measured, ideal_paths, output)
    for name, digest in report["verification"]["original_files_sha256"].items():
        if hashlib.sha256((options.run_directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError("an original run file changed during analysis")
    print(f"Saved: {output/'Analysis.md'}")
    print(json.dumps(report["comparisons"], indent=2))
    return report


if __name__ == "__main__":
    main()
