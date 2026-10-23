"""Analyze collected Z-only trajectories and controls from saved files only.

No Runtime connection, simulator, new evolution, or edits to the original run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_hardware import moments, serializable, source_integrity
except ImportError:
    from analyze_hardware import moments, serializable, source_integrity


def population_error(value, reference):
    delta = np.asarray(value)-reference
    return {"mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta**2))),
            "maximum": float(np.abs(delta).max()), "worst_site_1based": int(np.argmax(np.abs(delta))+1)}


def reconstruct(manifest, n):
    if manifest.get("status") != "collected" or manifest.get("experiment") != "Model3/Experiment5/hardware":
        raise ValueError("need a completely collected Model3 hardware run")
    config = manifest["configuration"]
    records = manifest["circuits"]
    if [r["pub_index"] for r in records] != list(range(len(records))):
        raise ValueError("publication indices must be contiguous")
    observations, job_ids, stop = [], [], 0
    for batch in manifest["batches"]:
        if batch["status"] != "collected" or batch["start"] != stop or not batch.get("job_id"):
            raise ValueError("incomplete or unordered batches")
        if len(batch["observations"]) != batch["stop"]-batch["start"]:
            raise ValueError("batch count disagrees with circuit range")
        observations.extend(batch["observations"])
        stop = batch["stop"]
        job_ids.append(batch["job_id"])
    if stop != len(records) or len(observations) != len(records):
        raise ValueError("missing circuit observations")
    controls, paths = {}, {}
    total_shots = 0
    system_mapping = None
    for record, observation in zip(records, observations):
        if record["basis"] != "z":
            raise ValueError("this analyzer requires Z-only measurements")
        value, covariance, shots = moments(observation["outcome_counts"], n)
        if shots != record.get("shots", config["shots"]):
            raise ValueError("per-circuit shot count mismatch")
        total_shots += shots
        mapping = record["final_logical_to_physical"][:n]
        if len(mapping) != n or len(set(mapping)) != n:
            raise ValueError("invalid final system mapping")
        if system_mapping is None:
            system_mapping = mapping
        if mapping != system_mapping:
            raise ValueError("controls require the same physical site mapping for every circuit")
        row = {"populations": value, "covariance_of_mean": covariance, "shots": shots,
               "pub_index": record["pub_index"], "physical_sites": mapping}
        if record.get("kind") == "control":
            name = record["control"]
            if name in controls:
                raise ValueError("duplicate control")
            row["expected_populations"] = np.asarray(record["expected_populations"])
            controls[name] = row
        else:
            index = record["trajectory"]
            if index in paths:
                raise ValueError("duplicate trajectory")
            paths[index] = row
    if list(paths) != config["trajectory_indices"]:
        raise ValueError("trajectory selection disagrees with manifest")
    if config.get("population_controls") and set(controls) != {"all_zero", "all_one", "initial_mixture"}:
        raise ValueError("expected all three population controls")
    if total_shots != config.get("total_shots", total_shots) or len(records) != config.get("total_circuits", len(records)):
        raise ValueError("total circuit/shot count mismatch")
    return paths, controls, job_ids, total_shots


def control_diagnostics(controls, population, covariance):
    """Test a stationary local affine response; adjustment is exploratory only.

    All-zero/one experiments contain preparation errors as well as readout.
    An initial-mixture experiment provides an independent consistency check.
    """
    a, b, mixture = (controls[name] for name in ("all_zero", "all_one", "initial_mixture"))
    low, high = a["populations"], b["populations"]
    fraction = mixture["expected_populations"]
    expected = (1-fraction)*low+fraction*high
    predicted_cov = ((1-fraction)[:, None]*a["covariance_of_mean"]*(1-fraction)[None, :]
                     +fraction[:, None]*b["covariance_of_mean"]*fraction[None, :])
    residual = mixture["populations"]-expected
    residual_cov = mixture["covariance_of_mean"]+predicted_cov
    se = np.sqrt(np.maximum(0, np.diag(residual_cov)))
    standardized = [float(value/error) if error > 0 else (0. if value == 0 else None)
                    for value, error in zip(residual, se)]
    span = high-low
    if np.any(span <= 0):
        raise ValueError("all-one minus all-zero response must be positive")
    adjusted = (population-low)/span
    derivatives = (1/span, (population-high)/span**2, -(population-low)/span**2)
    adjusted_cov = sum(d[:, None]*cov*d[None, :] for d, cov in zip(derivatives,
        (covariance, a["covariance_of_mean"], b["covariance_of_mean"])))
    return {"predicted_initial_mixture": expected, "initial_mixture_residual": residual,
            "initial_mixture_residual_covariance": residual_cov, "initial_mixture_residual_se": se,
            "initial_mixture_residual_nominal_z": standardized,
            "exploratory_affine_adjusted_populations": adjusted,
            "exploratory_adjusted_covariance_of_mean": adjusted_cov,
            "interpretation": "Local affine response assumes stable errors, independent control samples and ideal known preparations. The controls include preparation errors and were different circuits at different positions. This adjustment is not validated mitigation and does not identify a noise mechanism."}


def analyze(folder):
    filenames = ["manifest.json", "hardware_results.json", "source.json", "backend_snapshot.json"]
    raw = {name: (folder/name).read_bytes() for name in filenames if (folder/name).exists()}
    manifest, saved, source = (json.loads(raw[name]) for name in filenames[:3])
    config, physics = manifest["configuration"], source["configuration"]["physics"]
    integrity = source_integrity(raw["source.json"], config["source_sha256"])
    n = physics["n_qubits"]
    paths, controls, job_ids, total = reconstruct(manifest, n)
    indices = list(paths)
    values = np.array([row["populations"] for row in paths.values()])
    covariance = sum(row["covariance_of_mean"] for row in paths.values())/len(paths)**2
    population = values.mean(axis=0)
    covariances = np.asarray(source["final_covariances_real_gauge"]).reshape(-1, n, n)[indices]
    np.testing.assert_allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(covariances)
    if eigenvalues.min() < -1e-7 or eigenvalues.max() > 1+1e-7:
        raise ValueError("nonphysical saved ideal covariance")
    ideal_paths = np.diagonal(covariances, axis1=-2, axis2=-1)
    ideal = ideal_paths.mean(axis=0)
    J, gamma, prefactor = physics["J"], physics["gamma"], physics["exchange_prefactor"]
    if physics.get("delta", 0) != 0 or min(J, gamma, prefactor) <= 0:
        raise ValueError("requires the positive-rate XX model")
    hopping = 2*prefactor*J
    current = 2*hopping**2*gamma/(gamma**2+4*hopping**2)*(physics["p_left"]-physics["p_right"])
    physical = np.full(n, (physics["p_left"]+physics["p_right"])/2)
    physical[[0, -1]] = [physics["p_left"]-current/gamma, physics["p_right"]+current/gamma]
    references = {"physical_stationary": physical, "ideal_same_saved_paths": ideal,
        "ideal_mean_channel": np.asarray(source["mean_channel_observables"]["populations"])[0, -1]}
    np.testing.assert_allclose(values, saved["per_trajectory"]["populations"], atol=1e-13)
    np.testing.assert_allclose(population, saved["observables"]["populations"], atol=1e-13)
    if saved["per_trajectory"]["indices"] != indices or saved["job_ids"] != job_ids:
        raise ValueError("saved trajectory/job IDs disagree with manifest")
    for name, reference in references.items():
        np.testing.assert_allclose(reference, saved["references"][name]["populations"], atol=1e-12)
        np.testing.assert_allclose(population_error(population, reference)["maximum"], saved["local_observable_errors"][name], atol=1e-12)
    for row in saved["population_controls"]:
        actual = controls[row["control"]]
        if row["shots"] != actual["shots"] or row["pub_index"] != actual["pub_index"]:
            raise ValueError("saved control metadata mismatch")
        np.testing.assert_allclose(row["populations"], actual["populations"], atol=1e-13)
        np.testing.assert_allclose(row["expected_populations"], actual["expected_populations"], atol=1e-13)
    expected_controls = {"all_zero": np.zeros(n), "all_one": np.ones(n),
                         "initial_mixture": np.asarray(source["preparation"]["populations"])}
    if len(saved["population_controls"]) != len(controls):
        raise ValueError("saved control count mismatch")
    for name, row in controls.items():
        np.testing.assert_allclose(row["expected_populations"], expected_controls[name])
    across_cov = np.cov(values, rowvar=False, ddof=1)/len(paths)
    paired_cov = np.cov(values-ideal_paths, rowvar=False, ddof=1)/len(paths)
    np.testing.assert_allclose(np.sqrt(np.diag(across_cov)), saved["across_path_standard_errors"]["populations"], atol=1e-13)
    diagnostics = control_diagnostics(controls, population, covariance)
    snapshot = json.loads(raw["backend_snapshot.json"]) if "backend_snapshot.json" in raw else None
    if snapshot is not None:
        snapshot = {key: value for key, value in snapshot.items() if key != "backend_properties"}
        snapshot["full_backend_properties_retained_in_original_snapshot"] = True
    report = {"run_directory": str(folder.resolve()), "configuration": config, "physics": physics,
        "job_ids": job_ids, "total_shots": total, "total_circuits": len(manifest["circuits"]),
        "hardware_populations": population, "per_path_populations": values, "per_path_ideal_populations": ideal_paths,
        "path_indices": indices, "physical_sites": next(iter(paths.values()))["physical_sites"],
        "references": references, "controls": controls, "control_diagnostics": diagnostics,
        "shot_covariance_fixed_paths": covariance, "across_path_covariance_of_mean": across_cov,
        "paired_difference_covariance_of_mean": paired_cov,
        "shot_se_fixed_paths": np.sqrt(np.maximum(0, np.diag(covariance))),
        "across_path_se_including_shots": np.sqrt(np.maximum(0, np.diag(across_cov))),
        "comparisons": {name: population_error(population, ref) for name, ref in references.items()},
        "ideal_same_paths_vs_physical": population_error(ideal, physical),
        "exploratory_adjusted_comparisons": {name: population_error(diagnostics["exploratory_affine_adjusted_populations"], ref)
                                             for name, ref in references.items()},
        "snapshot": snapshot,
        "batches": [{key: value for key, value in batch.items() if key != "observations"} for batch in manifest["batches"]],
        "verification": {"source_integrity": integrity, "counts_and_saved_observables_reproduced": True,
                         "original_files_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
                         "isa_qpy_available": (folder/"isa_circuits.qpy").exists()}}
    evolved = [row for row in manifest["circuits"] if row.get("kind") != "control"]
    report["resources"] = {"native_depth_range": [min(r["metrics"]["depth"] for r in evolved), max(r["metrics"]["depth"] for r in evolved)],
                           "cz_counts": sorted({r["metrics"]["operation_counts"].get("cz", 0) for r in evolved}),
                           "reset_counts": sorted({r["metrics"]["operation_counts"].get("reset", 0) for r in evolved})}
    if report["verification"]["isa_qpy_available"]:
        digest = hashlib.sha256((folder/"isa_circuits.qpy").read_bytes()).hexdigest()
        if digest != config["isa_sha256"]:
            raise ValueError("original ISA hash mismatch")
        report["verification"]["original_files_sha256"]["isa_circuits.qpy"] = digest
    return report


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values, refs = report["hardware_populations"], report["references"]
    x = np.arange(1, len(values)+1)
    fig, axis = plt.subplots(figsize=(11, 5.3), layout="constrained")
    tolerance = .02
    axis.fill_between(x, refs["physical_stationary"]-tolerance, refs["physical_stationary"]+tolerance,
                      color="gray", alpha=.13, label="Physical target ±0.02")
    axis.plot(x, refs["physical_stationary"], "k--", label="Physical stationary target")
    axis.plot(x, refs["ideal_mean_channel"], ":", color="#666666", label="Ideal mean channel")
    axis.plot(x, refs["ideal_same_saved_paths"], "s-", color="#00897b", label="Ideal: same 128 paths")
    axis.errorbar(x, values, yerr=2*report["shot_se_fixed_paths"], fmt="o-", color="#b71c1c", capsize=3,
                  label="Phoenix: 128 paths × 64 shots, ±2 shot SE")
    axis.set(xlabel="Site", ylabel="Population", xticks=x, title="IBM Phoenix: populations after 72 cosine steps")
    axis.grid(alpha=.2); axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"populations.{suffix}", dpi=180)
    plt.close(fig)

    controls, diag = report["controls"], report["control_diagnostics"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    a, b, mixture = (controls[key] for key in ("all_zero", "all_one", "initial_mixture"))
    axes[0].bar(x-.18, a["populations"], width=.36, label="All-zero: measured excitation")
    axes[0].bar(x+.18, 1-b["populations"], width=.36, label="All-one: missing excitation")
    axes[0].set(ylabel="Control error", title="512 shots/control; includes preparation and readout")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, mixture["expected_populations"], "k--", label="Intended mixture")
    axes[1].plot(x, diag["predicted_initial_mixture"], "s-", color="#00897b", label="Prediction from zero/one controls")
    axes[1].errorbar(x, mixture["populations"], yerr=2*np.sqrt(np.diag(mixture["covariance_of_mean"])),
                     fmt="o-", color="#b71c1c", capsize=3, label="Initial-mixture control ±2 shot SE")
    axes[1].set(ylabel="Population", title="No Trotter evolution in these control circuits")
    axes[1].legend(fontsize=8)
    snapshot = report["snapshot"]
    if snapshot:
        props = [snapshot["qubits"].get(str(q), {}) for q in report["physical_sites"]]
        for shift, key, color in ((-.18, "t1", "#1565c0"), (.18, "t2", "#e69f00")):
            axes[2].bar(x+shift, [(p.get(key) or 0)*1e6 for p in props], width=.36, color=color, label=key.upper())
        axes[2].set(ylabel="Calibration time (µs)", title="Compilation-time calibration on the selected physical system qubits")
        axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set(xlabel="Site", xticks=x); axis.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"controls_and_calibration.{suffix}", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    residual = (report["per_path_populations"]-report["per_path_ideal_populations"]).T
    bound = np.abs(residual).max()
    im = axis.imshow(residual, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound,
                     extent=[-.5, len(report["path_indices"])-.5, len(x)+.5, .5])
    fig.colorbar(im, ax=axis, label="Hardware population − same-path ideal")
    axis.set(xlabel="Saved trajectory index (not calibrated execution time)", ylabel="Site", yticks=x,
             title="Population discrepancies across the 128 trajectories")
    fig.savefig(output/"per_path_residuals.png", dpi=180); plt.close(fig)
    if report.get("previous_runs"):
        fig, axis = plt.subplots(figsize=(10.5, 5), layout="constrained")
        axis.plot(x, refs["physical_stationary"], "k--", label="Physical stationary target")
        for row in report["previous_runs"]:
            axis.plot(x, row["hardware_populations"], "o-", label=row["label"])
        axis.plot(x, values, "o-", color="#b71c1c", label="128 paths × 64 shots (new)")
        axis.set(xlabel="Site", ylabel="Population", xticks=x, title="Population profiles from the three Phoenix runs")
        axis.legend(fontsize=8); axis.grid(alpha=.2)
        fig.savefig(output/"previous_runs.png", dpi=180); plt.close(fig)


def report_markdown(report):
    controls, diag = report["controls"], report["control_diagnostics"]
    values, refs = report["hardware_populations"], report["references"]
    lines = ["# Phoenix population diagnostic: 128 trajectories × 64 shots", "",
        f"Run `{Path(report['run_directory']).name}`; jobs {', '.join(report['job_ids'])}. All {report['total_circuits']} circuits and {report['total_shots']} shots collected. Control shots are excluded from evolved populations.", "",
        "## Main result", "",
        f"Maximum hardware population error against physical stationary values: **{report['comparisons']['physical_stationary']['maximum']:.6f}** (target 0.02). MAE: **{report['comparisons']['physical_stationary']['mae']:.6f}**.",
        f"Maximum discrepancy against the ideal for the same 128 paths: **{report['comparisons']['ideal_same_saved_paths']['maximum']:.6f}**; MAE **{report['comparisons']['ideal_same_saved_paths']['mae']:.6f}**.", "",
        "[Actual population plot](populations.png) ([PDF](populations.pdf)); [controls/calibration](controls_and_calibration.png) ([PDF](controls_and_calibration.pdf)); [per-path residuals](per_path_residuals.png).", "",
        "| Site | Physical qubit | Hardware | 2 shot SE | Ideal same paths | Physical target |", "|---:|---:|---:|---:|---:|---:|"]
    for j, value in enumerate(values):
        lines.append(f"| {j+1} | {report['physical_sites'][j]} | {value:.6f} | {2*report['shot_se_fixed_paths'][j]:.6f} | {refs['ideal_same_saved_paths'][j]:.6f} | {refs['physical_stationary'][j]:.6f} |")
    lines += ["", "The ±2-SE plot bars describe shot fluctuations conditional on these fixed paths. The separate across-path SEs include variation between the selected paths and must not be added to those bars. Both assume independent sampling and exclude systematic errors or device drift. There is only one job, so job-cluster uncertainty is not estimable.", "",
        "## Controls and local-response consistency", "",
        "Every control used 512 shots. Initial-mixture circuits contain the original preparation measurements and final readout, with no Trotter evolution.", "",
        "| Site | All-zero population | All-one population | Initial mixture | Predicted mixture from zero/one | Residual / nominal SE |", "|---:|---:|---:|---:|---:|---:|"]
    for j in range(len(values)):
        z = diag["initial_mixture_residual_nominal_z"][j]
        lines.append(f"| {j+1} | {controls['all_zero']['populations'][j]:.6f} | {controls['all_one']['populations'][j]:.6f} | {controls['initial_mixture']['populations'][j]:.6f} | {diag['predicted_initial_mixture'][j]:.6f} | {z:.3f} |")
    worst = int(np.argmax([abs(z) if z is not None else 0 for z in diag["initial_mixture_residual_nominal_z"]]))
    qubit = report["physical_sites"][worst]
    lines += ["", f"The largest initial-mixture consistency discrepancy is **site {worst+1}, physical qubit {qubit}**: "
        f"measured population **{controls['initial_mixture']['populations'][worst]:.6f}**, intended "
        f"**{controls['initial_mixture']['expected_populations'][worst]:.6f}**, while the zero/one controls predict "
        f"**{diag['predicted_initial_mixture'][worst]:.6f}**. The latter discrepancy is "
        f"**{diag['initial_mixture_residual_nominal_z'][worst]:.2f} nominal SE**, including shot uncertainty in all three controls.", ""]
    lines += ["", "For a stationary independent single-qubit response and a correctly prepared population p, the zero/one controls predict a+(b−a)p. Their shot uncertainty is included in the residual SE. The controls also contain preparation errors and use different circuits at different submission positions; disagreement can reflect preparation, measurement timing, crosstalk, or changing device behavior. These nominal standardized residuals are diagnostics, not certified simultaneous significance tests.", "",
        "An exploratory inverse affine adjustment (raw−a)/(b−a), without clipping, gives population MAE against physical of "
        f"**{report['exploratory_adjusted_comparisons']['physical_stationary']['mae']:.6f}**, versus raw **{report['comparisons']['physical_stationary']['mae']:.6f}**. Its maximum error is **{report['exploratory_adjusted_comparisons']['physical_stationary']['maximum']:.6f}**. Because the initial-mixture control tests the response assumptions and contains a substantial discrepancy, these adjusted values are not presented as validated mitigated results.", ""]
    if report["snapshot"]:
        snapshot = report["snapshot"]
        lines += ["## Calibration evidence", "", f"Snapshot captured at {snapshot['captured_at_utc']}. Values describe compilation-time calibration, not a direct measurement of errors during this job.", "",
                  "| Site | Physical qubit | T1 (µs) | T2 (µs) |", "|---:|---:|---:|---:|"]
        for j, q in enumerate(report["physical_sites"]):
            prop = snapshot["qubits"].get(str(q), {})
            lines.append(f"| {j+1} | {q} | {(prop.get('t1') or 0)*1e6:.3f} | {(prop.get('t2') or 0)*1e6:.3f} |")
        prop = snapshot["qubits"].get(str(qubit), {})
        lines += ["", f"The same flagged physical qubit {qubit} has reported **T1 = {(prop.get('t1') or 0)*1e6:.2f} µs** "
                  f"and **T2 = {(prop.get('t2') or 0)*1e6:.2f} µs**. This makes a preparation/timing/coherence problem a concrete concern, although these data do not isolate it from other errors."]
        pairs = {}
        for row in snapshot["instructions"]:
            if row["name"] == "cz" and row["error"] is not None:
                key = tuple(sorted(row["qubits"])); pairs[key] = max(pairs.get(key, 0), row["error"])
        lines += ["", "| Physical CZ pair | Reported error (%) |", "|---|---:|"]
        for pair, error in sorted(pairs.items(), key=lambda row: row[1], reverse=True):
            lines.append(f"| {pair[0]}–{pair[1]} | {100*error:.4f} |")
        lines += ["", "Short relaxation times and large adjacent CZ errors make a qubit a concrete layout concern. They do not establish how much of the full-chain population discrepancy is caused by that qubit. The worst evolved-population discrepancies need not occur at that same site because interactions couple the chain.", "",
            "IBM documents calibration properties in [QPU information](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information), and the timing cost of mid-circuit operations in [dynamic circuits](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits). These mechanisms are plausible interpretations, not measurements of a specific causal noise channel in this dataset.", ""]
    if report.get("previous_runs"):
        lines += ["## Previous population runs", "", "[Population profile comparison](previous_runs.png).", "",
            "| Run | MAE vs physical | Max vs physical | MAE vs matched ideal | Max vs matched ideal |", "|---|---:|---:|---:|---:|"]
        for row in [*report["previous_runs"], {"label": "128 paths × 64 shots (new)", "comparisons": report["comparisons"]}]:
            metrics = row["comparisons"]
            lines.append(f"| {row['label']} | {metrics['physical_stationary']['mae']:.6f} | {metrics['physical_stationary']['maximum']:.6f} | {metrics['ideal_same_saved_paths']['mae']:.6f} | {metrics['ideal_same_saved_paths']['maximum']:.6f} |")
        lines += ["", "Shot allocation, path selection, and execution occasions differ, so this is descriptive rather than a controlled test of shot allocation. Repeated shots sharpen estimates but do not guarantee improved mean hardware fidelity.", ""]
    lines += ["## Suggested next step", "",
        f"Select a new connected chain using current calibration quality, avoiding physical qubit {qubit} and the highest-error adjacent CZ edges. Validate the three short controls on that chain before spending another full-depth run. Keep the cosine schedule fixed during the layout comparison, and retain the newly compiled circuits. This is a recommendation based on the combined control/calibration evidence, not proof that changing one qubit will resolve all population discrepancies.", ""]
    lines += ["## Verification and limits", "",
        f"Native depth range: {report['resources']['native_depth_range']}; CZ counts {report['resources']['cz_counts']}; resets {report['resources']['reset_counts']}. Saved source integrity: `{report['verification']['source_integrity']}`. Original ISA QPY available: **{report['verification']['isa_qpy_available']}**.", "",
        "All raw count widths, shot counts, trajectory selection, mappings, control separation, job coverage, saved means, references, and nominal across-path SEs were independently reconstructed and checked. Physical stationary populations were independently reconstructed analytically. Hashes verify original files were preserved.", "",
        "Only final Z populations were measured; currents and hardware stationarity are not established. Submission index is not calibrated execution time. No noisy simulation, trajectory evolution, hardware query, new job, or automatic correction was performed. Job timing metadata are included in analysis.json; job wall time is not an individual circuit duration.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--compare-analysis", type=Path, nargs="*", default=[])
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_directory)
    output = args.output_directory or Path(__file__).resolve().parent/"analysis"/"hardware"/args.run_directory.name
    if output.resolve() == args.run_directory.resolve() or args.run_directory.resolve() in output.resolve().parents:
        raise ValueError("analysis output must be outside the original run directory")
    output.mkdir(parents=True, exist_ok=True)
    compare_hashes = {}
    report["previous_runs"] = []
    for filename in args.compare_analysis:
        raw = filename.read_bytes(); other = json.loads(raw)
        compare_hashes[str(filename)] = hashlib.sha256(raw).hexdigest()
        hardware = np.asarray(other["hardware"]["populations"])
        refs = other["references"]
        np.testing.assert_allclose(refs["physical_stationary"]["populations"], report["references"]["physical_stationary"], atol=1e-12)
        report["previous_runs"].append({"label": f"{other['paths']} paths × {other['shots_per_circuit']} shots",
            "run_directory": other["run_directory"], "hardware_populations": hardware,
            "comparisons": {name: population_error(hardware, refs[name]["populations"])
                            for name in ("physical_stationary", "ideal_same_saved_paths")}})
    rows, controls, diagnostic = [], report["controls"], report["control_diagnostics"]
    for j in range(len(report["hardware_populations"])):
        rows.append({"site_1based": j+1, "physical_qubit": report["physical_sites"][j],
            "hardware": report["hardware_populations"][j], "shot_se_fixed_paths": report["shot_se_fixed_paths"][j],
            "across_path_se_including_shots": report["across_path_se_including_shots"][j],
            **{name: ref[j] for name, ref in report["references"].items()},
            "exploratory_affine_adjustment": diagnostic["exploratory_affine_adjusted_populations"][j]})
    write_csv(output/"populations.csv", rows)
    rows = [{"site_1based": j+1, "physical_qubit": report["physical_sites"][j],
             **{name: row["populations"][j] for name, row in controls.items()},
             "predicted_initial_mixture": diagnostic["predicted_initial_mixture"][j],
             "residual_se": diagnostic["initial_mixture_residual_se"][j],
             "residual_nominal_z": diagnostic["initial_mixture_residual_nominal_z"][j]}
             for j in range(len(report["hardware_populations"]))]
    write_csv(output/"controls.csv", rows)
    plot(report, output)
    (output/"Analysis.md").write_text(report_markdown(report))
    for name, digest in report["verification"]["original_files_sha256"].items():
        if hashlib.sha256((args.run_directory/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"original input changed: {name}")
    for name, digest in compare_hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"comparison input changed: {name}")
    report["verification"]["comparison_file_sha256"] = compare_hashes
    report["verification"]["original_inputs_unchanged"] = True
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n")
    print(json.dumps({"output": str(output), "comparisons": report["comparisons"],
        "control_residual_nominal_z": report["control_diagnostics"]["initial_mixture_residual_nominal_z"]}, indent=2))


if __name__ == "__main__":
    main()
