"""Offline count analysis for collected zero/one/mixture preparation controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model3.Experiment5 import control_quality
from Model3.Experiment5.analyze_hardware import serializable, source_integrity
from Model3.Experiment5.analyze_populations import reconstruct


def analyze(folder):
    raw = {name: (folder/name).read_bytes() for name in ("manifest.json", "hardware_results.json", "source.json")}
    manifest, saved, source = [json.loads(raw[name]) for name in raw]
    cfg = manifest["configuration"]
    if not cfg.get("controls_only") or cfg.get("preparation_steps") != 0:
        raise ValueError("analysis requires a controls-only, zero-evolution run")
    integrity = source_integrity(raw["source.json"], cfg["source_sha256"])
    n = source["configuration"]["physics"]["n_qubits"]
    paths, controls, jobs, total = reconstruct(manifest, n)
    if paths or len(controls) != 3 or len(manifest["circuits"]) != 3:
        raise ValueError("expected exactly three preparation controls")
    expected = dict(all_zero=np.zeros(n), all_one=np.ones(n), initial_mixture=source["preparation"]["populations"])
    path = cfg["physical_path"]
    if not path or len(path) != n+2 or len(set(path)) != n+2:
        raise ValueError("invalid saved physical chain")
    for name, row in controls.items():
        np.testing.assert_allclose(row["expected_populations"], expected[name], atol=1e-14)
        if row["physical_sites"] != path[1:-1]:
            raise ValueError("control physical mapping differs from saved chain")
    if saved["job_ids"] != jobs or len(saved["population_controls"]) != 3:
        raise ValueError("saved control/job coverage mismatch")
    for row in saved["population_controls"]:
        measured = controls[row["control"]]
        np.testing.assert_allclose(measured["populations"], row["populations"], atol=1e-14)
        if row["shots"] != measured["shots"] or row["pub_index"] != measured["pub_index"]:
            raise ValueError("saved control shot/index mismatch")
    for row in manifest["circuits"]:
        ops = row["metrics"]["operation_counts"]
        if row["metrics"]["two_qubit_gates"] != 0 or ops.get("reset", 0):
            raise ValueError("preparation-only circuit contains two-qubit gates or resets")
    assessment = control_quality.assess_controls([dict(control=name, **row) for name, row in controls.items()],
        cfg.get("max_control_error", .03), cfg.get("control_alpha", .05))
    if assessment["passed"] != saved["control_assessment"]["passed"]:
        raise ValueError("saved assessment disagrees with raw controls")
    focus = (cfg.get("path_selection") or {}).get("focus_qubits", [])
    if not set(focus) <= set(path[1:-1]):
        raise ValueError("focus qubits are not measured system sites")
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    qpy = folder/"isa_circuits.qpy"
    archived = folder/"results_bundle.zip"
    qpy_status = "not available"
    if qpy.exists():
        actual = hashlib.sha256(qpy.read_bytes()).hexdigest()
        hashes[qpy.name] = actual
        qpy_status = "verified original QPY SHA-256"
    elif archived.exists():
        with zipfile.ZipFile(archived) as archive:
            actual = hashlib.sha256(archive.read("isa_circuits.qpy")).hexdigest()
        qpy_status = "verified original QPY SHA-256 inside bundle"
    else:
        actual = None
    if actual is not None and actual != cfg["isa_sha256"]:
        raise ValueError("compiled control QPY hash mismatch")
    rows = []
    for row in assessment["sites"]:
        se = row["mixture_residual_se"]
        rows.append({**row, "focus": row["physical_qubit"] in focus,
                     "nominal_standardized_residual": row["mixture_residual"]/se if se > 0 else (0. if row["mixture_residual"] == 0 else None)})
    return dict(experiment="Model3/Experiment5/preparation_analysis", run_directory=str(folder.resolve()),
        configuration=cfg, job_ids=jobs, total_shots=total, physical_sites=path[1:-1], focus_qubits=focus,
        controls=controls, assessment=assessment, sites=rows,
        verification=dict(source_integrity=integrity, raw_controls_reproduced=True, qpy=qpy_status, original_files_sha256=hashes),
        interpretation="Stationary local response diagnostic, not a validated mitigation or a separation of preparation, measurement, timing and drift errors.")


def save(report, output):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if output.resolve().is_relative_to(Path(report["run_directory"]).resolve()):
        raise ValueError("analysis output must be outside original run directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = report["sites"]; x = np.arange(1, len(rows)+1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    axes[0].bar(x-.18, [s["zero_population"] for s in rows], .36, label="All-zero: excitation probability")
    axes[0].bar(x+.18, [1-s["one_population"] for s in rows], .36, label="All-one: missing excitation")
    axes[0].set(ylabel="Observed control error", title="Fresh controls on the same physical chain")
    axes[1].plot(x, report["controls"]["initial_mixture"]["expected_populations"], "k:", label="Intended initial mixture")
    axes[1].plot(x, [s["predicted_mixture"] for s in rows], "s-", color="teal", label="Prediction from zero/one controls")
    axes[1].errorbar(x, [s["mixture_population"] for s in rows],
        yerr=2*np.sqrt(np.diag(report["controls"]["initial_mixture"]["covariance_of_mean"])),
        fmt="o-", color="firebrick", capsize=3, label="Measured mixture ±2 shot SE")
    axes[1].set(ylabel="Population", title="Initial mixture: preparation measurements included; no evolution")
    cutoff = report["assessment"]["nominal_z_cutoff"]
    residual_heights = [s["nominal_standardized_residual"] if s["nominal_standardized_residual"] is not None
                        else float(np.sign(s["mixture_residual"]))*1.3*cutoff for s in rows]
    axes[2].bar(x, residual_heights, color="firebrick")
    for j, s in enumerate(rows):
        if s["nominal_standardized_residual"] is None:
            axes[2].annotate("SE=0", (j+1, residual_heights[j]), ha="center", fontsize=8)
    for sign in (-1, 1):
        axes[2].axhline(sign*cutoff, color="k", linestyle="--", label="Nominal family-adjusted cutoff" if sign == 1 else None)
    axes[2].set(ylabel="Mixture residual / nominal SE", title="Uncertainty includes all three controls")
    for ax in axes:
        for j, row in enumerate(rows):
            if row["focus"]: ax.axvspan(j+.6, j+1.4, color="gold", alpha=.12)
        ax.set(xticks=x, xticklabels=[f"{s['site_1based']}\nq{s['physical_qubit']}" for s in rows], xlabel="Site / physical qubit")
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"preparation_controls.{ext}", dpi=170)
    plt.close(fig)
    with (output/"preparation_controls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    assessment = report["assessment"]
    text = ["# Fresh preparation/readout evaluation", "",
        f"Run `{Path(report['run_directory']).name}`. Jobs: {', '.join(report['job_ids'])}. Three controls, **{report['total_shots']} total shots**, no Trotter evolution.", "",
        f"Physical path: `{report['configuration']['physical_path']}`. Focus physical qubits: `{report['focus_qubits']}` (highlighted in gold). All sites enter the screening.", "",
        f"Diagnostic passed: **{assessment['passed']}**. Failed sites: `{assessment['failed_sites_1based']}`.", "",
        "![Preparation controls](preparation_controls.png)", "",
        "| Site | Physical qubit | Zero | One | Mixture | Zero/one prediction | Residual | Residual SE | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in rows:
        text.append(f"| {s['site_1based']} | {s['physical_qubit']} | {s['zero_population']:.6f} | {s['one_population']:.6f} | {s['mixture_population']:.6f} | {s['predicted_mixture']:.6f} | {s['mixture_residual']:.6f} | {s['mixture_residual_se']:.6f} | {s['passed']} |")
    text += ["", "For intended occupation p, a stable local affine response predicts (1-p) × zero + p × one. At p=0.5 this is the average of the two baseline populations. The mixture residual tests that prediction, including sampling uncertainty from all three measurements.", "",
        f"Baseline errors use one-sided Wilson upper bounds with threshold {assessment['max_baseline_error']:.3%}. Mixture tests use a nominal family-adjusted cutoff of {assessment['nominal_z_cutoff']:.3f}, with family alpha {assessment['family_alpha']}. These screening choices are separate from the 0.02 evolved-state accuracy target.", "",
        "Large zero/one errors point to a baseline preparation/readout concern. A mixture inconsistent with their prediction suggests that a stationary local affine response does not describe these preparations; preparation gates, initial measurement timing, crosstalk and drift remain possible contributors. This diagnostic does not separate those mechanisms or justify automatic inverse-response correction.", "",
        "The zero and one controls use deterministic preparations followed by final Z readout. The mixture control retains the original unconditioned initial preparation measurements, whose records are ignored, then final Z readout. There are no system exchange gates, bath collisions or bath resets. No full evolution run is automatically submitted, regardless of the outcome.", "",
        f"Integrity: {report['verification']['qpy']}; source hash and raw-count reconstruction checked. Individual control shots: `{ {name: c['shots'] for name, c in report['controls'].items()} }`.", ""]
    for s in rows:
        if s["reasons"]: text.append(f"Site {s['site_1based']} (physical {s['physical_qubit']}): {'; '.join(s['reasons'])}.")
    (output/"Analysis.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    for name, value in report["verification"]["original_files_sha256"].items():
        if hashlib.sha256((Path(report["run_directory"])/name).read_bytes()).hexdigest() != value:
            raise ValueError("an original control input changed during analysis")
    (output/"analysis.json").write_text(json.dumps(serializable(report), indent=2)+"\n", encoding="utf-8")


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)
    report = analyze(options.run_directory)
    output = options.output_directory or Path(__file__).resolve().parent/"analysis"/"hardware"/options.run_directory.name/"preparation"
    save(report, output)
    print(f"Preparation analysis: {output/'Analysis.md'}")
    return report


if __name__ == "__main__":
    main()
