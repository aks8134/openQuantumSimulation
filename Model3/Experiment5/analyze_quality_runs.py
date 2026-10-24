"""Analyze the saved October 23 controls and October 24 population run offline.

No account access, circuit execution, or writes to original run directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

try:
    from .analyze_hardware import serializable, source_integrity
    from .analyze_populations import reconstruct, population_error, write_csv
    from .control_quality import assess_controls
except ImportError:
    from analyze_hardware import serializable, source_integrity
    from analyze_populations import reconstruct, population_error, write_csv
    from control_quality import assess_controls


CONTROLS = "ibm_phoenix_20261023T192723_098429Z"
FULL = "ibm_phoenix_20261024T042722_287349Z"
PREVIOUS = "ibm_phoenix_20261023T135506_304864Z"


def load_run(folder, verify_bundle=True):
    names = ["manifest.json", "hardware_results.json", "source.json", "backend_snapshot.json"]
    raw = {name: (folder/name).read_bytes() for name in names}
    manifest, saved, source, snapshot = [json.loads(raw[name]) for name in names]
    config = manifest["configuration"]
    integrity = source_integrity(raw["source.json"], config["source_sha256"])
    n = source["configuration"]["physics"]["n_qubits"]
    paths, controls, jobs, shots = reconstruct(manifest, n)
    assert saved["job_ids"] == jobs
    assert saved["per_trajectory"]["indices"] == list(paths)
    if paths:
        values = np.array([r["populations"] for r in paths.values()])
        np.testing.assert_allclose(values, saved["per_trajectory"]["populations"], atol=1e-13)
        np.testing.assert_allclose(values.mean(axis=0), saved["observables"]["populations"], atol=1e-13)
        np.testing.assert_allclose(values.std(axis=0, ddof=1)/np.sqrt(len(paths)),
                                   saved["across_path_standard_errors"]["populations"], atol=1e-13)
    assert len(saved["population_controls"]) == len(controls)
    for row in saved["population_controls"]:
        actual = controls[row["control"]]
        np.testing.assert_allclose(actual["populations"], row["populations"], atol=1e-13)
        np.testing.assert_array_equal(actual["expected_populations"], row["expected_populations"])
        assert actual["shots"] == row["shots"]
        assert actual["pub_index"] == row["pub_index"]
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    bundle_check = None
    if verify_bundle:
        bundle = folder/"results_bundle.zip"
        with zipfile.ZipFile(bundle) as archive:
            assert archive.testzip() is None
            for name in names:
                assert json.loads(archive.read(name)) == json.loads(raw[name])
            assert hashlib.sha256(archive.read("source.json")).hexdigest() == config["source_sha256"]
            assert hashlib.sha256(archive.read("isa_circuits.qpy")).hexdigest() == config["isa_sha256"]
        hashes[bundle.name] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        bundle_check = "ZIP CRCs, JSON contents, original source SHA-256 and ISA QPY SHA-256 verified"
    return dict(folder=folder, manifest=manifest, saved=saved, source=source, snapshot=snapshot,
                paths=paths, controls=controls, jobs=jobs, shots=shots, hashes=hashes,
                source_integrity=integrity, bundle_check=bundle_check)


def analyze(base):
    control = load_run(base/CONTROLS)
    full = load_run(base/FULL)
    old = load_run(base/PREVIOUS, verify_bundle=False)
    config = full["manifest"]["configuration"]
    assert control["manifest"]["configuration"]["controls_only"]
    assert not full["controls"] and not control["paths"]
    assert config["physical_path"] == control["manifest"]["configuration"]["physical_path"]
    assert full["source"] == control["source"]
    assert config["source_sha256"] == control["manifest"]["configuration"]["source_sha256"]
    mapping = next(iter(full["paths"].values()))["physical_sites"]
    assert all(row["physical_sites"] == mapping for row in control["controls"].values())
    assessment = assess_controls([dict(control=name, **row) for name, row in control["controls"].items()])
    assert assessment == control["saved"]["control_assessment"]
    assert assessment == config["path_selection"]["control_assessment"]
    values = np.array([r["populations"] for r in full["paths"].values()])
    indices = list(full["paths"])
    n = values.shape[1]
    physics = full["source"]["configuration"]["physics"]
    assert physics.get("delta", 0) == 0 and n > 2
    hopping = 2*physics["exchange_prefactor"]*physics["J"]
    gamma = physics["gamma"]
    flow = 2*hopping**2*gamma/(gamma**2+4*hopping**2)*(physics["p_left"]-physics["p_right"])
    physical = np.full(n, (physics["p_left"]+physics["p_right"])/2)
    physical[[0, -1]] = [physics["p_left"]-flow/gamma, physics["p_right"]+flow/gamma]
    covariances = np.asarray(full["source"]["final_covariances_real_gauge"]).reshape(-1, n, n)[indices]
    np.testing.assert_allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(covariances)
    assert eigenvalues.min() >= -1e-7 and eigenvalues.max() <= 1+1e-7
    ideal_paths = np.diagonal(covariances, axis1=-2, axis2=-1)
    refs = dict(physical_stationary=physical, ideal_same_saved_paths=ideal_paths.mean(axis=0),
                ideal_mean_channel=np.asarray(full["source"]["mean_channel_observables"]["populations"])[0, -1])
    population = values.mean(axis=0)
    comparisons = {name: population_error(population, ref) for name, ref in refs.items()}
    for name, ref in refs.items():
        np.testing.assert_allclose(ref, full["saved"]["references"][name]["populations"], atol=1e-12)
        np.testing.assert_allclose(comparisons[name]["maximum"], full["saved"]["local_observable_errors"][name], atol=1e-12)
    shot_cov = sum(row["covariance_of_mean"] for row in full["paths"].values())/len(values)**2
    across_cov = np.cov(values, rowvar=False, ddof=1)/len(values)
    paired_cov = np.cov(values-ideal_paths, rowvar=False, ddof=1)/len(values)
    assert list(old["paths"]) == indices
    assert old["source"] == full["source"]
    old_ideal = np.asarray(old["source"]["final_covariances_real_gauge"]).reshape(-1, n, n)[indices]
    np.testing.assert_allclose(covariances, old_ideal, atol=1e-13)
    old_pop = np.mean([row["populations"] for row in old["paths"].values()], axis=0)
    old_comparisons = {name: population_error(old_pop, ref) for name, ref in refs.items()}
    resources = [row["metrics"] for row in full["manifest"]["circuits"]]
    report = dict(full_run=FULL, control_run=CONTROLS, previous_run=PREVIOUS,
        physical_path=config["physical_path"], physical_sites=mapping, control_assessment=assessment,
        hardware_populations=population, references=refs, comparisons=comparisons,
        previous_populations=old_pop, previous_comparisons=old_comparisons,
        ideal_vs_physical=population_error(refs["ideal_same_saved_paths"], physical),
        shot_se_fixed_paths=np.sqrt(np.diag(shot_cov)), across_path_se=np.sqrt(np.diag(across_cov)),
        paired_residual_se=np.sqrt(np.diag(paired_cov)),
        residual_nominal_z=(population-refs["ideal_same_saved_paths"])/np.sqrt(np.diag(paired_cov)),
        within_tolerance_sites_1based=(np.flatnonzero(abs(population-physical) <= .02)+1).tolist(),
        path_metrics=config["path_selection"]["metrics"],
        resources=dict(depth_range=[min(r["depth"] for r in resources), max(r["depth"] for r in resources)],
                       cz_counts=sorted({r["operation_counts"].get("cz", 0) for r in resources}),
                       reset_counts=sorted({r["operation_counts"].get("reset", 0) for r in resources}),
                       two_qubit_depth=sorted({r["two_qubit_depth"] for r in resources})),
        jobs={name: run["jobs"] for name, run in [(CONTROLS, control), (FULL, full)]},
        job_timing={name: [b.get("job_metrics", {}) for b in run["manifest"]["batches"]]
                    for name, run in [(CONTROLS, control), (FULL, full)]},
        verification={"same_128_saved_choices_and_ideal_covariances_as_previous_run": True,
                      "controls_and_full_share_source_and_mapping": True,
                      "raw_counts_reproduce_saved_means_and_standard_errors": True,
                      "inputs": {str(run["folder"]): dict(hashes=run["hashes"], source_integrity=run["source_integrity"],
                                                        bundle_check=run["bundle_check"]) for run in (control, full, old)}})
    return report


def outputs(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    r = report
    x = np.arange(1, len(r["physical_sites"])+1)
    physical = r["references"]["physical_stationary"]
    ideal = r["references"]["ideal_same_saved_paths"]
    fig, ax = plt.subplots(figsize=(11, 5.3), layout="constrained")
    ax.fill_between(x, physical-.02, physical+.02, color="gray", alpha=.16, label="Physical target ±0.02")
    ax.plot(x, physical, "k--", label="Physical stationary populations")
    ax.plot(x, ideal, "s-", color="teal", label="Ideal: identical 128 trajectories")
    ax.plot(x, r["previous_populations"], "o--", color="gray", alpha=.8, label="Previous Phoenix chain: 128 × 64 shots")
    ax.errorbar(x, r["hardware_populations"], yerr=2*r["shot_se_fixed_paths"], fmt="o-", color="firebrick", capsize=3,
                label="New Phoenix chain: 128 × 64 shots, ±2 shot SE")
    ax.set(xlabel="Site", ylabel="Population", xticks=x, title="72-step cosine schedule: populations on the new physical chain")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ("png", "pdf"): fig.savefig(output/f"populations.{ext}", dpi=180)
    plt.close(fig)
    sites = r["control_assessment"]["sites"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), layout="constrained")
    axes[0].bar(x-.18, [s["zero_population"] for s in sites], .36, label="All-zero error")
    axes[0].bar(x+.18, [1-s["one_population"] for s in sites], .36, label="All-one error")
    axes[0].set(ylabel="Observed error", title="Preparation/readout controls: 2048 shots each")
    axes[1].axhline(.5, color="k", linestyle="--", label="Intended half mixture")
    axes[1].plot(x, [s["predicted_mixture"] for s in sites], "s-", color="teal", label="Zero/one prediction")
    axes[1].errorbar(x, [s["mixture_population"] for s in sites],
        yerr=2*np.sqrt(np.array([s["mixture_population"]*(1-s["mixture_population"])/2047 for s in sites])),
        fmt="o-", color="firebrick", capsize=3, label="Measured mixture ±2 shot SE")
    axes[1].set(ylabel="Population", title="No Trotter evolution; all sites pass the specified diagnostic")
    for ax in axes: ax.set(xlabel="Site", xticks=x); ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ("png", "pdf"): fig.savefig(output/f"controls.{ext}", dpi=180)
    plt.close(fig)
    rows = [dict(site_1based=int(j+1), physical_qubit=r["physical_sites"][j], hardware=r["hardware_populations"][j],
                 ideal_same_paths=ideal[j], physical_stationary=physical[j], previous_hardware=r["previous_populations"][j],
                 shot_se_fixed_paths=r["shot_se_fixed_paths"][j], across_path_se=r["across_path_se"][j],
                 paired_residual_se=r["paired_residual_se"][j], residual_nominal_z=r["residual_nominal_z"][j]) for j in range(len(x))]
    write_csv(output/"populations.csv", rows)
    write_csv(output/"controls.csv", [{k: v for k, v in s.items() if k != "reasons"} for s in sites])
    lines = ["# Phoenix population analysis: calibration-selected chain", "",
        f"Controls: `{CONTROLS}` (3 circuits × 2048 shots). Full run: `{FULL}` (128 trajectories × 64 shots). Both jobs are fully collected.", "",
        "The preparation controls pass, and the full population profile improves substantially over the previous chain. The 0.02 steady-state population target is still unmet.", "",
        "![Final populations](populations.png)", "", "[Population PDF](populations.pdf) · [Population data](populations.csv) · [Control plot](controls.png)", "",
        "| Comparison | Previous chain MAE | New chain MAE | Previous maximum | New maximum |",
        "|---|---:|---:|---:|---:|"]
    for key, label in [("physical_stationary", "Physical stationary populations"), ("ideal_same_saved_paths", "Ideal, same 128 trajectories")]:
        old, new = r["previous_comparisons"][key], r["comparisons"][key]
        lines.append(f"| {label} | {old['mae']:.6f} | {new['mae']:.6f} | {old['maximum']:.6f} | {new['maximum']:.6f} |")
    lines += ["", "The schedule, 128 saved branch sequences and ideal final covariances match the previous 128 × 64 experiment. Execution date, physical layout and device conditions differ, so the improvement cannot be uniquely attributed to layout.", "",
        "| Site | Physical qubit | Hardware | Ideal same paths | Physical target | ±2 shot SE |",
        "|---:|---:|---:|---:|---:|---:|"]
    for j in range(len(x)):
        lines.append(f"| {j+1} | {r['physical_sites'][j]} | {r['hardware_populations'][j]:.6f} | {ideal[j]:.6f} | {physical[j]:.6f} | {2*r['shot_se_fixed_paths'][j]:.6f} |")
    lines += ["", f"Sites within 0.02 of physical stationary values: **{r['within_tolerance_sites_1based']}** (point estimates). The maximum same-path ideal population error against physical is **{r['ideal_vs_physical']['maximum']:.6f}**.", "",
        "The worst physical discrepancy is site 3: 0.549438 versus 0.500000. The worst matched-ideal discrepancy is site 2: 0.529785 versus 0.487263. Remaining discrepancies at several sites exceed nominal sampling uncertainty; more shots alone do not remove a systematic offset.", "",
        "The plotted bars are ±2 shot standard errors conditional on these fixed 128 paths. Across-path and paired hardware-minus-ideal standard errors are saved separately in the CSV/JSON; they are not added to these bars. Normal approximations assume independent samples and exclude device drift. One full job cannot estimate between-job uncertainty.", "",
        "## Preparation controls", "", "![Controls](controls.png)", "",
        f"All ten sites pass the existing diagnostic: baseline Wilson upper bounds below 0.03 and mixture residuals within the nominal family-adjusted cutoff {r['control_assessment']['nominal_z_cutoff']:.3f}.",
        f"Largest observed zero/one error: {max(max(s['zero_population'], 1-s['one_population']) for s in sites):.4%}. Largest absolute mixture residual: {max(abs(s['mixture_residual']) for s in sites):.6f}.",
        "The largest mixture residual is at site 7 (physical qubit 76): 0.472656 measured versus 0.496582 predicted, about 2.16 nominal SE, below the screening cutoff. Passing these short controls does not certify 0.02 accuracy, bath resets, or performance after 72 steps. Controls and evolution were separate jobs executed at different times.", "",
        "## Layout and circuit resources", "", f"Physical path, including the two endpoint baths: `{r['physical_path']}`.",
        f"Selected calibration worst CZ error: {100*r['path_metrics']['worst_cz_error']:.4f}%; minimum T1/T2 over the path: {1e6*r['path_metrics']['minimum_t1_seconds']:.1f}/{1e6*r['path_metrics']['minimum_t2_seconds']:.1f} µs.",
        f"Native circuit depth: {r['resources']['depth_range']}; CZ gates: {r['resources']['cz_counts']}; two-qubit depth: {r['resources']['two_qubit_depth']}; bath resets: {r['resources']['reset_counts']}. Circuit depth remains substantial despite better calibration.", "",
        "## Interpretation and next experiment", "",
        "The new chain is a useful improvement and its preparation controls are consistent with expectations. Healthy preparation with residual full-circuit disagreement points toward effects accumulated during evolution, but these data do not isolate gate errors, idle relaxation, reset errors, or time-dependent calibration.",
        "Next, use a small set of prefixes of the same saved cosine trajectories (for example 0, 12, 24, 48 and 72 steps), comparing each prefix to its own ideal prediction. An intermediate prefix is not expected to match the final steady state. This can identify where hardware disagreement grows before another large full-depth run. No hardware work was performed for this analysis.", "",
        "## Integrity checks and scope", "",
        "Reconstructed every final Z population and shot count from raw bitstrings, checked trajectory order, mappings, saved means, reference covariances, standard errors, job coverage and control assessment. Both bundles pass ZIP integrity checks; archived source and compiled ISA QPY hashes match their manifests. QPY files remain inside the bundles; no recompilation was performed. Git line-ending conversions in the checked-out JSON are accepted only when they reproduce the exact original source hash.",
        "Only final populations were measured. These results do not establish hardware stationarity or current accuracy. There was no simulator evolution, account access, job submission, automatic correction, or modification of original run files.", ""]
    (output/"Analysis.md").write_text("\n".join(lines))
    for folder, info in r["verification"]["inputs"].items():
        for name, digest in info["hashes"].items():
            assert hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest() == digest
    r["verification"]["original_files_unchanged"] = True
    (output/"analysis.json").write_text(json.dumps(serializable(r), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent/"hardware_runs")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).resolve().parent/"analysis"/"hardware"/FULL)
    args = parser.parse_args()
    if args.output_directory.resolve().is_relative_to(args.runs_root.resolve()):
        raise ValueError("output must be outside original hardware runs")
    report = analyze(args.runs_root)
    outputs(report, args.output_directory)
    print(json.dumps(serializable({"output": str(args.output_directory), "controls_passed": report["control_assessment"]["passed"],
                                  "comparisons": report["comparisons"], "residual_nominal_z": report["residual_nominal_z"]}), indent=2))


if __name__ == "__main__":
    main()
