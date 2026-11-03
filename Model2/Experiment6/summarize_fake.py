"""Export portable tables, figures and a report from completed local Aer counts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
LABELS = {"raw": "Raw", "raw_spam_corrected": "Raw + SPAM",
          "twirled": "Twirled", "twirled_spam_corrected": "Twirled + SPAM",
          "zne": "Linear ZNE", "zne_spam_corrected": "Linear ZNE + SPAM"}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY / "results")
    options = parser.parse_args(arguments)
    data = json.loads(options.analysis.read_text())
    if data["experiment"] != "Model2/Experiment6/FakeKingston_ZNE":
        raise ValueError("expected the completed Model2 FakeKingston analysis")
    cfg, stats = data["configuration"], data["statistics"]
    profiles = stats["profiles"]
    output = options.output_directory
    output.mkdir(parents=True, exist_ok=True)
    metrics = [dict(method=name, label=LABELS[name], **{k: row[k] for k in
               ("ideal_mae", "ideal_maximum", "physical_mae", "physical_maximum")},
               maximum_fixed_path_se=max(row["fixed_path_delta_method_se"])) for name, row in profiles.items()]
    write_csv(output / "metrics.csv", metrics)
    ideal, target = np.array(data["ideal_same_paths"]), np.array(data["physical_target"])
    population_rows = []
    for site in range(len(ideal)):
        row = dict(site=site, physical_qubit=cfg["active_compact_to_physical"][site], ideal_same_paths=ideal[site], target=target[site])
        for name, profile in profiles.items():
            row[name] = profile["populations"][site]
            row[name + "_fixed_path_se"] = profile["fixed_path_delta_method_se"][site]
            row[name + "_bootstrap_lower"] = profile["bootstrap_ci95"][0][site]
            row[name + "_bootstrap_upper"] = profile["bootstrap_ci95"][1][site]
        population_rows.append(row)
    write_csv(output / "populations.csv", population_rows)
    scale = stats["linear_zne"]
    write_csv(output / "noise_scales.csv", [dict(site=site, factor=factor,
        population=scale["raw_scale_populations"][k][site],
        fixed_path_se=scale["raw_scale_fixed_path_se"][k][site],
        linear_fit=scale["linear_fit_populations"][k][site])
        for k, factor in enumerate(scale["factors"]) for site in range(len(ideal))])
    write_csv(output / "comparisons.csv", [dict(method=row["method"], reference=row["reference"],
        metric=row["metric"], difference=row["difference"], bootstrap_lower=row["bootstrap_ci95"][0],
        bootstrap_upper=row["bootstrap_ci95"][1]) for row in stats["comparisons"]])
    resource_groups = {}
    for path in options.analysis.parent.glob("trajectory_*.json"):
        for row in json.loads(path.read_text())["records"]:
            arm = "raw" if row["arm"] == "raw" else f"twirled factor {row['actual_factor']:g}"
            resource_groups.setdefault(arm, []).append(row["metrics"])
    resources = []
    for arm, group in sorted(resource_groups.items()):
        resources.append(dict(arm=arm, circuits=len(group),
            **{name:float(np.mean([row["operation_counts"].get(name, 0) for row in group]))
               for name in ("cz", "sx", "x", "reset")},
            mean_depth=float(np.mean([row["depth"] for row in group]))))
    if resources:
        write_csv(output / "resources.csv", resources)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    sites = np.arange(len(ideal))
    for name, color, shift in [("raw", "#60656b", -.12), ("twirled", "#2171b5", -.04),
                               ("zne", "#cb642c", .04), ("zne_spam_corrected", "#7a5195", .12)]:
        row = profiles[name]
        axes[0].errorbar(sites + shift, row["populations"], yerr=row["fixed_path_delta_method_se"],
                         marker="o", linestyle="-", color=color, capsize=2, label=LABELS[name], alpha=.9)
    axes[0].plot(sites, ideal, "k--", linewidth=1.6, label="Exact: same reservoir paths")
    axes[0].axhline(.25, color="#aaaaaa", linestyle=":", label="Stationary target")
    axes[0].set(xlabel="System site (0-based)", ylabel="Excitation population", xticks=sites,
                title="Model2 cosine44 on FakeKingston — bars show ±1 fixed-path shot SE")
    axes[0].legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.18))
    errors = np.array([np.array(profiles[name]["populations"]) - ideal for name in LABELS])
    limit = np.max(abs(errors))
    mesh = axes[1].imshow(errors, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set(xticks=sites, xlabel="System site (0-based)", title="Signed population error against the exact matched paths")
    axes[1].set_yticks(np.arange(len(LABELS)), labels=list(LABELS.values()))
    axes[1].tick_params(axis="y", labelleft=True)
    fig.colorbar(mesh, ax=axes[1], label="Measured − exact")
    fig.canvas.draw()
    fig.savefig(output / "populations.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "populations.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, panels = plt.subplots(2, 5, figsize=(13, 6), sharex=True, constrained_layout=True)
    for site, panel in enumerate(panels.flat):
        factors = np.array(scale["factors"])
        observed = np.array(scale["raw_scale_populations"])[:, site]
        se = np.array(scale["raw_scale_fixed_path_se"])[:, site]
        intercept = profiles["zne"]["populations"][site]
        slope = (scale["linear_fit_populations"][-1][site] - scale["linear_fit_populations"][0][site]) / (factors[-1] - factors[0])
        x = np.array([0., factors[-1]])
        panel.plot(x, intercept + slope*x, color="#cb642c", label="Linear CZ extrapolation")
        panel.errorbar(factors, observed, yerr=se, fmt="o", color="#2171b5", capsize=2, label="Twirled ±1 SE")
        panel.axhline(ideal[site], color="black", linestyle="--", label="Exact matched ideal")
        panel.set(title=f"Site {site} · physical q{cfg['active_compact_to_physical'][site]}")
        panel.set_xticks([0, 1, 1.25, 1.5], labels=["0", "1", "1.25", "1.5"], fontsize=8)
        if site >= 5:
            panel.set_xlabel("CZ count factor")
        if site % 5 == 0:
            panel.set_ylabel("Excitation population")
        if site == 0:
            panel.legend(fontsize=7)
    fig.suptitle("Model2 / FakeKingston — fixed linear extrapolation at each site")
    fig.savefig(output / "noise_scaling.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "noise_scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Model2: FakeKingston mitigation pilot", "",
        f"Completed actual noisy Aer simulations for **{cfg['trajectory_count']} matched reservoir paths × {cfg['shots']} shots per path per arm/factor**, "
        f"with {cfg['randomizations']} twirl variants ({cfg['shots']//cfg['randomizations']} shots each). "
        f"The run contains **{cfg['total_circuits']} circuits and {cfg['total_shots']:,} actual shots**, including controls. "
        "This is a population-only pilot, not the proposed 16,384-path hardware experiment.", "",
        "## Result", "", "All errors below are absolute excitation-population errors. The matched noiseless reference averages the "
        "same reservoir paths, analytically averaging the half-mixed preparation; it isolates circuit noise and shots from finite path sampling.", "",
        "| Method | Mean error vs matched ideal | Max error vs matched ideal | Max error vs stationary 0.25 |",
        "| --- | ---: | ---: | ---: |"]
    for name, row in profiles.items():
        lines.append(f"| {LABELS[name]} | {row['ideal_mae']:.6f} | {row['ideal_maximum']:.6f} | {row['physical_maximum']:.6f} |")
    lines += ["", "### Paired comparisons", "", "Differences are method error minus reference error; negative favors the method. "
              f"Intervals are descriptive paired trajectory bootstrap intervals ({cfg['bootstrap']:,} replicates), "
              "including independently resampled preparation-control counts for corrected methods. They are not simultaneous confidence bounds.", "",
              "| Method vs reference | Mean-error difference | Bootstrap 95% interval |", "| --- | ---: | --- |"]
    for row in stats["comparisons"]:
        if row["metric"] != "ideal_mae":
            continue
        lo, hi = row["bootstrap_ci95"]
        lines.append(f"| {LABELS[row['method']]} vs {LABELS[row['reference']]} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    for reference in ("twirled", "raw"):
        comparison = next(row for row in stats["comparisons"] if row["method"] == "zne"
                          and row["reference"] == reference and row["metric"] == "ideal_mae")
        lo, hi = comparison["bootstrap_ci95"]
        if hi < 0:
            conclusion = "has lower mean absolute error in this pilot, with the descriptive interval below zero"
        elif lo > 0:
            conclusion = "has higher mean absolute error in this pilot, with the descriptive interval above zero"
        else:
            conclusion = "has no resolved mean-error advantage in this pilot; the descriptive interval spans zero"
        lines += ["", f"Linear ZNE compared with {LABELS[reference]} **{conclusion}**."]
    max_ideal_error = max(abs(ideal - target))
    lines += ["", f"Even the noiseless mean of these {cfg['trajectory_count']} paths has maximum stationary-target error **{max_ideal_error:.6f}**. "
              "This sample size does not establish the 0.02 total observable criterion. Currents are not measured here.", "",
              "![Population comparison](populations.png)", "", "![CZ noise scaling](noise_scaling.png)", "",
              "The dashed curves in the scaling plots are the fully noiseless matched reference. The extrapolation removes "
              "only noise that scales with the folded CZ circuit; its factor-zero intercept can retain unscaled noise.", "",
              "## Frozen experiment and mitigation", "",
              "N=10, J=1, Γ/J=2, pL=pR=0.25, Δ=0, exchange prefactor 1/4; half-mixed initial state. "
              "The original 44-step warped cosine is unchanged: initial dt=4.207281841881579, final dt=3.760760459292396, "
              "power=16.424973388132297; jump-first evolution, then even/odd exchange layers.", "",
              "The exact Model3 manual mitigation helpers are reused. Each native CZ is Pauli twirled. Selective CZ→CZ·CZ·CZ folding "
              "uses nested seeded subsets and factors 1, 1.25, 1.5 (880, 1,100, 1,320 CZ). Only one-qubit gates are optimized afterward; "
              "all 44 reservoir resets and preparation measurements are retained. Complementary terminal masks balance readout flips. "
              "No general transpilation follows the folding. No target fitting, clipping, dynamical decoupling, or automatic Runtime mitigation is used.", "",
              "| Arm | Mean CZ | Mean SX | Mean X | Mean resets | Mean depth |",
              "| --- | ---: | ---: | ---: | ---: | ---: |",
              *[f"| {row['arm']} | {row['cz']:.0f} | {row['sx']:.1f} | {row['x']:.1f} | {row['reset']:.0f} | {row['mean_depth']:.1f} |"
                for row in resources], "",
              "These counts show the remaining one-qubit gates and the extra X frames. CZ count factors do not uniformly scale "
              "every physical noise channel or the complete circuit duration.", "",
              f"The unweighted linear extrapolation weights are {np.round(scale['weights'], 8).tolist()}; "
              f"the shot-SE amplification is {scale['intercept_shot_se_amplification']:.6f}× relative to one equally sampled scale. "
              "ZNE consumes three scales; its result is not a shot-cost-matched comparison with raw or twirled alone.", "",
              f"Zero, one and held-out half-mixture controls each receive {cfg['control_shots']:,} shots for both raw and twirled readout. "
              "The four twirled variants split that budget. Exploratory marginal SPAM correction uses (population−zero)/(one−zero), "
              "requires contrast ≥0.5, and includes shared-control uncertainty. The half-mixture is held out rather than fitted.", "",
              "| Control treatment | Maximum corrected half-mixture deviation from 0.5 |",
              "| --- | ---: |",
              *[f"| {arm} | {np.max(abs(np.array(row['corrected'])-.5)):.6f} |"
                for arm, row in stats["heldout_mixture"].items()], "",
              "## Snapshot and scope", "",
              f"FakeKingston calibration date: `{cfg.get('snapshot_date', 'not recorded')}`. Physical chain, baths at the ends: "
              f"`{cfg['physical_path']}`. This mapping was fixed before examining noisy mitigation results. "
              "The 12 active physical qubits are compactly relabeled for simulation, preserving their T1/T2/frequency, "
              "gate durations and errors, connectivity and readout errors. The saved noise-model hash identifies the exact model.", "",
              "Aer uses the snapshot's standard local gate noise, gate-duration thermal relaxation and readout noise. "
              "These unscheduled circuits do not include additional system idling through bath resets or dynamic-circuit latency, "
              "nor hardware drift/crosstalk. CZ-only ZNE leaves preparation, reset and unscaled error channels present. "
              "The fake noise model is not a prediction of the current physical Kingston device.", "",
              "Bootstrap intervals condition on the sampled variants and the static snapshot. Shot covariance and fixed-path delta-method "
              "SEs are also exported; the figure uses those SEs. Small-path bootstrap uncertainty and nonlinear absolute-error bias "
              "limit interpretation. There is no live IBM connection or hardware submission.", "", "## Reproduce", "", "From the repository root:", "", "```bash",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/model2_fake_mpl \\",
              "  .venv/bin/python Model2/Experiment6/fake_kingston_zne.py --paths 32 --shots 128 \\",
              "  --control-shots 2048 --randomizations 4 --seed 29 --sim-seed 2026 --resume", "```", "",
              f"Local raw run: `{options.analysis.parent.resolve()}`. Saved artifacts include raw outcome counts, native QPY variants, "
              "source trajectories, complete noise model, seeds, hashes, package versions and per-group Aer metadata. "
              "`--resume` checks snapshot, source, implementation and saved-circuit hashes before using cached units.", "",
              "To regenerate these tables and figures:", "", "```bash",
              f".venv/bin/python Model2/Experiment6/summarize_fake.py {options.analysis}", "```", "",
              "Independent raw-count audit (no Aer replay):", "", "```bash",
              f".venv/bin/python Model2/Experiment6/verify_run.py {options.analysis.parent}", "```", "",
              f"Sum of timed Aer calls: **{data['simulation_seconds']/60:.2f} minutes** on {cfg['threads']} threads "
              "(excludes compilation, imports, analysis and interrupted attempts).", "",
              f"Packages: `{cfg['versions']}`.", "", f"Noise model SHA256: `{cfg['noise_model_sha256']}`.", ""]
    (output / "Results.md").write_text("\n".join(lines))
    print(output / "Results.md")


if __name__ == "__main__":
    main()
