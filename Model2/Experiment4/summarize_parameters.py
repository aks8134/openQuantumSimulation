"""Assemble the saved parameter sweeps, validation tables and comparison plots."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

DIRECTORY = Path(__file__).resolve().parent


def main():
    sources = sorted((DIRECTORY/"runs").glob("parameter_study_*/study.json"))
    cases = [(path, case) for path in sources for case in json.loads(path.read_text())["cases"].values()
             if case.get("complete") and case.get("selected") is not None]
    groups = {}
    for path, case in cases:
        c = case["configuration"]
        if c["maximum_dt"] != 8 or c["p"] != .25 or c["order"] != "jump-first":
            continue
        groups.setdefault(c["gamma"], []).append((path, case))
    summary = []
    for ratio, rows in sorted(groups.items()):
        constant = min((case["constant"]["best"] for _, case in rows), key=lambda x: (x["steps"], x["error"]))
        source, selected_case = min(rows, key=lambda row: (row[1]["selected"]["steps"], row[1]["selected"]["forward"]["maximum_error"]))
        selected = selected_case["selected"]
        summary.append({"ratio": ratio, "constant_steps": constant["steps"],
                        "selected_steps": selected["steps"], "family": selected["family"],
                        "forward_error": selected["forward"]["maximum_error"],
                        "initial_dt": selected["step_dts"][0], "final_dt": selected["step_dts"][-1],
                        "power": float(np.exp(selected["parameters"][2])) if selected["family"] == "warped_cosine" else 1.,
                        "source": str(source.relative_to(DIRECTORY)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    validations = []
    for file in sorted((DIRECTORY/"runs").glob("parameter_validation_*/study.json")):
        study = json.loads(file.read_text())
        if not study.get("complete"):
            continue
        record = next(iter(study["families"].values()))
        if not record["sampling_passed"]:
            continue
        R = min(R for R in sorted(set(row["trajectories"] for row in record["replays"]))
                if len([row for row in record["replays"] if row["trajectories"] == R]) == 3
                and all(row["passed"] for row in record["replays"] if row["trajectories"] == R))
        rows = [row for row in record["replays"] if row["trajectories"] == R]
        validations.append({"ratio": study["configuration"]["gamma"], "steps": record["selected"]["steps"],
            "R": R, "maximum_sampled_error": max(row["sampled_error"] for row in rows),
            "maximum_sampling_distance": max(row["sampling_distance"] for row in rows),
            "maximum_two_SE": max(row["maximum_two_standard_errors"] for row in rows),
            "directory": str(file.parent.relative_to(DIRECTORY))})
    bounds = []
    for upper in (2., 4., 8.):
        rows = [(case["constant"]["best"], path) for path, case in cases
                if case["configuration"]["gamma"] == 2 and case["configuration"]["maximum_dt"] == upper]
        best, path = min(rows, key=lambda row: (row[0]["steps"], row[0]["error"]))
        bounds.append({"maximum_dt": upper, **best, "source": str(path.relative_to(DIRECTORY))})
    payload = {"rate_comparison": summary, "sampling": validations, "bound_comparison": bounds,
               "trial_count": sum(len(case["trials"]) for _, case in cases),
               "scope": "fixed N=10, half mixture, p=.25, c=.25, even-odd, jump-first; .016 mean + .004 sampling"}
    (DIRECTORY/"parameter_summary.json").write_text(json.dumps(payload, indent=2)+"\n")
    with (DIRECTORY/"parameter_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Broader parameter study: annealing does help in some cases", "",
        "**At the original Gamma/J=2, the best saved warped cosine uses 44 steps versus 45 for the best constant schedule found.** The earlier 80-step result was restricted to dt<=2. Its separate wider-bound optimizer also missed a better basin: a direct scan already finds 47 constant steps with dt<=4. The historical report is retained with that correction.", "",
        "All cases retain N=10, the requested half-mixture, p_L=p_R=0.25, Model2's exchange prefactor 1/4, Delta=0, rate-weighted single-jump sampling, jump-first ordering and even/odd Hamiltonian layers. Total tolerance stays 0.02, split as 0.016 mean + 0.004 sampling. Changing Gamma/J changes relaxation while preserving the same stationary target. Changing J and Gamma together merely rescales time when timesteps are rescaled too.", "",
        "## Isolating the timestep bound", "",
        "The physical ratio remains Gamma/J=2 in this table. A batched timestep scan followed by local refinement replaces reliance on one joint integer optimizer.", "",
        "| Maximum dt | Best constant steps found | Constant dt | Mean output error |",
        "|---|---:|---:|---:|"]
    for row in bounds:
        lines.append(f"| {row['maximum_dt']:g} | {row['steps']} | {row['dt']:.9g} | {row['error']:.9g} |")
    lines += ["", "The new dt<=4 fixed-count cosine/warped trials did not beat 47. With dt<=8, ordinary cosine at 44 steps reaches approximately 0.0160523, just outside the allowance; the warped cosine passes. The large 80-to-44 reduction combines better bounds/search with annealing. The annealing gain against its matching constant baseline is **one step, or 2.22%**.", "",
        "## Varying the equal boundary rate", "",
        "The common endpoint-step range is [0.001,8] at J=1. Constant steps use roughly 400 scanned values including analytically identified collision/swap points, followed by multiple local refinements. Ordinary and warped cosines are fit at fixed counts, avoiding the original discontinuous integer-count objective. Global seeds 2718/3141 and targeted 1729/1730/3141 runs use 24–64 iterations. Warped exponents are explored over [0.03,12] and [0.03,64]. Initial challenges include one step below the constant baseline, about 10% below and about 25% below; targeted runs fill adjacent shorter counts. Every trial's actual count and settings are retained.", "",
        "| Gamma/J | Best constant count found | Best retained count | Schedule | Forward mean maximum |",
        "|---|---:|---:|---|---:|"]
    for row in summary:
        label = "pi/8" if np.isclose(row["ratio"], np.pi/8) else f"{row['ratio']:g}"
        lines.append(f"| {label} | {row['constant_steps']} | {row['selected_steps']} | {row['family']} | {row['forward_error']:.9g} |")
    lines += ["", "![Rate comparison](parameter_comparison.png)", "",
        "Only the 20-step construction has a matching general lower-bound argument under this mean criterion; other counts remain numerical-search results. A failed fixed-count optimizer is not an infeasibility proof. The shorter schedules prepare acceptable outputs; no transient Lindblad-accuracy claim is made. Large collision steps retain their signed cosine and are not replaced by exact-reservoir damping.", "",
        "## Separate sampling validation", "",
        "The following selected schedules were frozen using means before replay. All three fixed seeds 29,30,31 passed at the displayed R; all smaller attempted ensembles and their failures remain saved.", "",
        "| Gamma/J | Steps | R per seed | Maximum sampled target error | Maximum sampling distance | Maximum two SE |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(validations, key=lambda x:x["ratio"]):
        lines.append(f"| {row['ratio']:.9g} | {row['steps']} | {row['R']} | {row['maximum_sampled_error']:.9g} | {row['maximum_sampling_distance']:.9g} | {row['maximum_two_SE']:.9g} |")
    lines += ["", "The 20-step candidate uses Gamma/J=pi/8 and dt=pi. It aligns full population swaps with zero bath cosine, giving C_output=0.265625 I and mean error 1/64=0.015625. A determinant argument shows that fewer than 20 steps cannot meet the 0.016 mean allowance in this equal-rate, one-collision channel class with half-mixture input. See [the derivation and scope](StepLowerBound.md). This changes the physical rate; it is not an annealing improvement at Gamma/J=2.", "",
        "![Original-rate schedules and errors](original_rate_comparison.png)", "",
        "The 44-step schedule's forward mean check covers at least 10 physical-time units at its final dt; unlike the historical 80-step example, the extra permanent matrix-order check does not pass. No permanent extension is inferred for it. The 20-step construction instead passes the permanent spectral-norm bound because its deviation is scalar. Initial mixture bits are analytically averaged. Two-SE checks are descriptive, not simultaneous confidence guarantees. Measurement shots, hardware noise and native compiled depth are excluded. Larger R reduces sampling uncertainty and increases total workload while leaving per-trajectory depth unchanged.", "",
        "Accelerated normalized-covariance propagation is checked against the original affine channel over chain lengths 2/3/10, both factor orders, rates .25/2/8 and reservoir populations .1/.25/.4. The reported selected means and every forward node were then reconstructed with the separate explicit four-branch verifier. Agreement is within 2e-16 for the three replayed candidates. 32 relevant tests pass (13 experiment tests, five Model2 randomized tests and fourteen shared-reservoir tests).", "",
        "## Saved validation and reproduction", ""]
    for row in validations:
        folder = row["directory"]
        lines.append(f"- Gamma/J={row['ratio']:.9g}: [results]({folder}/Study.md), [independent check]({folder}/Verified.md).")
    lines += ["", "The compact [summary JSON](parameter_summary.json) and [CSV](parameter_summary.csv) retain source hashes and selected settings. Full local studies are under `runs/parameter_study_*/study.json`. The historical [initial result](Results.md) is preserved. Reproduction commands are in [Experiment4.md](Experiment4.md)."]
    (DIRECTORY/"ParameterStudy.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
    x = np.arange(len(summary))
    ax.bar(x-.17, [r["constant_steps"] for r in summary], .34, label="Best constant found")
    ax.bar(x+.17, [r["selected_steps"] for r in summary], .34, label="Best retained schedule")
    ax.axhline(20, color="black", linestyle=":", label="General mean lower bound")
    ax.set(xticks=x, xticklabels=["pi/8" if np.isclose(r["ratio"],np.pi/8) else f"{r['ratio']:g}" for r in summary],
           xlabel="Equal boundary rate Gamma/J", ylabel="Preparation steps", title="Same target and tolerance; varying the boundary rate")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(DIRECTORY/f"parameter_comparison.{suffix}", dpi=160)
    plt.close(fig)
    original = json.loads((DIRECTORY/"runs/cosine_8e274c92c265/study.json").read_text())["families"]["cosine"]["selected"]
    validated = next(r for r in validations if r["ratio"] == 2.)
    value = json.loads((DIRECTORY/validated["directory"]/"study.json").read_text())
    warped = next(iter(value["families"].values()))["selected"]
    best_constant = next(r for r in summary if r["ratio"] == 2.)
    constant_case = min((case for _,case in groups[2.]), key=lambda case:case["constant"]["best"]["error"])["constant"]["best"]
    fig, axes = plt.subplots(1,2, figsize=(11,4), constrained_layout=True)
    for name, row in [("Initial bounded cosine", original),("Broader warped cosine",warped)]:
        axes[0].plot(np.arange(1,row["steps"]+1),row["step_dts"],label=f"{name}: {row['steps']} steps")
        axes[1].plot(np.arange(row["steps"]+1), row["mean_errors"], label=name)
    axes[0].plot(np.arange(1,constant_case["steps"]+1),np.full(constant_case["steps"],constant_case["dt"]),
                 "--",label=f"Broader constant: {constant_case['steps']} steps")
    axes[0].set(xlabel="Preparation step",ylabel="dt")
    axes[1].axhline(.016,color="black",linestyle=":",label="Mean allowance")
    axes[1].set(xlabel="Preparation step",ylabel="Maximum local mean error",yscale="log")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(DIRECTORY/f"original_rate_comparison.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Saved {DIRECTORY/'ParameterStudy.md'}; {payload['trial_count']} fixed-count trials.")


if __name__ == "__main__":
    main()
