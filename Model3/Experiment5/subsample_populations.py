"""Subsample saved population measurements; no hardware access or evolution.

Each replicate selects S whole bitstrings without replacement independently
within each fixed trajectory. Shot levels are nested within a replicate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_hardware import analyze, serializable
except ImportError:
    from analyze_hardware import analyze, serializable


def expand_counts(counts, n):
    if not counts or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("counts must be nonnegative integers")
    bits, weights = [], []
    for key, count in sorted(counts.items()):
        key = key.replace(" ", "")
        if len(key) != n or set(key) - {"0", "1"}:
            raise ValueError("invalid final-register bitstring")
        bits.append([int(bit) for bit in key[::-1]])
        weights.append(count)
    return np.repeat(np.asarray(bits, dtype=np.uint8), weights, axis=0)


def subsample(pools, shot_levels, repeats, seed, batch_size=32):
    """Return [replicate, shot level, trajectory, site] sample means.

    Independent row permutations give uniform subsets, preserving all site
    correlations in each measured bitstring. Randomness is shared across shot
    levels only to make comparisons within a replicate less noisy.
    """
    pools = np.asarray(pools)
    if pools.ndim != 3 or 0 in pools.shape or not np.isin(pools, [0, 1]).all():
        raise ValueError("pools must have shape [trajectory, shot, site] of bits")
    paths, total, sites = pools.shape
    levels = np.asarray(shot_levels)
    if (levels.ndim != 1 or not len(levels) or levels.dtype.kind not in "iu"
            or np.any(levels < 1) or np.any(levels > total)
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("shot levels must be increasing integers between 1 and pool size")
    if repeats < 2 or batch_size < 1 or total < 2:
        raise ValueError("need at least two replicates/shots and a positive batch size")
    rng = np.random.default_rng(seed)
    result = np.empty((repeats, len(levels), paths, sites))
    full = pools.mean(axis=1)
    partial = np.flatnonzero(levels < total)
    for path, pool in enumerate(pools):
        for start in range(0, repeats, batch_size):
            stop = min(start + batch_size, repeats)
            if len(partial):
                order = rng.permuted(np.broadcast_to(np.arange(total), (stop-start, total)), axis=1)
                cumulative = np.cumsum(pool[order[:, :levels[partial].max()]], axis=1, dtype=np.int32)
                result[start:stop, partial, path] = cumulative[:, levels[partial]-1]/levels[partial, None]
            if levels[-1] == total:
                result[start:stop, -1, path] = full[path]
    return result


def finite_population_covariance(pools, shot_levels):
    """Exact conditional covariance of the equal-weight path-averaged mean."""
    paths, total, _ = pools.shape
    covariance = sum(np.cov(pool.astype(float), rowvar=False, ddof=1)
                     for pool in pools)/paths**2
    levels = np.asarray(shot_levels, dtype=float)
    return ((1-levels/total)/levels)[:, None, None]*covariance


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(draws, pools, levels, references):
    full_paths = pools.mean(axis=1)
    full = full_paths.mean(axis=0)
    pooled = draws.mean(axis=2)
    covariance = finite_population_covariance(pools, levels)
    rows, per_site, per_path = [], [], []
    for k, shots in enumerate(levels):
        delta = pooled[:, k]-full
        maximum = np.abs(delta).max(axis=1)
        mae = np.abs(delta).mean(axis=1)
        rmse = np.sqrt(np.mean(delta**2, axis=1))
        row = {"shots_per_path": shots, "total_z_shots": shots*len(pools),
               "median_max_deviation_from_full": float(np.median(maximum)),
               "q95_max_deviation_from_full": float(np.quantile(maximum, .95)),
               "mean_absolute_deviation_from_full": float(mae.mean()),
               "mean_rmse_from_full": float(rmse.mean()),
               "fraction_all_sites_within_0.02_of_full": float(np.mean(maximum <= .02)),
               "max_exact_conditional_se": float(np.sqrt(np.diag(covariance[k])).max())}
        for name, target in references.items():
            errors = np.abs(pooled[:, k]-target)
            for label, values in (("mae", errors.mean(axis=1)), ("max", errors.max(axis=1))):
                for qname, q in (("q025", .025), ("median", .5), ("q975", .975)):
                    row[f"{label}_vs_{name}_{qname}"] = float(np.quantile(values, q))
        rows.append(row)
        for site in range(len(full)):
            values = pooled[:, k, site]
            per_site.append({"shots_per_path": shots, "site_1based": site+1,
                "full_shot_population": float(full[site]), "subsample_mean": float(values.mean()),
                "q025": float(np.quantile(values, .025)), "median": float(np.median(values)),
                "q975": float(np.quantile(values, .975)), "empirical_sd": float(values.std(ddof=1)),
                "exact_conditional_sd": float(np.sqrt(covariance[k, site, site]))})
            for path in range(len(pools)):
                values = draws[:, k, path, site]
                per_path.append({"shots_per_path": shots, "path_position": path,
                    "site_1based": site+1, "full_shot_population": float(full_paths[path, site]),
                    "subsample_mean": float(values.mean()), "q025": float(np.quantile(values, .025)),
                    "q975": float(np.quantile(values, .975))})
    return rows, per_site, per_path, covariance


def plot(draws, levels, full, references, comparison, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = draws.mean(axis=2)
    sites = np.arange(1, len(full)+1)
    chosen = [s for s in (1, 16, 128, 2048) if s in levels]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.7*len(chosen)), layout="constrained")
    for axis, shots in zip(np.atleast_1d(axes), chosen):
        k = levels.index(shots)
        low, high = np.quantile(pooled[:, k], [.025, .975], axis=0)
        axis.fill_between(sites, low, high, color="#e69f00", alpha=.26,
                          label="Central 95% of subsamples (per site)")
        axis.plot(sites, full, "o-", color="#1565c0", label="Earlier hardware: full 8192 shots/path")
        axis.plot(sites, references["matched_ideal"], "s--", color="#00897b", label="Ideal: same four paths")
        axis.plot(sites, references["physical"], "k:", label="Physical steady-state target")
        if comparison is not None:
            axis.plot(sites, comparison["populations"], "d-.", color="#a53683", markersize=4,
                      label="Latest hardware: 8192 paths × 1 shot")
        axis.set(title=f"{draws.shape[2]} fixed paths × {shots} shots/path ({draws.shape[2]*shots:,} Z shots)",
                 xlabel="Site", ylabel="Population", xticks=sites)
        axis.grid(alpha=.2)
        if shots == 1:
            axis.set_ylim(-.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"population_subsamples.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    partial = [row for row in rows if row["shots_per_path"] < levels[-1]]
    x = [row["shots_per_path"] for row in partial]
    axes[0].plot(x, [row["median_max_deviation_from_full"] for row in partial], "o-", label="Median")
    axes[0].plot(x, [row["q95_max_deviation_from_full"] for row in partial], "s-", label="95th percentile")
    axes[0].axhline(.02, color="gray", linestyle=":", label="0.02 deviation")
    axes[0].set(xscale="log", yscale="log", xlabel="Shots per fixed trajectory",
                ylabel="Maximum population deviation from full-shot estimate")
    axes[0].legend(fontsize=8)
    for name, color in (("physical", "#333333"), ("matched_ideal", "#00897b")):
        axes[1].plot(x, [r[f"mae_vs_{name}_median"] for r in partial], "o-", color=color,
                     label=f"Subsample MAE vs {name.replace('_', ' ')}")
        axes[1].fill_between(x, [r[f"mae_vs_{name}_q025"] for r in partial],
                            [r[f"mae_vs_{name}_q975"] for r in partial], color=color, alpha=.12)
        axes[1].axhline(np.abs(full-references[name]).mean(), color=color, linestyle=":")
    if comparison is not None:
        axes[1].axhline(np.abs(comparison["populations"]-references["physical"]).mean(),
                        color="#a53683", linestyle="--", label="Latest hardware MAE vs physical")
    axes[1].set(xscale="log", yscale="log", xlabel="Shots per fixed trajectory",
                ylabel="Mean absolute population error across sites")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2, which="both")
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"shot_dependence.{suffix}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--compare-analysis", type=Path,
                        help="Optional existing one-shot analysis.json, read without modification")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()
    report, measured, _ = analyze(args.run_directory)
    n = report["physics"]["n_qubits"]
    manifest = json.loads((args.run_directory/"manifest.json").read_bytes())
    observations = [row for batch in manifest["batches"] for row in batch["observations"]]
    pools_by_path = {record["trajectory"]: expand_counts(row["outcome_counts"], n)
                     for record, row in zip(manifest["circuits"], observations) if record["basis"] == "z"}
    pools = np.stack([pools_by_path[path] for path in measured["indices"]])
    np.testing.assert_allclose(pools.mean(axis=1), measured["populations"], atol=1e-14)
    references = {"physical": report["references"]["physical_stationary"]["populations"],
                  "matched_ideal": report["references"]["ideal_same_saved_paths"]["populations"]}
    comparison = None
    hashes = {str(args.run_directory/name): digest
              for name, digest in report["verification"]["original_files_sha256"].items()}
    if args.compare_analysis:
        raw = args.compare_analysis.read_bytes()
        hashes[str(args.compare_analysis)] = hashlib.sha256(raw).hexdigest()
        other = json.loads(raw)
        np.testing.assert_allclose(other["references"]["physical_stationary"]["populations"], references["physical"])
        comparison = {"run_directory": other["run_directory"], "paths": other["paths"],
                      "shots_per_circuit": other["shots_per_circuit"],
                      "populations": np.asarray(other["hardware"]["populations"])}
    output = args.output_directory or Path(__file__).resolve().parent/"analysis"/"hardware"/args.run_directory.name/"population_subsampling"
    if output.resolve() == args.run_directory.resolve() or args.run_directory.resolve() in output.resolve().parents:
        raise ValueError("output must be outside the original run directory")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Subsampling {len(pools)} paths × {len(pools[0])} saved Z shots; {args.repeats} replicates", flush=True)
    draws = subsample(pools, args.shots, args.repeats, args.seed)
    rows, site_rows, path_rows, covariance = summarize(draws, pools, args.shots, references)
    full = pools.mean(axis=(0, 1))
    summaries = {name: {"mae": float(np.abs(full-target).mean()), "max": float(np.abs(full-target).max())}
                 for name, target in references.items()}
    mc_cov = np.stack([np.cov(draws[:, k].mean(axis=1), rowvar=False, ddof=1) for k in range(len(args.shots))])
    payload = {"seed": args.seed, "repeats": args.repeats, "path_indices": measured["indices"],
               "pool_shots_per_path": len(pools[0]), "shot_levels": args.shots,
               "sampling": "uniform whole-bitstring subsets without replacement, fixed paths; nested shot levels",
               "full_shot_populations": full, "full_shot_per_path_populations": pools.mean(axis=1),
               "references": references, "comparison": comparison, "full_shot_errors": summaries,
               "summary": rows, "conditional_covariance_exact": covariance,
               "conditional_covariance_monte_carlo": mc_cov, "input_sha256": hashes,
               "limitations": ["Conditions on the observed counts and four selected paths.",
                   "Quantile bands are resampling ranges, not confidence intervals for ideal physics.",
                   "The finite-pool correction makes variability vanish when all saved shots are selected.",
                   "Does not reproduce the 8192-distinct-path experiment or identify hardware drift/noise.",
                   "Shot ordering is unavailable; time correlations cannot be assessed."]}
    variance_mask = np.diagonal(covariance, axis1=1, axis2=2) > 0
    variance_ratio = (np.diagonal(mc_cov, axis1=1, axis2=2)[variance_mask]
                      /np.diagonal(covariance, axis1=1, axis2=2)[variance_mask])
    payload["max_relative_monte_carlo_variance_discrepancy"] = float(np.abs(variance_ratio-1).max()) if variance_ratio.size else 0.
    if comparison is not None:
        payload["comparison_population_errors"] = {
            "mae_vs_physical": float(np.abs(comparison["populations"]-references["physical"]).mean()),
            "max_vs_physical": float(np.abs(comparison["populations"]-references["physical"]).max())}
    write_csv(output/"summary.csv", rows)
    write_csv(output/"population_quantiles.csv", site_rows)
    write_csv(output/"per_path_population_quantiles.csv", path_rows)
    plot(draws, args.shots, full, references, comparison, rows, output)
    text = ["# Population shot-subsampling analysis", "",
        f"Source: `{args.run_directory.name}`. Fixed trajectory indices: {measured['indices']}.",
        f"{args.repeats:,} replicates per shot level; seed {args.seed}. Only saved Z-basis counts were used.", "",
        "Each replicate independently samples whole measured bitstrings without replacement within each path, then averages the four path populations equally. Site correlations are preserved. The shot levels share nested subsets within a replicate.", "",
        "## Actual populations", "",
        "[Population profiles and pointwise central 95% resampling bands](population_subsamples.png) ([PDF](population_subsamples.pdf)).",
        "[Dependence on shot count](shot_dependence.png) ([PDF](shot_dependence.pdf)).", "",
        "| Site | Full-shot hardware | Matched four-path ideal | Physical target |" + (" Latest hardware |" if comparison is not None else ""),
        "|---|---:|---:|---:|" + ("---:|" if comparison is not None else "")]
    for site in range(n):
        text.append(f"| {site+1} | {full[site]:.6f} | {references['matched_ideal'][site]:.6f} | {references['physical'][site]:.6f} |"
                    + (f" {comparison['populations'][site]:.6f} |" if comparison is not None else ""))
    text += ["", "## Subsampling variability", "",
        "Deviations below are from the full-shot hardware populations, not from the ideal prediction. The 95th-percentile maximum uses all ten sites jointly.", "",
        "| Shots/path | Total Z shots | Median max deviation | 95th-percentile max deviation | All sites within 0.02 | MAE vs physical: median [2.5%, 97.5%] |",
        "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['shots_per_path']} | {row['total_z_shots']} | {row['median_max_deviation_from_full']:.5f} | "
                    f"{row['q95_max_deviation_from_full']:.5f} | {100*row['fraction_all_sites_within_0.02_of_full']:.2f}% | "
                    f"{row['mae_vs_physical_median']:.5f} [{row['mae_vs_physical_q025']:.5f}, {row['mae_vs_physical_q975']:.5f}] |")
    text += ["", "## Interpretation and limits", "",
        f"The full-shot hardware MAE is {summaries['physical']['mae']:.8f} against the physical target and {summaries['matched_ideal']['mae']:.8f} against the same four ideal paths. Increasing subsample size concentrates the estimates around this measured hardware profile; it does not move their expectation toward ideal.", "",
        "At 2048 shots/path, four paths use 8192 Z measurements, matching the total number of Z measurements in the latest one-shot run. This matches the measurement count only: the trajectory ensemble and hardware execution occasion differ.", "",
        "With one shot on each of only four paths, each site estimate is restricted to 0, 0.25, 0.5, 0.75 or 1. This is very different from one shot on each of 8192 paths.", "",
        "For path r with N saved shots, the exact conditional covariance of an S-shot subsample mean is (1-S/N) times the unbiased sample covariance of whole bitstrings, divided by S. The equal-weight four-path mean has the sum of these covariances divided by 4 squared. Exact and empirical covariance matrices are saved in analysis.json.", "",
        "These are finite-pool resampling ranges, not confidence bands for the true hardware or ideal state. Selecting all 8192 saved shots gives zero deviation from the full-shot estimate by construction; an independent new 8192-shot run would still fluctuate. The original full-shot populations have nominal per-site shot SE about 0.0026–0.0028, conditional on the four paths.", "",
        "The data are aggregated counts without shot ordering. This study cannot establish temporal drift, identify gate/reset/readout mechanisms, or reproduce the newer trajectory ensemble. Central 95% profile bands are per-site; only the maximum-deviation statistic in the table addresses all sites jointly.", "",
        f"Empirical single-site resampling variances agree with their exact conditional values to within {100*payload['max_relative_monte_carlo_variance_discrepancy']:.2f}% relative error across all nonzero-variance shot levels/sites ({args.repeats:,} Monte Carlo replicates).", ""]
    budget_row = next((row for row in rows if row["shots_per_path"] == 2048), None)
    if budget_row is not None:
        text += [f"At the matched measurement budget (2048 shots per old path), the median population MAE versus physical is {budget_row['mae_vs_physical_median']:.6f}, with central 95% resampling range [{budget_row['mae_vs_physical_q025']:.6f}, {budget_row['mae_vs_physical_q975']:.6f}].", ""]
        if comparison is not None:
            text += [f"The latest observed population MAE is {payload['comparison_population_errors']['mae_vs_physical']:.6f}, outside that central range. Under the earlier fixed empirical distributions, reducing the measurement count usually does not reproduce the newer error. This is descriptive, not a significance test comparing runs: the four-path ensemble, execution conditions, and uncertainty in the finite original pool are not held equal to the newer experiment.", ""]
    continuation = " " + chr(92)
    command = ["MPLCONFIGDIR=/tmp/model3-subsampling-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
               ".venv/bin/python Model3/Experiment5/subsample_populations.py",
               f"  --run-directory {args.run_directory}"]
    if args.compare_analysis:
        command.append(f"  --compare-analysis {args.compare_analysis}")
    text += ["## Reproduction and preservation", "", "```bash",
             *(line+continuation for line in command[:-1]), command[-1], "```", "",
             "CSV files contain the summary, every site quantile and every path/site quantile. No Runtime connection, circuit execution or trajectory evolution was used. Input hashes were verified unchanged after writing the analysis.", ""]
    (output/"Analysis.md").write_text("\n".join(text))
    for filename, digest in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"input changed during analysis: {filename}")
    payload["inputs_unchanged"] = True
    (output/"analysis.json").write_text(json.dumps(serializable(payload), indent=2)+"\n")
    print(json.dumps({"output": str(output), "summary": rows}, indent=2))


if __name__ == "__main__":
    main()
