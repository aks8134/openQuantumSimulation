"""Compare saved Experiment4 arrays. No evolution, channel, or solver imports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    runs = output.parent / "runs"
    rows, details, sources, curves = [], {}, {}, []
    for directory in sorted(runs.glob("budget_*")):
        loaded = {}
        for filename in ("study.json", "search.json", "replay_R64.json"):
            path = directory / filename
            if not path.exists():
                break
            raw = path.read_bytes()
            sources[str(path.relative_to(output.parent))] = hashlib.sha256(raw).hexdigest()
            loaded[filename] = json.loads(raw)
        if len(loaded) != 3:
            continue
        study, search, replay = (loaded[k] for k in ("study.json", "search.json", "replay_R64.json"))
        config, selected = study["configuration"], search["selected"]
        n, J = config["n_qubits"], config["J"]
        times = np.asarray(replay["times"])
        first = selected["checked_from_step"]

        def observables(key):
            return np.concatenate((np.asarray(replay[key]["populations"])[0],
                                   np.asarray(replay[key]["excitation_flows"])[0] / J), axis=-1)

        target = np.r_[replay["stationary_observables"]["populations"],
                       np.asarray(replay["stationary_observables"]["excitation_flows"]) / J]
        mean = observables("mean_channel_observables")
        sampled = observables("observables")
        exact = observables("exact_reference_observables")
        se = observables("trajectory_standard_errors")
        measures = {}
        series = {}
        for label, a, b in (("mean_to_target", mean, target),
                            ("sampled_to_target", sampled, target),
                            ("exact_to_target", exact, target),
                            ("mean_to_exact", mean, exact),
                            ("sampled_to_mean", sampled, mean)):
            errors = np.abs(a - b)
            distance = errors.max(axis=-1)
            series[label] = distance
            node, observable = np.unravel_index(errors[first:].argmax(), errors[first:].shape)
            minimum = int(distance.argmin())
            measures[label] = {"final": float(distance[-1]),
                "tail_max": float(distance[first:].max()),
                "minimum_over_saved_nodes": float(distance[minimum]),
                "minimum_time": float(times[minimum]),
                "tail_max_time": float(times[first + node]),
                "tail_max_observable": (f"population site {observable}" if observable < n
                    else f"current bond {observable-n}-{observable-n+1}")}
        assessment = replay["assessment"]
        for computed, saved in ((measures["mean_to_target"]["tail_max"], assessment["tail_mean_error"]),
                                (measures["sampled_to_target"]["tail_max"], assessment["tail_sampled_error"]),
                                (measures["sampled_to_mean"]["tail_max"], assessment["max_sampling_distance"]),
                                (2*float(se[first:].max()), assessment["max_two_standard_errors"])):
            if not np.isclose(computed, saved, rtol=1e-12, atol=1e-14):
                raise ValueError(f"Saved assessment mismatch: {directory.name}")
        constants = [v for v in search["constant_baselines"] if v["tail_window_covered"]]
        baseline = min(constants, key=lambda v: v["tail_mean_error"])
        row = {"run": directory.name, "max_dt": max(config["step_grid"]),
            "block_steps": config["block_steps"], "preparation": selected["preparation"],
            "steps": selected["steps"], "final_time": selected["final_time"],
            "mean_tail_error": measures["mean_to_target"]["tail_max"],
            "sampled_tail_error": measures["sampled_to_target"]["tail_max"],
            "mean_final_error": measures["mean_to_target"]["final"],
            "sampled_final_error": measures["sampled_to_target"]["final"],
            "exact_tail_error": measures["exact_to_target"]["tail_max"],
            "sampling_tail_distance": measures["sampled_to_mean"]["tail_max"],
            "max_two_SE": 2*float(se[first:].max()),
            "best_constant_mean_tail_error": baseline["tail_mean_error"],
            "mean_error_reduction_vs_best_constant_percent":
                100*(1-selected["tail_mean_error"]/baseline["tail_mean_error"]),
            "beam_expansions": search["beam_expansions"],
            "search_limit_reached": search["search_limit_reached"]}
        rows.append(row)
        details[directory.name] = {"configuration": config,
            "selected": {k: v for k, v in selected.items() if k not in
                         ("step_dts", "time_nodes", "final_mean_covariance_real_gauge")},
            "metrics": measures, "best_per_preparation": search["best_per_preparation"],
            "best_constant": baseline,
            "mean_tail_population_error": float(abs(mean[first:, :n]-target[:n]).max()),
            "mean_tail_current_error_over_J": float(abs(mean[first:, n:]-target[n:]).max()),
            "mean_tail_max_peak_to_peak": float(np.ptp(mean[first:], axis=0).max()),
            "max_mean_covariance_replay_difference": replay["max_mean_covariance_replay_difference"]}
        curves.append((row, times, series, first))

    # Confirm input JSON files were not altered during analysis.
    for relative, digest in sources.items():
        if hashlib.sha256((output.parent / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed while analyzing: {relative}")
    result = {"scope": "Saved JSON arrays only; no new trajectories, evolution, stationary solves, or search.",
              "metric": "max over all sites of population error and all bonds of current error / J",
              "rows": rows, "details": details, "source_sha256": sources}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (row, times, series, first) in zip(axes.flat, curves):
        for label, color, style, title in (
                ("mean_to_target", "C0", "-", "Mean vs physical steady state"),
                ("sampled_to_target", "C1", "-", "Sampled vs physical steady state"),
                ("exact_to_target", "C2", "--", "Exact dynamics vs steady state"),
                ("sampled_to_mean", "C3", ":", "Sampled vs mean")):
            axis.semilogy(times, series[label].clip(min=1e-6), color=color, linestyle=style,
                          label=title, linewidth=1.3)
        axis.axhline(.01, color="black", linestyle="--", linewidth=.9, label="Total tolerance 0.01")
        axis.axhline(.006, color="gray", linestyle=":", linewidth=.9, label="Mean allowance 0.006")
        axis.axvspan(times[first], times[-1], color="gray", alpha=.12)
        axis.set(xlabel="Physical time", ylabel="Maximum local error", ylim=(1e-3, 1),
                 title=f"{row['run'].removeprefix('budget_')}: max dt {row['max_dt']:g}, block {row['block_steps']}\n"
                       f"{row['preparation']}, {row['steps']} steps")
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7, loc="upper right")
    figure.suptitle("Existing Experiment4 runs: all 10 populations and 9 currents\nShading: assessed final 10 units of physical time")
    figure.savefig(output / "steady_state_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
