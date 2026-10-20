"""Plot up to three representative observables from saved JSON, without evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
COLORS = ("#1764ab", "#d16b14", "#298348")


def selected_indices(width, supplied):
    indices = list(dict.fromkeys((0, (width-1)//2, width-1))) if supplied is None else supplied
    if not 1 <= len(indices) <= 3 or len(set(indices)) != len(indices):
        raise ValueError("select one to three distinct indices")
    if any(index < 0 or index >= width for index in indices):
        raise ValueError(f"indices must be between 0 and {width-1}")
    return indices


def plot_saved(payload, sites, bonds, path, *, late_fraction=None):
    times = np.asarray(payload["times"])
    labels = payload["initial_states"]
    config = payload["configuration"]
    physical = config["physics"]
    figure, axes = plt.subplots(len(labels), 3, figsize=(17, 3.8*len(labels)), squeeze=False)
    for prep, label in enumerate(labels):
        for column, (family, indices, title, ylabel) in enumerate((
                ("populations", sites, "Selected site populations", "Population"),
                ("excitation_flows", bonds, "Selected bond currents", "Excitation current"))):
            axis = axes[prep, column]
            for color, index in zip(COLORS, indices):
                sample = np.asarray(payload["observables"][family])[prep, :, index]
                error = np.asarray(payload["trajectory_standard_errors"][family])[prep, :, index]
                exact = np.asarray(payload["exact_reference_observables"][family])[prep, :, index]
                mean = np.asarray(payload["mean_channel_observables"][family])[prep, :, index]
                name = f"site {index}" if column == 0 else f"bond {index}–{index+1}"
                axis.plot(times, sample, color=color, linewidth=1.7, label=name)
                axis.fill_between(times, sample-error, sample+error, color=color, alpha=.13)
                axis.plot(times, exact, "--", color=color, linewidth=1.7)
                axis.plot(times, mean, ":", color=color, linewidth=2.0)
                axis.axhline(payload["stationary_observables"][family][index],
                             color=color, linestyle="-.", linewidth=1.1, alpha=.65)
            axis.set_title(f"Initial {label}\n{title}", fontsize=11)
            axis.set_ylabel(ylabel)
            axis.legend(fontsize=9, ncol=len(indices), loc="best")
        axis = axes[prep, 2]
        for name, display, style, color in (("sampled", "Sampled mean", "-", COLORS[0]),
                ("mean_channel", "Mean channel", ":", COLORS[1]),
                ("exact", "Exact Lindblad", "--", "#333333")):
            distance = np.asarray(payload["steady_state"][name]["local_observable_distance"])[prep]
            axis.semilogy(times, np.maximum(distance, np.finfo(float).tiny),
                          linestyle=style, color=color, linewidth=1.7, label=display)
        tolerance = payload["steady_state"]["tolerance"]
        axis.axhline(tolerance, color="#993399", linestyle="-.", label=f"Tolerance {tolerance:g}")
        axis.set_title(f"Initial {label}\nStationary distance over ALL sites/bonds", fontsize=11)
        axis.set_ylabel("Max population / current÷J deviation")
        axis.legend(fontsize=8, loc="best")
    start = times[0] if late_fraction is None else times[-1]-late_fraction*(times[-1]-times[0])
    for axis in axes.flat:
        axis.set_xlim(start, times[-1])
        axis.set_xlabel("Physical time")
        axis.grid(alpha=.25)
    # Set late-time vertical limits using visible saved observations,
    # their uncertainty bands, and the physical stationary targets.
    if late_fraction is not None:
        visible = times >= start
        for prep in range(len(labels)):
            for col, (family, indices) in enumerate((("populations", sites), ("excitation_flows", bonds))):
                values = []
                for index in indices:
                    sample = np.asarray(payload["observables"][family])[prep, visible, index]
                    se = np.asarray(payload["trajectory_standard_errors"][family])[prep, visible, index]
                    values.extend((sample-se, sample+se,
                        np.asarray(payload["exact_reference_observables"][family])[prep, visible, index],
                        np.asarray(payload["mean_channel_observables"][family])[prep, visible, index],
                        np.array([payload["stationary_observables"][family][index]])))
                low, high = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
                margin = max(.003, .08*(high-low))
                axes[prep, col].set_ylim(low-margin, high+margin)
            distance_values = np.concatenate([np.asarray(payload["steady_state"][name]["local_observable_distance"])[prep, visible]
                                              for name in ("sampled", "mean_channel", "exact")]+[np.array([tolerance])])
            positive = distance_values[distance_values > 0]
            if len(positive):
                axes[prep, 2].set_ylim(float(positive.min())/1.25, float(positive.max())*1.25)
    handles = [Line2D([0], [0], color="#333333", linestyle=style, label=label)
               for style, label in (("-", "Sampled mean (shading: ±1 SE)"),
                                     ("--", "Exact continuous Lindblad"),
                                     (":", "Deterministic mean channel"),
                                     ("-.", "Physical stationary target"))]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    schedule = config["schedule_input"]
    if schedule["segments"] is not None:
        schedule_text = " ".join(f"{end:g}:{dt:g}" for end, dt in schedule["segments"])
    elif schedule["constant_dt"] is not None:
        schedule_text = f"constant dt={schedule['constant_dt']:g}"
    else:
        schedule_text = f"{len(payload['schedule']['step_dts'])} explicit steps"
    view = "Full saved time range" if late_fraction is None else f"Late-time view: {start:g} ≤ t ≤ {times[-1]:g}"
    figure.suptitle(f"Model3: N={physical['n_qubits']}, J={physical['J']:g}, γ={physical['gamma']:g}, R={config['trajectories']}"
                    f"  |  schedule END:DT: {schedule_text}\n{view} — selected representative sites/bonds; zero-based indices", fontsize=12)
    figure.tight_layout(rect=(0, .04, 1, .94))
    try:
        figure.savefig(path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(figure)
    return [float(start), float(times[-1])]


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=DIRECTORY/"results")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"figures"/"three_observables")
    parser.add_argument("--sites", type=int, nargs="+", help="up to three zero-based site indices")
    parser.add_argument("--bonds", type=int, nargs="+", help="up to three zero-based bond indices")
    parser.add_argument("--late-fraction", type=float, default=1/3, help="fraction of saved horizon shown in the zoom")
    options = parser.parse_args(arguments)
    if not np.isfinite(options.late_fraction) or not 0 < options.late_fraction <= 1:
        raise ValueError("late-fraction must be in (0,1]")
    paths = sorted(options.results_directory.glob("classical_randomized_*.json"))
    if not paths:
        raise ValueError("no saved experiment results found")
    options.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for source in paths:
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        width = payload["configuration"]["physics"]["n_qubits"]
        sites, bonds = selected_indices(width, options.sites), selected_indices(width-1, options.bonds)
        stem = source.stem
        full = options.output_directory/f"{stem}_three_observables.png"
        late = options.output_directory/f"{stem}_three_observables_late.png"
        plot_saved(payload, sites, bonds, full)
        window = plot_saved(payload, sites, bonds, late, late_fraction=options.late_fraction)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during plotting: {source}")
        records.append({"source": str(source.resolve()), "source_sha256": digest,
            "N": width, "run_id": stem.rsplit("_", 1)[-1], "sites": sites, "bonds": bonds,
            "schedule": payload["configuration"]["schedule_input"],
            "full": full.name, "late": late.name, "late_time_range": window})
        print(f"Plotted {records[-1]['run_id']}: sites {sites}, bonds {bonds}", flush=True)
    manifest = {"scope": "Saved arrays only; no evolution or new reference computations. Original plots and results preserved.",
                "stationary_distance": "Original saved metric over ALL sites/bonds, not only the displayed subset.",
                "runs": records}
    (options.output_directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    index = ["# Three representative observable plots", "",
        "Plots regenerated from existing saved data only. No simulations were rerun.", "",
        "For N=10: sites 0, 4, 9 and bonds 0–1, 4–5, 8–9. For N=3: all three sites and both bonds.", "",
        "Solid = sampled mean; shading = ±1 standard error; dashed = exact Lindblad; dotted = deterministic mean channel; dash-dot = physical stationary target.", "",
        "The right column retains the stationary-distance metric over **all** sites and bonds. Late-time plots show the last third of the saved horizon; this is a viewing window, not a declaration of convergence.", "",
        "| N | Run | Schedule END:DT | Full range | Late-time zoom |", "|---:|---|---|---|---|"]
    for record in sorted(records, key=lambda r: (r["N"], r["run_id"])):
        segments = record["schedule"]["segments"]
        description = " ".join(f"{end:g}:{dt:g}" for end, dt in segments) if segments is not None else str(record["schedule"])
        index.append(f"| {record['N']} | `{record['run_id']}` | `{description}` | [Full]({record['full']}) | [Zoom]({record['late']}) |")
    (options.output_directory/"Plots.md").write_text("\n".join(index)+"\n")
    print(f"Saved {2*len(records)} figures and index: {options.output_directory/'Plots.md'}", flush=True)


if __name__ == "__main__":
    main()
