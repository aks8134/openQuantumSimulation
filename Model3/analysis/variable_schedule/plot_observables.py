"""Render report figures from saved observables only; no physical evolution."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
MANUAL = (
    ("754e0cbbdb98", "0.1 → 0.05 → 0.01 (T=20)", "#1764ab"),
    ("e5257c026bae", "1 → 0.5 → 0.2 → 0.1 → 0.05", "#d16b14"),
    ("1f4884fb6f5a", "2 until t=30; then 0.1", "#298348"),
    ("5c1d59d5d02a", "constant 2", "#ce3434"),
    ("7cbec7bdcc09", "3 until t=50; then 0.1", "#9365bb"),
)
FAMILIES = (("populations", (0, 4, 9)), ("excitation_flows", (0, 4, 8)))
sources = {}
endpoints = []


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def axes_for(title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey="row")
    fig.suptitle(title, fontsize=15, y=.98)
    for row, (family, indices) in enumerate(FAMILIES):
        for ax, index in zip(axes[row], indices):
            ax.set_title(f"Site {index}" if row == 0 else f"Bond {index}–{index+1}")
            ax.grid(alpha=.22)
            ax.set_xlabel("Physical time")
        axes[row, 0].set_ylabel("Population ⟨n⟩" if row == 0 else "Excitation current I")
    axes[0, 0].set_ylim(-.03, 1.03)
    return fig, axes


def finish(fig, stem, handles):
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)),
               fontsize=10, frameon=False, bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .11, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=160)
    plt.close(fig)


def record_endpoints(data, name):
    for prep, initial in enumerate(data["initial_states"]):
        for family, indices in FAMILIES:
            for index in indices:
                endpoints.append({"run": name, "initial_state": initial,
                    "family": family, "index_zero_based": index, "time": data["times"][-1],
                    "sampled": data["observables"][family][prep][-1][index],
                    "trajectory_se": data["trajectory_standard_errors"][family][prep][-1][index],
                    "deterministic_mean": data["mean_channel_observables"][family][prep][-1][index],
                    "exact_lindblad": data["exact_reference_observables"][family][prep][-1][index],
                    "physical_stationary": data["stationary_observables"][family][index]})


def manual_figures():
    runs = []
    for identifier, label, color in MANUAL:
        path = f"Model3/Experiment3/results/classical_randomized_N10_J1_g1_R64_{identifier}.json"
        data = read(path)
        record_endpoints(data, identifier)
        runs.append((data, label, color))
    # Use the finely recorded T=60 reference, without interpolation/evolution.
    reference = runs[1][0]
    for prep, name in enumerate(("vacuum", "filled", "left_excitation")):
        fig, axes = axes_for(f"Manual timestep schedules: actual observables | {name.replace('_', ' ')}")
        for row, (family, indices) in enumerate(FAMILIES):
            for ax, index in zip(axes[row], indices):
                for data, label, color in runs:
                    ax.plot(data["times"], np.asarray(data["mean_channel_observables"][family])[prep, :, index],
                            color=color, lw=1.65)
                ax.plot(reference["times"], np.asarray(reference["exact_reference_observables"][family])[prep, :, index],
                        "--", color="black", lw=1.5)
                ax.axhline(reference["stationary_observables"][family][index], color="#666666", ls="-.", lw=1.2)
        handles = [Line2D([], [], color=color, label=label, lw=2) for _, label, color in runs]
        handles += [Line2D([], [], color="black", ls="--", label="Exact continuous Lindblad"),
                    Line2D([], [], color="#666666", ls="-.", label="Physical steady state")]
        finish(fig, f"manual_observables_{name}", handles)


def replay_figure(data, stem, title):
    fig, axes = axes_for(title)
    for row, (family, indices) in enumerate(FAMILIES):
        for ax, index in zip(axes[row], indices):
            sample = np.asarray(data["observables"][family])[0, :, index]
            se = np.asarray(data["trajectory_standard_errors"][family])[0, :, index]
            ax.plot(data["times"], sample, color="#1764ab", lw=1.9)
            ax.fill_between(data["times"], sample-se, sample+se, color="#1764ab", alpha=.17)
            ax.plot(data["times"], np.asarray(data["mean_channel_observables"][family])[0, :, index],
                    ":", color="#d16b14", lw=2)
            ax.plot(data["times"], np.asarray(data["exact_reference_observables"][family])[0, :, index],
                    "--", color="#333333", lw=1.5)
            ax.axhline(data["stationary_observables"][family][index], color="#298348", ls="-.", lw=1.3)
    handles = [Line2D([], [], color="#1764ab", label="Sampled mean (shading: ±1 trajectory SE)"),
               Line2D([], [], color="#d16b14", ls=":", label="Deterministic mean channel"),
               Line2D([], [], color="#333333", ls="--", label="Exact continuous Lindblad"),
               Line2D([], [], color="#298348", ls="-.", label="Physical steady state")]
    finish(fig, stem, handles)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manual_figures()
    forward = read("Model3/Experiment4/runs/budget_419029a6f9bc/replay_R8192.json")
    record_endpoints(forward, "117_step_forward")
    replay_figure(forward, "forward117_observables", "117-step preparation: actual observables | half mixed, R=8192, seed 29")
    cosine = read("Model3/Experiment7/runs/comparison_d3fd93cb452e/cosine/replay_seed29_R8192.json")
    record_endpoints(cosine, "72_step_cosine")
    replay_figure(cosine, "cosine_observables", "72-step cosine preparation: actual observables | half mixed, R=8192, seed 29")
    dts = np.asarray(cosine["schedule"]["step_dts"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.arange(1, len(dts)+1), dts, "o-", ms=3, color="#1764ab")
    ax.set(xlabel="Preparation step", ylabel="Physical timestep h", title="Selected cosine schedule: 72 executed timesteps")
    ax.grid(alpha=.22)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"cosine_timesteps.{suffix}", dpi=160)
    plt.close(fig)
    with (OUTPUT / "observable_endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    for relative, digest in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    (OUTPUT / "observable_figures.json").write_text(json.dumps({
        "scope": "Plot saved observable values only; no errors plotted, new evolution, interpolation, or hardware calls.",
        "manual_curves": "Deterministic mean channels; five schedules, separate fixed preparation per figure.",
        "sites_zero_based": [0, 4, 9], "bonds_zero_based": [[0, 1], [4, 5], [8, 9]],
        "source_sha256": sources, "original_sources_unchanged": True}, indent=2)+"\n")
    print(f"Saved observable figures, PDFs, endpoint CSV, and source hashes in {OUTPUT}")


if __name__ == "__main__":
    main()
