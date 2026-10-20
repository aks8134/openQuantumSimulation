"""Ideal randomized circuit-channel expectations using N amplitudes per trajectory.

This reproduces the finite-step even/odd and selected-boundary dilation/reset
maps, not an exact exponential damping substitution. No circuit construction,
transpilation, sampling, hardware access, or mitigation is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model1.Experiment11 import physics

DIRECTORY = Path(__file__).resolve().parent
FAMILIES = ("populations", "xy_correlations", "excitation_flows")


@dataclass(frozen=True, slots=True)
class Evolution:
    times: np.ndarray
    means: tuple[np.ndarray, ...]
    standard_errors: tuple[np.ndarray, ...]
    trajectory_excitation: np.ndarray
    final_amplitudes: np.ndarray
    step_dts: tuple[float, ...]
    step_counts: tuple[int, ...]
    boundary_choices: tuple[tuple[int, ...], ...]


def advance(amplitudes, model, dt, boundaries):
    """Return the next unnormalized no-jump amplitudes; inputs are not mutated."""
    amplitudes = np.asarray(amplitudes, dtype=complex)
    boundaries = np.asarray(boundaries)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != model.n_qubits:
        raise ValueError("amplitudes must have shape (trajectories, n_qubits)")
    if boundaries.shape != (len(amplitudes),) or not np.all(np.isin(boundaries, (0, 1))):
        raise ValueError("one left/right boundary choice is required per trajectory")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    output = amplitudes.copy()
    cosine, sine = np.cos(model.J * dt), -1j * np.sin(model.J * dt)
    for parity in (0, 1):
        left = output[:, parity:-1:2].copy()
        right = output[:, parity+1::2].copy()
        output[:, parity:-1:2] = cosine * left + sine * right
        output[:, parity+1::2] = sine * left + cosine * right
    # Field phase relative to vacuum; harmless for these observables but retained.
    output *= np.exp(1j * model.h * dt)
    selected_sites = np.where(boundaries == 0, 0, model.n_qubits - 1)
    # Keep the cosine's sign even for large angles. Do not replace by exp(-gamma*dt).
    output[np.arange(len(output)), selected_sites] *= np.cos(np.sqrt(2 * model.gamma * dt))
    return output


def summarize(values):
    count = len(values)
    mean = values.mean(axis=0)
    error = np.zeros_like(mean) if count == 1 else values.std(axis=0, ddof=1) / np.sqrt(count)
    return mean, error


def evolve(model, times, *, trajectories=4, seed=29, maximum_dt=0.1, progress=None):
    if (isinstance(trajectories, bool) or not isinstance(trajectories, (int, np.integer))
            or trajectories < 1 or isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("trajectories must be a positive integer and seed a nonnegative integer")
    values = physics.validate_times(times)
    dts, counts, choices = physics.schedule(values, trajectories, seed, maximum_dt)
    boundaries = np.asarray(choices, dtype=int)
    amplitude = np.zeros((trajectories, model.n_qubits), dtype=complex)
    amplitude[:, model.initial_site] = 1
    means = tuple(np.empty((width, len(values))) for width in (model.n_qubits, model.n_qubits-1, model.n_qubits-1))
    errors = tuple(np.empty_like(value) for value in means)
    total = np.empty((trajectories, len(values)))
    previous_count = 0
    for ti, count in enumerate(counts):
        for step in range(previous_count, count):
            amplitude = advance(amplitude, model, dts[step], boundaries[:, step])
        population = np.abs(amplitude)**2
        coherence = amplitude[:, :-1] * amplitude[:, 1:].conj()
        for destination, error, measured in zip(means, errors,
                (population, 4*coherence.real, 2*model.J*coherence.imag)):
            destination[:, ti], error[:, ti] = summarize(measured)
        total[:, ti] = population.sum(axis=1)
        previous_count = count
        if progress is not None:
            progress(ti+1, len(values), float(values[ti]), float(total[:, ti].mean()))
    return Evolution(values.copy(), means, errors, total, amplitude, dts, counts, choices)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def canonical_time(value):
    value = float(value)
    rounded = round(value, 12)
    return rounded if abs(value - rounded) <= 8 * abs(np.spacing(value)) else value


def configuration(options):
    return {"schema_version": 1, "physics": options.model.payload(),
        "trajectories": options.trajectories, "seed_trajectories": options.seed_trajectories,
        "trotter_delta_t": options.trotter_delta_t, "steady_tolerance": options.steady_tolerance,
        "method": "ideal finite-step randomized boundary dilation; even then odd bonds; signed cosine damping"}


def output_paths(options, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    number = lambda value: format(value, ".8g").replace(".", "p")
    model = options.model
    stem = (f"classical_randomized_N{model.n_qubits}_J{number(model.J)}_h{number(model.h)}"
            f"_g{number(model.gamma)}_R{options.trajectories}_dt{number(options.trotter_delta_t)}_{digest}")
    return {"result": options.output_directory / "results" / f"{stem}.json",
            "observables": options.output_directory / "figures" / f"{stem}_observables.png"}


def make_payload(evolution, config, *, existing=None, wall_seconds=0.0):
    if existing is not None and existing.get("configuration") != config:
        raise ValueError("existing classical result has an incompatible configuration")
    schedule = {"requested_times": evolution.times.tolist(), "step_dts": list(evolution.step_dts),
                "step_counts": list(evolution.step_counts),
                "boundary_choices": [list(row) for row in evolution.boundary_choices]}
    run_id = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()[:16]
    replacement = {canonical_time(t) for t in evolution.times}
    samples = {} if existing is None else {row["time"]: row for row in existing["samples"]
                                          if canonical_time(row["time"]) not in replacement}
    for ti, t in enumerate(evolution.times):
        samples[canonical_time(t)] = {"time": canonical_time(t), "run_id": run_id,
            "substep_count": evolution.step_counts[ti],
            "observables": {name: data[:, ti].tolist() for name, data in zip(FAMILIES, evolution.means)},
            "trajectory_standard_errors": {name: data[:, ti].tolist()
                                           for name, data in zip(FAMILIES, evolution.standard_errors)},
            "trajectory_total_excitation": evolution.trajectory_excitation[:, ti].tolist()}
    rows = [samples[t] for t in sorted(samples)]
    total = np.asarray([row["trajectory_total_excitation"] for row in rows]).T
    mean_total, error_total = summarize(total)
    tolerance = config["steady_tolerance"]
    runs = {} if existing is None else dict(existing.get("runs", {}))
    runs[run_id] = {**schedule, "simulation_wall_seconds": wall_seconds,
        "final_amplitudes": {"real": evolution.final_amplitudes.real.tolist(),
                             "imag": evolution.final_amplitudes.imag.tolist()}}
    history = [] if existing is None else list(existing.get("run_history", []))
    history.append({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id})
    return {"experiment": "Model1/Experiment11/classical", "configuration": config,
        "times": [row["time"] for row in rows], "samples": rows, "runs": runs, "run_history": history,
        "observables": {name: np.asarray([row["observables"][name] for row in rows]).T.tolist() for name in FAMILIES},
        "trajectory_standard_errors": {name: np.asarray([row["trajectory_standard_errors"][name] for row in rows]).T.tolist() for name in FAMILIES},
        "total_excitation": mean_total.tolist(), "total_excitation_standard_error": error_total.tolist(),
        "vacuum_probability": (1-mean_total).tolist(),
        "uncertainty": {"trajectory_standard_error_estimable": config["trajectories"] > 1,
                        "measurement_shot_noise": False, "hardware_noise": False,
                        "note": "sample standard deviation / sqrt(R); R=1 returns zero placeholders, not an uncertainty estimate"},
        "steady_state": {"target": "vacuum", "distance_metric": "mean total excitation = ensemble trace distance to vacuum",
            "tolerance": tolerance, "reached": bool(mean_total[-1] <= tolerance),
            "final_distance": float(mean_total[-1]), "final_time": rows[-1]["time"],
            "first_sampled_time_within_tolerance": next((row["time"] for row, p in zip(rows, mean_total) if p <= tolerance), None)},
        "figure_generation_status": "pending", "reference_generation_status": "disabled"}


def plot_result(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = np.asarray(payload["times"])
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    reference = payload.get("exact_reference_observables")
    for axis, name, title in zip(axes.flat[:3], FAMILIES, ("Site populations", "XY correlations", "Excitation flow")):
        for index, (curve, error) in enumerate(zip(payload["observables"][name], payload["trajectory_standard_errors"][name])):
            curve, error = np.asarray(curve), np.asarray(error)
            line, = axis.plot(times, curve, marker="o" if len(times) == 1 else None,
                             label=f"site {index}" if name == "populations" else f"bond {index}–{index+1}")
            axis.fill_between(times, curve-error, curve+error, color=line.get_color(), alpha=0.13)
            if reference is not None:
                axis.plot(times, reference[name][index], "--", color=line.get_color(), alpha=0.8)
        axis.set_title(title + ("; dashed = exact Lindblad" if reference is not None else ""))
        axis.set_xlabel("Time")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small", ncol=2)
    axis = axes[1, 1]
    mean = np.asarray(payload["total_excitation"])
    error = np.asarray(payload["total_excitation_standard_error"])
    floor = np.finfo(float).tiny
    axis.semilogy(times, np.maximum(mean, floor), label="randomized mean excitation")
    axis.fill_between(times, np.maximum(mean-error, floor), np.maximum(mean+error, floor), alpha=0.13)
    if reference is not None:
        axis.semilogy(times, np.maximum(payload["exact_reference_total_excitation"], floor), "--", label="exact Lindblad excitation")
    axis.axhline(payload["steady_state"]["tolerance"], color="black", linestyle=":", label="steady tolerance")
    axis.set_title("Trace distance to the vacuum steady state")
    axis.set_xlabel("Time")
    axis.grid(alpha=0.3)
    axis.legend(fontsize="small")
    cfg = payload["configuration"]
    model = cfg["physics"]
    figure.suptitle(f"Classical randomized Lie–Trotter: N={model['n_qubits']}, J={model['J']}, γ={model['gamma']}, R={cfg['trajectories']}, Δt≤{cfg['trotter_delta_t']}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--initial-site", type=int)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--seed-trajectories", type=int, default=29)
    parser.add_argument("--trotter-delta-t", type=float, default=0.1)
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--times", type=float, nargs="+")
    grid.add_argument("--t-final", type=float)
    parser.add_argument("--time-points", type=int)
    parser.add_argument("--steady-tolerance", type=float, default=1e-3)
    parser.add_argument("--optimized-classical-reference", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    options = parser.parse_args(arguments)
    options.model = physics.Model(options.n_qubits, options.J, options.h, options.gamma, options.initial_site)
    if options.trajectories < 1 or options.seed_trajectories < 0:
        raise ValueError("trajectories must be positive and seed nonnegative")
    if not np.isfinite(options.trotter_delta_t) or options.trotter_delta_t <= 0:
        raise ValueError("trotter-delta-t must be positive and finite")
    if not np.isfinite(options.steady_tolerance) or not 0 < options.steady_tolerance < 1:
        raise ValueError("steady-tolerance must lie strictly between zero and one")
    if options.times is not None:
        if options.time_points is not None:
            raise ValueError("time-points cannot be combined with times")
        options.times = physics.validate_times(options.times)
    else:
        horizon = 20.0 if options.t_final is None else options.t_final
        count = 101 if options.time_points is None else options.time_points
        if not np.isfinite(horizon) or horizon <= 0 or count < 2:
            raise ValueError("t-final must be positive and finite, and time-points >= 2")
        options.times = np.linspace(0, horizon, count)
    return options


def main(options):
    config = configuration(options)
    paths = output_paths(options, config)
    existing = json.loads(paths["result"].read_text()) if paths["result"].exists() else None
    if existing is not None and existing.get("configuration") != config:
        raise ValueError("existing result configuration differs")
    if 2*options.model.gamma*options.trotter_delta_t >= 0.1:
        warnings.warn("2*gamma*delta_t is not small: the original signed-cosine dilation is retained; compare against exact Lindblad evolution")
    print(f"Classical ideal-channel simulation: {options.model.payload()}, R={options.trajectories}, Δt≤{options.trotter_delta_t}", flush=True)
    print("No Aer, shots, transpilation, or hardware jobs", flush=True)
    stride = max(1, len(options.times)//20)
    def progress(index, count, t, total):
        if index == 1 or index == count or index % stride == 0:
            print(f"Sample {index}/{count}: t={t:g}, mean excitation={total:.6e}", flush=True)
    start = time.perf_counter()
    evolution = evolve(options.model, options.times, trajectories=options.trajectories,
        seed=options.seed_trajectories, maximum_dt=options.trotter_delta_t,
        progress=None if options.no_progress else progress)
    payload = make_payload(evolution, config, existing=existing, wall_seconds=time.perf_counter()-start)
    with_reference = options.optimized_classical_reference or (existing is not None and "exact_reference_observables" in existing)
    payload["reference_generation_status"] = "pending" if with_reference else "disabled"
    atomic_json(paths["result"], payload)
    print(f"Saved data before reference/plots: {paths['result']}", flush=True)
    if with_reference:
        reference = physics.exact_reference(options.model, payload["times"])
        payload["exact_reference_observables"] = {name: data.tolist() for name, data in zip(FAMILIES, reference)}
        exact_total = reference[0].sum(axis=0)
        payload["exact_reference_total_excitation"] = exact_total.tolist()
        payload["exact_reference_steady_state"] = {"reached": bool(exact_total[-1] <= options.steady_tolerance),
            "final_distance": float(exact_total[-1]), "tolerance": options.steady_tolerance,
            "first_sampled_time_within_tolerance": next((t for t, p in zip(payload["times"], exact_total) if p <= options.steady_tolerance), None)}
        payload["max_absolute_observable_error_vs_exact"] = np.maximum.reduce([
            np.max(np.abs(np.asarray(payload["observables"][name])-data), axis=0)
            for name, data in zip(FAMILIES, reference)]).tolist()
        payload["reference_generation_status"] = "complete"
        atomic_json(paths["result"], payload)
    if options.no_plots:
        payload["figure_generation_status"] = "disabled"
    else:
        plot_result(payload, paths["observables"])
        payload["figure_generation_status"] = "complete"
        payload["observables_figure"] = paths["observables"].name
        print(f"Saved figure: {paths['observables']}", flush=True)
    atomic_json(paths["result"], payload)
    steady = payload["steady_state"]
    print(f"Randomized distance at t={steady['final_time']:g}: {steady['final_distance']:.6e}; tolerance {options.steady_tolerance:g}, reached={steady['reached']}", flush=True)
    if with_reference:
        print(f"Exact Lindblad distance: {payload['exact_reference_steady_state']['final_distance']:.6e}", flush=True)
    return payload


if __name__ == "__main__":
    main(parse_arguments())
