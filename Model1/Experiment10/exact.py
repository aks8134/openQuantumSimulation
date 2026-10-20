"""Exact boundary-loss XY chain in its invariant single-excitation sector.

Only the N-component no-jump amplitude is propagated. The unconditional
Lindblad state is its outer product plus the accumulated vacuum probability.
This script does not construct a many-qubit or reduced-space Liouvillian.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Model:
    n_qubits: int = 10
    J: float = 1.0
    h: float = 0.0
    gamma: float = 5.0
    initial_site: int | None = None

    def __post_init__(self):
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, (int, np.integer)) or self.n_qubits < 2:
            raise ValueError("n_qubits must be an integer of at least two")
        if not all(np.isfinite(value) for value in (self.J, self.h, self.gamma)):
            raise ValueError("J, h and gamma must be finite")
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        site = self.n_qubits // 2 if self.initial_site is None else self.initial_site
        if isinstance(site, bool) or not isinstance(site, (int, np.integer)) or not 0 <= site < self.n_qubits:
            raise ValueError("initial_site must be an integer in [0, n_qubits)")
        object.__setattr__(self, "initial_site", int(site))


@dataclass(frozen=True)
class Evolution:
    times: np.ndarray
    populations: np.ndarray
    xy_correlations: np.ndarray
    excitation_flows: np.ndarray
    total_excitation: np.ndarray
    vacuum_probability: np.ndarray
    final_amplitude: np.ndarray
    max_continuity_residual: float
    max_excitation_balance_residual: float


def build_generator(model):
    """Return G = -i H_1 - gamma/2 (|0><0| + |N-1><N-1|).

    H_1 has hopping J and diagonal -h relative to the vacuum energy. The
    uniform field contributes only an irrelevant common amplitude phase.
    """
    diagonal = np.full(model.n_qubits, 1j * model.h, dtype=complex)
    diagonal[[0, -1]] -= model.gamma / 2.0
    hopping = np.full(model.n_qubits - 1, -1j * model.J, dtype=complex)
    return diags((hopping, diagonal, hopping), (-1, 0, 1), format="csr")


def validate_times(times):
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("times must be finite and nonnegative")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("times must be strictly increasing, without duplicates")
    return grid


def evolve(model, times, *, steady_tolerance=1e-8, until_steady=False, progress=None):
    """Compute exact sampled observables, storing O(N * number_of_times) data.

    Requested times are observation times, not Trotter/integration steps.
    With until_steady, the requested horizon is a cap: stop at the first
    sampled time with total excitation <= steady_tolerance. It is not a
    guarantee that convergence will occur before that cap.
    """
    grid = validate_times(times)
    if not np.isfinite(steady_tolerance) or not 0 < steady_tolerance < 1:
        raise ValueError("steady_tolerance must lie strictly between zero and one")
    generator = build_generator(model)
    generator_trace = generator.diagonal().sum()
    amplitude = np.zeros(model.n_qubits, dtype=complex)
    amplitude[model.initial_site] = 1.0
    populations = np.empty((model.n_qubits, grid.size), dtype=float)
    correlations = np.empty((model.n_qubits - 1, grid.size), dtype=float)
    flows = np.empty_like(correlations)
    total = np.empty(grid.size, dtype=float)
    continuity_error = balance_error = 0.0
    previous_time = 0.0

    for index, observation_time in enumerate(grid):
        interval = float(observation_time) - previous_time
        if interval > 0:
            amplitude = expm_multiply(interval * generator, amplitude,
                                      traceA=interval * generator_trace)
        population = np.abs(amplitude)**2
        coherence = amplitude[:-1] * amplitude[1:].conj()
        flow = 2.0 * model.J * coherence.imag
        populations[:, index] = population
        correlations[:, index] = 4.0 * coherence.real
        flows[:, index] = flow
        total[index] = population.sum()

        # Analytic instantaneous derivatives, not coarse-grid differences.
        derivative = 2.0 * np.real(amplitude.conj() * (generator @ amplitude))
        rhs = np.zeros(model.n_qubits)
        rhs[:-1] -= flow
        rhs[1:] += flow
        rhs[[0, -1]] -= model.gamma * population[[0, -1]]
        continuity_error = max(continuity_error, float(np.max(np.abs(derivative - rhs))))
        loss = -model.gamma * (population[0] + population[-1])
        balance_error = max(balance_error, float(abs(derivative.sum() - loss)))
        previous_time = float(observation_time)
        if progress is not None:
            progress(index + 1, grid.size, previous_time, float(total[index]))
        if until_steady and total[index] <= steady_tolerance:
            break

    count = index + 1
    return Evolution(grid[:count].copy(), populations[:, :count].copy(),
                     correlations[:, :count].copy(), flows[:, :count].copy(),
                     total[:count].copy(), 1.0 - total[:count], amplitude.copy(),
                     continuity_error, balance_error)


def configuration(model, requested_times, *, steady_tolerance, until_steady):
    return {"solver_version": 1, "n_qubits": int(model.n_qubits),
            "J": float(model.J), "h": float(model.h), "gamma": float(model.gamma),
            "initial_site": model.initial_site, "requested_times": list(map(float, requested_times)),
            "steady_tolerance": float(steady_tolerance), "until_steady": bool(until_steady)}


def output_paths(output_directory, config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:12]
    gamma = str(config["gamma"]).replace(".", "p")
    stem = f"exact_N{config['n_qubits']}_gamma{gamma}_site{config['initial_site']}_{digest}"
    directory = Path(output_directory)
    return {"result": directory / "results" / f"{stem}.json",
            "observables": directory / "figures" / f"{stem}_observables.png"}


def save_result(path, model, evolution, config, wall_seconds):
    excitation = evolution.total_excitation
    payload = {
        "experiment": "Model1/Experiment10 exact single-excitation evolution",
        "configuration": config,
        "method": "exact sparse exponential action on the no-jump amplitude; vacuum loss restored analytically",
        "validity": "one initially localized excitation, number-conserving XY Hamiltonian, two boundary lowering jumps only",
        "reduced_hilbert_dimension": model.n_qubits + 1,
        "propagated_vector_dimension": model.n_qubits,
        "times": evolution.times.tolist(),
        "observables": {"populations": evolution.populations.tolist(),
                        "xy_correlations": evolution.xy_correlations.tolist(),
                        "excitation_flows": evolution.excitation_flows.tolist(),
                        "document_currents": (-2.0 * evolution.excitation_flows).tolist(),
                        "total_excitation": excitation.tolist(),
                        "vacuum_probability": evolution.vacuum_probability.tolist()},
        "final_amplitude": {"real": evolution.final_amplitude.real.tolist(),
                            "imag": evolution.final_amplitude.imag.tolist()},
        "steady_state": {"target": "vacuum", "distance_metric": "trace distance to the vacuum = total excitation",
                         "tolerance": config["steady_tolerance"],
                         "reached": bool(excitation[-1] <= config["steady_tolerance"]),
                         "final_distance": float(excitation[-1]),
                         "final_time": float(evolution.times[-1]),
                         "first_sampled_time_within_tolerance": next(
                             (float(t) for t, p in zip(evolution.times, excitation)
                              if p <= config["steady_tolerance"]), None),
                         "unique_vacuum_condition": "gamma > 0 and J != 0 for this connected chain"},
        "diagnostics": {"max_continuity_residual": evolution.max_continuity_residual,
                        "max_excitation_balance_residual": evolution.max_excitation_balance_residual,
                        "max_total_excitation_increase": float(max(0.0, np.max(np.diff(excitation), initial=0.0))),
                        "minimum_vacuum_probability": float(evolution.vacuum_probability.min()),
                        "max_probability_overshoot": float(max(0.0, excitation.max() - 1.0))},
        "evolution_wall_seconds": wall_seconds,
        "figure_generation_status": "pending",
    }
    atomic_write_json(path, payload)
    return payload


def atomic_write_json(path, payload):
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


def plot_result(payload, output_path):
    # No Qiskit or interactive display dependency, and no many-qubit arrays.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(payload["times"])
    values = payload["observables"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    marker = "o" if times.size == 1 else None
    for axis, key, title, ylabel in (
        (axes[0, 0], "populations", "Site populations", r"$\langle n_i\rangle$"),
        (axes[0, 1], "xy_correlations", "Nearest-neighbor XY correlations", r"$\langle X_iX_{i+1}+Y_iY_{i+1}\rangle$"),
        (axes[1, 0], "excitation_flows", "Excitation flow", r"$I_{i\to i+1}$"),
    ):
        array = np.asarray(values[key])
        palette = plt.get_cmap("tab20" if array.shape[0] <= 20 else "turbo")
        for index, curve in enumerate(array):
            label = f"site {index}" if key == "populations" else f"bond {index}–{index + 1}"
            axis.plot(times, curve, label=label, marker=marker, linewidth=1.2,
                      color=palette(index / max(1, array.shape[0] - 1)))
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small", ncol=2 if array.shape[0] > 5 else 1)
        if key == "populations":
            axis.set_ylim(bottom=0)
        elif key == "xy_correlations" and np.max(np.abs(array)) <= 1e-12:
            axis.set_ylim(-0.05, 0.05)
            axis.text(0.5, 0.12, "All bond curves overlap at zero (within 1e−12)",
                      ha="center", transform=axis.transAxes, fontsize="small")
    axis = axes[1, 1]
    # Floor only displayed values to make log(0) safe; stored data is unchanged.
    axis.semilogy(times, np.maximum(values["total_excitation"], np.finfo(float).tiny),
                  label="total excitation = distance to vacuum", marker=marker)
    axis.axhline(payload["steady_state"]["tolerance"], color="black", linestyle="--", label="vacuum tolerance")
    axis.set_title("Approach to the vacuum steady state")
    axis.set_xlabel("Time")
    axis.set_ylabel("Remaining excitation / trace distance")
    axis.grid(alpha=0.3)
    axis.legend(fontsize="small")
    cfg = payload["configuration"]
    figure.suptitle(f"Exact boundary-loss XY chain: N={cfg['n_qubits']}, γ={cfg['gamma']}, J={cfg['J']}, h={cfg['h']}")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--initial-site", type=int, help="zero-based initial excitation site; default N//2")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--t-final", type=float, default=None, help="uniform-grid horizon (default 1000)")
    grid.add_argument("--times", type=float, nargs="+", help="strictly increasing nonnegative observation times, including a single value")
    parser.add_argument("--time-points", type=int, default=None, help="uniform-grid count (default 1001); not valid with --times")
    parser.add_argument("--steady-tolerance", type=float, default=1e-8)
    parser.add_argument("--until-steady", action="store_true", help="stop at the first sampled vacuum-tolerance crossing; the requested horizon remains a cap")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-plots", action="store_true", help="save data only")
    parser.add_argument("--no-progress", action="store_true", help="suppress sampled-time progress messages")
    options = parser.parse_args(arguments)
    Model(options.n_qubits, options.J, options.h, options.gamma, options.initial_site)
    if options.times is not None:
        if options.time_points is not None:
            raise ValueError("--time-points cannot be combined with --times")
        options.times = validate_times(options.times)
    else:
        horizon = 1000.0 if options.t_final is None else options.t_final
        count = 1001 if options.time_points is None else options.time_points
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("--t-final must be positive and finite; use --times 0 for only t=0")
        if count < 2:
            raise ValueError("--time-points must be at least two")
        options.times = np.linspace(0.0, horizon, count)
    if not np.isfinite(options.steady_tolerance) or not 0 < options.steady_tolerance < 1:
        raise ValueError("--steady-tolerance must lie strictly between zero and one")
    return options


def main(options):
    model = Model(options.n_qubits, options.J, options.h, options.gamma, options.initial_site)
    config = configuration(model, options.times, steady_tolerance=options.steady_tolerance,
                           until_steady=options.until_steady)
    paths = output_paths(options.output_directory, config)
    print(f"Exact invariant-sector solver: N={model.n_qubits}, γ={model.gamma}, J={model.J}, h={model.h}", flush=True)
    print(f"Initial excitation at site {model.initial_site}; propagating {model.n_qubits} amplitudes (no Liouvillian)", flush=True)
    print(f"Requested final time {options.times[-1]:g}; {len(options.times)} observation times", flush=True)
    stride = max(1, len(options.times) // 20)

    def progress(count, total, current_time, excitation):
        if count == 1 or count == total or count % stride == 0 or (options.until_steady and excitation <= options.steady_tolerance):
            print(f"Sample {count}/{total}: t={current_time:g}, remaining excitation={excitation:.6e}", flush=True)

    start = time.perf_counter()
    result = evolve(model, options.times, steady_tolerance=options.steady_tolerance,
                    until_steady=options.until_steady, progress=None if options.no_progress else progress)
    payload = save_result(paths["result"], model, result, config, time.perf_counter() - start)
    print(f"Saved data before plotting: {paths['result']}", flush=True)
    print(f"Final distance to vacuum: {payload['steady_state']['final_distance']:.9e}; tolerance {options.steady_tolerance:g}", flush=True)
    print("Vacuum tolerance reached" if payload["steady_state"]["reached"] else "Vacuum tolerance NOT reached within the requested horizon", flush=True)
    print(f"Maximum analytic continuity residual: {result.max_continuity_residual:.3e}", flush=True)
    if options.no_plots:
        payload["figure_generation_status"] = "disabled"
    else:
        print("Generating observable/steady-state figure...", flush=True)
        plot_result(payload, paths["observables"])
        payload["figure_generation_status"] = "complete"
        payload["observables_figure"] = paths["observables"].name
        print(f"Saved figure: {paths['observables']}", flush=True)
    atomic_write_json(paths["result"], payload)
    return payload


if __name__ == "__main__":
    main(parse_arguments())
