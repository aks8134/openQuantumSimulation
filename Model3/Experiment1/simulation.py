"""Boundary-driven XXZ transport: exact Lindblad and shared-ancilla dilation.

Defaults match the three-site unequal-reservoir example: H=J/2 sum(XX+YY),
Gamma=2, p_L=0.9, p_R=0.1. Reuse Model 2's time evolution engines.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.sparse.linalg import expm_multiply, spsolve


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical
from Model2.Experiment1 import simulation as engines


def stationary_state(liouvillian, dimension, *, excitation_numbers=None):
    """Solve L rho=0 with tr(rho)=1, using column-major vectorization."""
    indices = np.arange(dimension**2)
    if excitation_numbers is not None:
        # U(1) covariance and uniqueness put the stationary state in the
        # charge-zero sector. Retain coherences within each number sector.
        weights = np.asarray(excitation_numbers)
        indices = np.flatnonzero((weights[:, None] == weights[None, :]).reshape(-1, order="F"))
    constrained = liouvillian[indices, :][:, indices].tolil(copy=True)
    constrained[0, :] = np.eye(dimension).reshape(-1, order="F")[indices]
    rhs = np.zeros(len(indices), complex)
    rhs[0] = 1
    vector = np.zeros(dimension**2, complex)
    vector[indices] = spsolve(constrained.tocsc(), rhs)
    rho = vector.reshape(dimension, dimension, order="F")
    if not np.all(np.isfinite(rho)):
        raise ValueError("stationary solve did not yield a finite density matrix")
    return rho


def build_model(number_of_qubits=3, J=1.0, delta=0.0, gamma=2.0,
                p_left=0.9, p_right=0.1, exchange_prefactor=0.5):
    """Return H, four endpoint jumps, Z, n, I and the full stationary rho."""
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (p_left, p_right)):
        raise ValueError("reservoir populations must be finite and in [0, 1]")
    H, _, Z_ops, populations, currents, _ = engines.build_model(
        number_of_qubits, J, delta, gamma, 0.5, exchange_prefactor,
    )
    jumps = []
    for site, p in ((0, p_left), (number_of_qubits - 1, p_right)):
        loss = classical.operator_on_site(classical.SIGMA_MINUS, site, number_of_qubits)
        jumps.extend((np.sqrt(gamma * p) * loss.getH(),
                      np.sqrt(gamma * (1 - p)) * loss))
    target = stationary_state(classical.build_liouvillian(H, jumps), H.shape[0],
                              excitation_numbers=[k.bit_count() for k in range(H.shape[0])])
    return H, tuple(jumps), Z_ops, populations, currents, target


def dilation_kraus(H, jumps, dt, formula):
    """Apply the same dilation exponential to its reset isometry directly.

    Sparse exponential action avoids constructing full joint unitaries.
    No change is made to the channel, factor order, or ancilla reset rule.
    """
    system_term, jump_terms = classical.build_hamiltonian_dilation_terms(H, jumps, dt)
    if formula == "unsplit":
        generators = (sum(jump_terms, start=system_term),)
    elif formula == "lie":
        generators = (system_term, *jump_terms)
    elif formula == "strang":
        generators = (*(term / 2 for term in jump_terms), system_term,
                      *(term / 2 for term in reversed(jump_terms)))
    else:
        raise ValueError("formula must be unsplit, lie or strang")
    dimension = H.shape[0]
    isometry = np.zeros(((len(jumps) + 1) * dimension, dimension), complex)
    isometry[:dimension] = np.eye(dimension)
    for generator in generators:
        exponent = -1j * generator
        isometry = expm_multiply(exponent, isometry, traceA=exponent.diagonal().sum())
    return isometry.reshape(len(jumps) + 1, dimension, dimension)


def xx_stationary_targets(number_of_qubits=3, J=1.0, gamma=2.0,
                          p_left=0.9, p_right=0.1, exchange_prefactor=0.5):
    """Equal-rate XX profile, with hopping a=2*c*J (also valid for N=2)."""
    hopping = 2 * exchange_prefactor * J
    current = 2 * hopping**2 * gamma / (gamma**2 + 4 * hopping**2) * (p_left - p_right)
    populations = np.full(number_of_qubits, (p_left + p_right) / 2)
    populations[0] = p_left - current / gamma
    populations[-1] = p_right + current / gamma
    flows = np.full(number_of_qubits - 1, current)
    return dict(populations=populations, Z=1 - 2 * populations, I=flows, j=-2 * flows)


def xx_correlations(times, labels, *, J=1.0, gamma=2.0, p_left=0.9,
                    p_right=0.1, exchange_prefactor=0.5):
    """Independent XX one-body solution C_ij=<c_j^dagger c_i>, for delta=0.

    A C* + C* A^dagger = -P, C(t)=C*+e^(At)(C(0)-C*)e^(A^dagger t).
    This function returns observables rather than a many-body density matrix.
    """
    times = engines.validate_times(times)
    n = len(labels[0])
    hopping = 2 * exchange_prefactor * J
    h = np.diag(np.full(n - 1, hopping), 1) + np.diag(np.full(n - 1, hopping), -1)
    rates = np.zeros(n)
    rates[[0, -1]] = gamma
    injection = np.zeros(n)
    injection[[0, -1]] = gamma * np.asarray([p_left, p_right])
    A = -1j * h - np.diag(rates) / 2
    target = solve_continuous_lyapunov(A, -np.diag(injection))
    initial = np.asarray([np.diag([int(bit) for bit in bits]) for bits in labels])
    covariance = np.empty((len(labels), len(times), n, n), complex)
    for index, time in enumerate(times):
        E = expm(A * time)
        covariance[:, index] = target + E @ (initial - target) @ E.conj().T
    populations = np.diagonal(covariance, axis1=-2, axis2=-1).real
    currents = 2 * hopping * np.diagonal(covariance, offset=1, axis1=-2, axis2=-1).imag
    return 1 - 2 * populations, currents


def run_experiment(times, *, number_of_qubits=3, J=1.0, delta=0.0, gamma=2.0,
                   p_left=0.9, p_right=0.1, exchange_prefactor=0.5, max_dt=0.05,
                   methods=("unsplit", "lie", "strang"), progress=None):
    timings = {}
    def timed(label, operation):
        if progress is not None:
            progress(f"{label}: starting")
        started = time.perf_counter()
        value = operation()
        timings[label] = time.perf_counter() - started
        if progress is not None:
            progress(f"{label}: finished in {timings[label]:.2f} s")
        return value
    times = engines.validate_times(times)
    H, jumps, Z_ops, populations, currents, target = timed("Model and stationary solve", lambda: build_model(
        number_of_qubits, J, delta, gamma, p_left, p_right, exchange_prefactor,
    ))
    labels, initial = engines.initial_states(number_of_qubits)
    L = classical.build_liouvillian(H, jumps)
    trajectories = {"exact": timed("Exact Lindblad trajectory", lambda: engines.exact_evolve(initial, L, times))}
    for method in methods:
        trajectories[method] = timed(f"{method.capitalize()} dilation", lambda: engines.dilation_evolve(
            initial, H, jumps, times, max_dt=max_dt, formula=method, kraus_builder=dilation_kraus,
        ))
    if progress is not None:
        progress("Observables and diagnostics: starting")
    diagnostic_start = time.perf_counter()
    Z = {name: engines.expectations(rhos, Z_ops) for name, rhos in trajectories.items()}
    I = {name: engines.expectations(rhos, currents) for name, rhos in trajectories.items()}
    target_Z = engines.expectations(target[None, None], Z_ops)[0, 0]
    target_I = engines.expectations(target[None, None], currents)[0, 0]
    stationary = dict(populations=((1 - target_Z) / 2).tolist(), Z=target_Z.tolist(),
                      I=target_I.tolist(), j=(-2 * target_I).tolist())
    exact = trajectories["exact"]
    derivatives = np.asarray([
        [(L @ rho.reshape(-1, order="F")).reshape(H.shape, order="F")
         for rho in state] for state in exact
    ])
    pop = (1 - Z["exact"]) / 2
    rhs = np.zeros_like(pop)
    rhs[..., 0] += gamma * (p_left - pop[..., 0])
    rhs[..., -1] += gamma * (p_right - pop[..., -1])
    rhs[..., :-1] -= I["exact"]
    rhs[..., 1:] += I["exact"]
    diagnostics = {
        "stationary_residual": float(np.linalg.norm(L @ target.reshape(-1, order="F"))),
        "stationary_trace_error": float(abs(np.trace(target) - 1)),
        "stationary_hermiticity_error": float(np.linalg.norm(target - target.conj().T)),
        "stationary_minimum_eigenvalue": float(np.linalg.eigvalsh(target).min()),
        "continuity_residual": float(np.max(np.abs(engines.expectations(derivatives, populations) - rhs))),
    }
    if delta == 0:
        analytic = xx_stationary_targets(number_of_qubits, J, gamma, p_left, p_right, exchange_prefactor)
        diagnostics["maximum_stationary_Z_error_vs_analytic"] = float(np.max(np.abs(target_Z - analytic["Z"])))
        diagnostics["maximum_stationary_I_error_vs_analytic"] = float(np.max(np.abs(target_I - analytic["I"])))
        correlation_Z, correlation_I = xx_correlations(
            times, labels, J=J, gamma=gamma, p_left=p_left,
            p_right=p_right, exchange_prefactor=exchange_prefactor,
        )
        diagnostics["maximum_Z_error_vs_correlations"] = float(np.max(np.abs(Z["exact"] - correlation_Z)))
        diagnostics["maximum_I_error_vs_correlations"] = float(np.max(np.abs(I["exact"] - correlation_I)))
    for name, rhos in trajectories.items():
        diagnostics[name] = {
            "maximum_trace_error": float(np.max(np.abs(np.trace(rhos, axis1=-2, axis2=-1) - 1))),
            "maximum_hermiticity_error": float(np.max(np.linalg.norm(rhos - rhos.conj().swapaxes(-1, -2), axis=(-2, -1)))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(rhos).min()),
            "maximum_density_error_vs_exact": float(np.max(np.linalg.norm(rhos - exact, axis=(-2, -1)))),
            "maximum_Z_error_vs_exact": float(np.max(np.abs(Z[name] - Z["exact"]))),
            "maximum_current_error_vs_exact": float(np.max(np.abs(I[name] - I["exact"]))),
            "final_Z": Z[name][:, -1].tolist(),
            "final_current": I[name][:, -1].tolist(),
            "final_distance_to_target": np.linalg.norm(rhos[:, -1] - target, axis=(-2, -1)).tolist(),
        }
        if name in {"lie", "strang"} and "unsplit" in trajectories:
            diagnostics[name]["maximum_density_error_vs_unsplit_dilation"] = float(
                np.max(np.linalg.norm(rhos - trajectories["unsplit"], axis=(-2, -1))))
    timings["Observables and diagnostics"] = time.perf_counter() - diagnostic_start
    if progress is not None:
        progress(f"Observables and diagnostics: finished in {timings['Observables and diagnostics']:.2f} s")
    return dict(times=times, labels=labels, target=target, stationary=stationary,
                trajectories=trajectories, magnetizations=Z, currents=I, diagnostics=diagnostics,
                timings_seconds=timings)


def save_results(result, output_dir, stem, parameters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(times=result["times"], initial_states=np.asarray(result["labels"]), target=result["target"])
    for observable, values in result["stationary"].items():
        arrays[f"stationary_{observable}"] = values
    for name, rhos in result["trajectories"].items():
        arrays[f"rho_{name}"] = rhos
        arrays[f"Z_{name}"] = result["magnetizations"][name]
        arrays[f"n_{name}"] = (1 - result["magnetizations"][name]) / 2
        arrays[f"I_{name}"] = result["currents"][name]
        arrays[f"j_{name}"] = -2 * result["currents"][name]
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    summary = dict(parameters=parameters, initial_states=result["labels"], times=result["times"].tolist(),
                   stationary=result["stationary"], diagnostics=result["diagnostics"],
                   timings_seconds=result["timings_seconds"])
    (output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    styles = {"exact": "-", "unsplit": "--", "lie": ":", "strang": "-."}
    for state, bits in enumerate(result["labels"]):
        for name, rhos in result["trajectories"].items():
            for site in range(parameters["number_of_qubits"]):
                axes[state, 0].plot(result["times"], result["magnetizations"][name][state, :, site],
                                    styles[name], color=f"C{site}", label=f"{name}, site {site + 1}")
            for bond in range(parameters["number_of_qubits"] - 1):
                axes[state, 1].plot(result["times"], result["currents"][name][state, :, bond],
                                    styles[name], color=f"C{bond}", label=f"{name}, bond {bond + 1}")
            distance = np.linalg.norm(rhos[state] - result["target"], axis=(-2, -1))
            axes[state, 2].plot(result["times"], np.maximum(distance, 1e-16), styles[name], label=name)
        for site, value in enumerate(result["stationary"]["Z"]):
            axes[state, 0].axhline(value, color=f"C{site}", lw=1, alpha=0.6)
        axes[state, 1].axhline(result["stationary"]["I"][0], color="black", lw=1, label="stationary current")
        axes[state, 2].set_yscale("log")
        for column, title in enumerate(("site magnetization", "excitation current", "distance to stationary state")):
            axes[state, column].set_title(f"|{bits}>: {title}")
            axes[state, column].set_xlabel("Time")
            axes[state, column].grid(alpha=0.25)
            axes[state, column].legend(fontsize=7, ncol=2 if column < 2 else 1)
        for column, label in enumerate(("<Z>", "<I>", "Frobenius norm")):
            axes[state, column].set_ylabel(label)
    figure.suptitle(f"Model 3: pL={parameters['p_left']:g}, pR={parameters['p_right']:g}, "
                   f"H prefactor={parameters['exchange_prefactor']:g} J")
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=160)
    plt.close(figure)


def main(default_mode="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--times", type=float, nargs="+")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--p-left", type=float, default=0.9)
    parser.add_argument("--p-right", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--exchange-prefactor", type=float, choices=(0.25, 0.5), default=0.5)
    parser.add_argument("--mode", choices=("all", "exact", "unsplit", "lie", "strang"), default=default_mode)
    parser.add_argument("--skip-unsplit", action="store_true", help="omit the additional unsplit reference in lie/strang mode")
    parser.add_argument("--controls", action="store_true", help="also run equal-reservoir and reversed-bias controls")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.times is None and (args.num_times < 2 or not np.isfinite(args.t_final) or args.t_final <= 0):
        parser.error("require --num-times >= 2 and finite --t-final > 0")
    times = args.times if args.times is not None else np.linspace(0, args.t_final, args.num_times)
    methods = {"all": ("unsplit", "lie", "strang"), "exact": (),
               "unsplit": ("unsplit",), "lie": ("unsplit", "lie"),
               "strang": ("unsplit", "strang")}[args.mode]
    if args.skip_unsplit:
        if args.mode not in {"lie", "strang"}:
            parser.error("--skip-unsplit requires --mode lie or --mode strang")
        methods = tuple(method for method in methods if method != "unsplit")
    cases = [(args.p_left, args.p_right)]
    if args.controls:
        cases.extend(((0.25, 0.25), (args.p_right, args.p_left)))
    for p_left, p_right in dict.fromkeys(cases):
        parameters = dict(number_of_qubits=args.n_qubits, J=args.J, delta=args.delta,
                          gamma=args.gamma, p_left=p_left, p_right=p_right,
                          exchange_prefactor=args.exchange_prefactor, max_dt=args.max_dt)
        try:
            print(f"Starting N={args.n_qubits}, mode={args.mode}, pL={p_left:g}, pR={p_right:g}, "
                  f"max_dt={args.max_dt:g}", flush=True)
            result = run_experiment(times, **parameters, methods=methods,
                                    progress=lambda message: print(message, flush=True))
        except ValueError as error:
            parser.error(str(error))
        print(f"Reservoirs: pL={p_left:g}, pR={p_right:g}; H prefactor={args.exchange_prefactor:g} J")
        print("Stationary targets: " + json.dumps(result["stationary"]))
        print(json.dumps(result["diagnostics"], indent=2))
        stem = (f"{args.mode}_N{args.n_qubits}_pL{p_left:g}_pR{p_right:g}"
                f"_J{args.J:g}_D{args.delta:g}_G{args.gamma:g}_c{args.exchange_prefactor:g}")
        print("Saving arrays, diagnostics and plot: starting", flush=True)
        save_results(result, args.output_dir, stem, parameters)
        print(f"Saved density matrices, observables, diagnostics and plot: {args.output_dir / stem}")


if __name__ == "__main__":
    main()
