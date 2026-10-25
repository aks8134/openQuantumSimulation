"""Exact XX one-body dynamics and bounded local-observable steady-state checks.

Uses N x N matrices only; no many-body density matrix, circuit or sampling.
Conventions: C_ij=<c_j^dagger c_i>, H=c J sum(XX+YY), I=2a Im C_j,j+1.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def run_experiment(times, *, number_of_qubits=3, J=1., delta=0., gamma=2.,
                   p_left=.9, p_right=.1, exchange_prefactor=.5, initial_states=None,
                   steady_tolerance=.02, until_steady=False, progress=None):
    started = time.perf_counter()
    n = number_of_qubits
    if not isinstance(n, int) or n < 2:
        raise ValueError("number_of_qubits must be an integer >=2")
    if not all(np.isfinite(x) for x in (J, delta, gamma, p_left, p_right, exchange_prefactor, steady_tolerance)):
        raise ValueError("covariance parameters must be finite")
    if delta != 0:
        raise ValueError("covariance solver requires delta=0 (XX)")
    if J == 0 or gamma <= 0 or exchange_prefactor <= 0 or steady_tolerance <= 0:
        raise ValueError("require J !=0, gamma >0, prefactor >0 and steady tolerance >0")
    if not 0 <= p_left <= 1 or not 0 <= p_right <= 1:
        raise ValueError("reservoir populations must be in [0,1]")
    requested = np.asarray(times, dtype=float)
    if (requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested))
            or np.any(requested < 0) or np.any(np.diff(requested) <= 0)):
        raise ValueError("times must be nonempty, finite, nonnegative and strictly increasing")
    labels = list(initial_states) if initial_states is not None else ["0"*n, "1"*n, "1"+"0"*(n-1)]
    if (not labels or len(set(labels)) != len(labels)
            or any(label != "half_mixed" and (len(label) != n or set(label)-{"0", "1"}) for label in labels)):
        raise ValueError("initial states must be distinct N-bit strings or half_mixed")
    populations = np.array([[.5]*n if label == "half_mixed" else [int(bit) for bit in label] for label in labels])
    initial = np.array([np.diag(row) for row in populations], dtype=complex)
    a = 2*exchange_prefactor*J
    h = np.diag(np.full(n-1, a), 1)+np.diag(np.full(n-1, a), -1)
    rates = np.zeros(n); rates[[0, -1]] = gamma
    P = np.zeros((n, n)); P[0, 0] = gamma*p_left; P[-1, -1] = gamma*p_right
    A = -1j*h-np.diag(rates)/2
    target = solve_continuous_lyapunov(A, -P)
    # A+A†=-diag(rates)<=0: exp(A t) contracts the Euclidean norm.
    # The Frobenius covariance error therefore bounds all future local
    # population/current-over-|J| errors, with the prefactor below.
    bound_factor = max(1., 4*exchange_prefactor)
    covariances, bound_rows, evaluated = [], [], []
    stop_reason = "requested_times_completed"
    if progress:
        progress(f"Exact covariance evolution: {n} x {n} matrices, {len(labels)} preparations")
    for t in requested:
        E = expm(A*t)
        value = target+E@(initial-target)@E.conj().T
        bounds = bound_factor*np.linalg.norm(value-target, axis=(-2, -1))
        covariances.append(value); bound_rows.append(bounds); evaluated.append(float(t))
        if until_steady and np.all(bounds <= steady_tolerance):
            stop_reason = "future_local_error_bound_met"
            break
    if until_steady and stop_reason != "future_local_error_bound_met":
        stop_reason = "time_cap_reached_without_bound"
    C = np.stack(covariances, axis=1)
    t = np.asarray(evaluated)
    future_bounds = np.stack(bound_rows, axis=1)
    pop = np.diagonal(C, axis1=-2, axis2=-1).real
    current = 2*a*np.diagonal(C, offset=1, axis1=-2, axis2=-1).imag
    target_pop = np.diag(target).real
    target_current = 2*a*np.diag(target, 1).imag
    pop_error = np.max(abs(pop-target_pop), axis=-1)
    current_error = np.max(abs(current-target_current), axis=-1)/abs(J)
    local = np.maximum(pop_error, current_error)
    eigenvalues = np.linalg.eigvalsh(C)
    derivative = A@C+C@A.conj().T+P
    continuity = np.zeros_like(pop)
    continuity[..., 0] += gamma*(p_left-pop[..., 0])
    continuity[..., -1] += gamma*(p_right-pop[..., -1])
    continuity[..., :-1] -= current
    continuity[..., 1:] += current
    analytic_current = 2*a*a*gamma/(gamma*gamma+4*a*a)*(p_left-p_right)
    analytic_pop = np.full(n, (p_left+p_right)/2)
    analytic_pop[[0, -1]] = [p_left-analytic_current/gamma, p_right+analytic_current/gamma]
    first = lambda values: [next((float(t[k]) for k, v in enumerate(row) if v <= steady_tolerance), None) for row in values]
    steady = dict(tolerance=steady_tolerance, until_steady=bool(until_steady), stop_reason=stop_reason,
        requested_final_time=float(requested[-1]), actual_final_time=float(t[-1]),
        metric="max(max site population error, max bond excitation-current error / abs(J))",
        final_local_errors=local[:, -1].tolist(), final_future_error_bounds=future_bounds[:, -1].tolist(),
        final_local_within_tolerance=(local[:, -1] <= steady_tolerance).tolist(),
        future_bound_met=(future_bounds[:, -1] <= steady_tolerance).tolist(),
        first_observed_time_within_tolerance=first(local), first_bound_passing_time=first(future_bounds),
        bound_factor=bound_factor,
        interpretation="The covariance Frobenius bound controls these local observables at every later time for this exact generator (up to numerical precision). It is not a many-body state distance. Crossing times are resolved only on the requested observation grid; failure before the cap is reported explicitly.")
    diagnostics = dict(stationary_lyapunov_residual=float(np.linalg.norm(A@target+target@A.conj().T+P)),
        maximum_covariance_hermiticity_error=float(np.max(np.linalg.norm(C-C.conj().swapaxes(-1, -2), axis=(-2, -1)))),
        minimum_covariance_eigenvalue=float(eigenvalues.min()), maximum_covariance_eigenvalue=float(eigenvalues.max()),
        stationary_minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(target).min()),
        stationary_maximum_covariance_eigenvalue=float(np.linalg.eigvalsh(target).max()),
        continuity_residual=float(np.max(abs(np.diagonal(derivative, axis1=-2, axis2=-1).real-continuity))),
        maximum_stationary_population_error_vs_analytic=float(np.max(abs(target_pop-analytic_pop))),
        maximum_stationary_current_error_vs_analytic=float(np.max(abs(target_current-analytic_current))))
    return dict(times=t, labels=labels, initial_populations=populations, covariance=C, target_covariance=target,
        populations=pop, magnetizations=1-2*pop, currents=current,
        stationary=dict(populations=target_pop.tolist(), Z=(1-2*target_pop).tolist(), I=target_current.tolist(), j=(-2*target_current).tolist()),
        local_observable_errors=local, future_error_bounds=future_bounds, diagnostics=diagnostics,
        steady_state=steady, timings_seconds=dict(covariance_evolution_and_checks=time.perf_counter()-started))


def save_results(result, output_dir, stem, parameters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    r = result
    np.savez_compressed(output/f"{stem}.npz", times=r["times"], initial_states=r["labels"],
        initial_populations=r["initial_populations"], C_exact=r["covariance"], C_stationary=r["target_covariance"],
        n_exact=r["populations"], Z_exact=r["magnetizations"], I_exact=r["currents"], j_exact=-2*r["currents"],
        local_observable_errors=r["local_observable_errors"], future_error_bounds=r["future_error_bounds"],
        **{f"stationary_{key}": value for key, value in r["stationary"].items()})
    summary = dict(experiment="Model3/Experiment1/covariance", solver="exact_covariance", parameters=parameters,
        initial_states=r["labels"], initial_populations=r["initial_populations"].tolist(), times=r["times"].tolist(),
        stationary=r["stationary"], steady_state=r["steady_state"], diagnostics=r["diagnostics"],
        final_populations=r["populations"][:, -1].tolist(), final_currents=r["currents"][:, -1].tolist(),
        timings_seconds=r["timings_seconds"],
        scope="Exact XX populations/currents from one-body covariances; no Trotter steps, trajectories, shots or hardware noise.")
    (output/f"{stem}.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    n = parameters["number_of_qubits"]
    sites = list(dict.fromkeys([0, (n-1)//2, n-1]))
    bonds = list(dict.fromkeys([0, (n-2)//2, n-2]))
    fig, axes = plt.subplots(len(r["labels"]), 3, figsize=(15, 3.6*len(r["labels"])),
                            squeeze=False, layout="constrained")
    for state, label in enumerate(r["labels"]):
        for j in sites:
            color = plt.get_cmap("tab10")(j % 10)
            axes[state, 0].plot(r["times"], r["populations"][state, :, j], color=color, label=f"Site {j+1}")
            axes[state, 0].axhline(r["stationary"]["populations"][j], color=color, linestyle="--", alpha=.65)
        for j in bonds:
            axes[state, 1].plot(r["times"], r["currents"][state, :, j], label=f"Bond {j+1}")
        axes[state, 1].axhline(r["stationary"]["I"][0], color="k", linestyle="--", label="Stationary")
        axes[state, 2].semilogy(r["times"], np.maximum(r["local_observable_errors"][state], 1e-16), label="Actual local error")
        axes[state, 2].semilogy(r["times"], np.maximum(r["future_error_bounds"][state], 1e-16), "--", label="Future error bound")
        axes[state, 2].axhline(r["steady_state"]["tolerance"], color="k", linestyle=":", label="Tolerance")
        for col, ylabel in enumerate(("Population", "Excitation current", "Population/current-over-|J| error")):
            axes[state, col].set(xlabel="Physical time", ylabel=ylabel, title=f"{label}: {ylabel}")
            axes[state, col].grid(alpha=.2); axes[state, col].legend(fontsize=8)
    fig.suptitle(f"Exact Model3 XX reference: N={n}, J={parameters['J']:g}, gamma={parameters['gamma']:g}")
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{stem}.{suffix}", dpi=160)
    plt.close(fig)
    columns = ["initial_state", "time"]+[f"n_{j+1}" for j in range(n)]+[f"I_{j+1}" for j in range(n-1)]+["local_error", "future_error_bound"]
    with (output/f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(columns)
        for state, label in enumerate(r["labels"]):
            for index, t in enumerate(r["times"]):
                writer.writerow([label, t, *r["populations"][state, index], *r["currents"][state, index],
                                 r["local_observable_errors"][state, index], r["future_error_bounds"][state, index]])
    text = ["# Exact XX covariance reference", "",
        f"N={n}, J={parameters['J']:g}, gamma={parameters['gamma']:g}, pL={parameters['p_left']:g}, pR={parameters['p_right']:g}.", "",
        f"![Populations, excitation currents and convergence]({stem}.png)", "",
        f"Stop reason: **{r['steady_state']['stop_reason']}**. Last physical time: **{r['times'][-1]:.6g}**. Tolerance: **{r['steady_state']['tolerance']:.6g}**.", "",
        "| Initial state | Final local error | Future local error bound | Bound passed |", "|---|---:|---:|---|"]
    for state, label in enumerate(r["labels"]):
        text.append(f"| {label} | {r['local_observable_errors'][state, -1]:.8g} | {r['future_error_bounds'][state, -1]:.8g} | {r['steady_state']['future_bound_met'][state]} |")
    text += ["", f"Stationary populations: `{r['stationary']['populations']}`.",
        f"Stationary excitation currents: `{r['stationary']['I']}`.", "",
        "Plots display the first/middle/last sites and bonds. The error metric, stopping decision and CSV use every site and bond. Dashed population/current lines show their stationary targets.", "",
        r["steady_state"]["interpretation"], "",
        "This is continuous-time exact Lindblad evolution of one-body observables. Observation times are not Trotter timesteps. No finite-step dilation bias, trajectory sampling, measurement shots or device noise is included. The full many-body density matrix is not constructed.", ""]
    (output/f"{stem}.md").write_text("\n".join(text), encoding="utf-8")
