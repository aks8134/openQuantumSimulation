"""Equal-reservoir XXZ benchmark: exact Lindblad and shared-ancilla dilation.

Times are absolute elapsed times from preparation at t=0, including when the
first requested time is positive. The selected default exchange prefactor is
J/4, with current J/4 (YX - XY). Use --exchange-prefactor 0.5 to match Model 1
and the excitation current in the original Model 2 excerpt.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from library import classical


def validate_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be finite and nonnegative")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    return times


def build_model(number_of_qubits=3, J=1.0, delta=0.0, gamma=2.0,
                p=0.25, exchange_prefactor=0.25):
    """Return H, four jumps, local Z, population/current operators and rho*."""
    if not isinstance(number_of_qubits, (int, np.integer)) or number_of_qubits < 2:
        raise ValueError("number_of_qubits must be an integer of at least two")
    if not all(np.isfinite(x) for x in (J, delta, gamma, p, exchange_prefactor)):
        raise ValueError("model parameters must be finite")
    if J == 0 or gamma <= 0 or not 0 < p < 1 or exchange_prefactor <= 0:
        raise ValueError("require J != 0, gamma > 0, 0 < p < 1 and prefactor > 0")
    # Reuse Model 1's tensor ordering and Pauli construction.
    H, _, X_ops, Y_ops, Z_ops = classical.build_linear_chain(
        number_of_qubits, J=2 * exchange_prefactor * J, h=0.0, gamma=0.0,
    )
    for left, right in zip(Z_ops[:-1], Z_ops[1:]):
        H += exchange_prefactor * J * delta * (left @ right)
    jumps = []
    for site in (0, number_of_qubits - 1):
        loss = classical.operator_on_site(
            classical.SIGMA_MINUS, site, number_of_qubits,
        )
        jumps.extend((np.sqrt(gamma * p) * loss.getH(),
                      np.sqrt(gamma * (1 - p)) * loss))
    identity = eye(2**number_of_qubits, dtype=complex, format="csr")
    populations = [0.5 * (identity - Z) for Z in Z_ops]
    currents = [
        exchange_prefactor * J * (
            Y_ops[site] @ X_ops[site + 1] - X_ops[site] @ Y_ops[site + 1]
        )
        for site in range(number_of_qubits - 1)
    ]
    target = np.asarray([[1.0]], dtype=complex)
    for _ in range(number_of_qubits):
        target = np.kron(target, np.diag([1 - p, p]))
    return H, tuple(jumps), Z_ops, populations, currents, target


def initial_states(number_of_qubits):
    bitstrings = ("0" * number_of_qubits, "1" * number_of_qubits,
                  "1" + "0" * (number_of_qubits - 1))
    states = []
    for bits in bitstrings:
        psi = classical.computational_state(
            number_of_qubits, [site for site, bit in enumerate(bits) if bit == "1"],
        )
        states.append(np.outer(psi, psi.conj()))
    return bitstrings, np.asarray(states)


def exact_evolve(initial, liouvillian, times):
    """Unsplit exp(t L), to floating-point precision, for arbitrary time grids."""
    times = validate_times(times)
    dimension = initial.shape[-1]
    vectors = np.column_stack([rho.reshape(-1, order="F") for rho in initial])
    trajectory = np.empty((len(initial), len(times), dimension, dimension), complex)
    previous_time = 0.0
    generator_trace = liouvillian.diagonal().sum()
    for index, time in enumerate(times):
        interval = time - previous_time
        if interval > 0:
            vectors = expm_multiply(interval * liouvillian, vectors,
                                    traceA=interval * generator_trace)
        for state, vector in enumerate(vectors.T):
            trajectory[state, index] = vector.reshape(dimension, dimension, order="F")
        previous_time = time
    return trajectory


def dilation_kraus(H, jumps, dt, formula):
    """The same dimensionless block-Hamiltonian construction as Model 1."""
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
        isometry = expm(-1j * generator.toarray()) @ isometry
    return isometry.reshape(len(jumps) + 1, dimension, dimension)


def dilation_evolve(initial, H, jumps, times, *, max_dt=0.05, formula="unsplit", kraus_builder=None):
    """Trace/reset the five-level ancilla after every step of length <= max_dt."""
    times = validate_times(times)
    if not np.isfinite(max_dt) or max_dt <= 0:
        raise ValueError("max_dt must be finite and positive")
    if formula not in {"unsplit", "lie", "strang"}:
        raise ValueError("formula must be unsplit, lie or strang")
    trajectory = np.empty((len(initial), len(times), *H.shape), complex)
    density = initial.copy()
    previous_time = 0.0
    cache = {}
    builder = dilation_kraus if kraus_builder is None else kraus_builder
    for index, time in enumerate(times):
        interval = time - previous_time
        if interval > 0:
            # Equal substeps within each interval; retain a cache across intervals.
            step_count = max(1, int(np.ceil(interval / max_dt - 1e-12)))
            dt = interval / step_count
            key = round(dt, 14)
            if key not in cache:
                cache[key] = builder(H, jumps, dt, formula)
            kraus = cache[key]
            for _ in range(step_count):
                density = sum(K @ density @ K.conj().T for K in kraus)
        trajectory[:, index] = density
        previous_time = time
    return trajectory


def expectations(trajectory, operators):
    return np.einsum("stij,oji->sto", trajectory,
                     np.asarray([op.toarray() for op in operators])).real


def run_experiment(times, *, number_of_qubits=3, J=1.0, delta=0.0, gamma=2.0,
                   p=0.25, exchange_prefactor=0.25, max_dt=0.05,
                   methods=("unsplit", "lie", "strang")):
    times = validate_times(times)
    H, jumps, Z_ops, populations, currents, target = build_model(
        number_of_qubits, J, delta, gamma, p, exchange_prefactor,
    )
    labels, initial = initial_states(number_of_qubits)
    liouvillian = classical.build_liouvillian(H, jumps)
    trajectories = {"exact": exact_evolve(initial, liouvillian, times)}
    for method in methods:
        trajectories[method] = dilation_evolve(
            initial, H, jumps, times, max_dt=max_dt, formula=method,
        )
    magnetizations = {name: expectations(rhos, Z_ops) for name, rhos in trajectories.items()}
    flows = {name: expectations(rhos, currents) for name, rhos in trajectories.items()}
    # Evaluate continuity from L rho, avoiding finite-difference time-grid errors.
    exact = trajectories["exact"]
    derivatives = np.asarray([
        [(liouvillian @ rho.reshape(-1, order="F")).reshape(H.shape, order="F")
         for rho in state] for state in exact
    ])
    pop = expectations(exact, populations)
    continuity_rhs = np.zeros_like(pop)
    continuity_rhs[..., 0] += gamma * (p - pop[..., 0])
    continuity_rhs[..., -1] += gamma * (p - pop[..., -1])
    continuity_rhs[..., :-1] -= flows["exact"]
    continuity_rhs[..., 1:] += flows["exact"]
    diagnostics = {
        "stationary_residual": float(np.linalg.norm(liouvillian @ target.reshape(-1, order="F"))),
        "continuity_residual": float(np.max(np.abs(expectations(derivatives, populations) - continuity_rhs))),
    }
    for name, rhos in trajectories.items():
        diagnostics[name] = {
            "maximum_trace_error": float(np.max(np.abs(np.trace(rhos, axis1=-2, axis2=-1) - 1))),
            "maximum_hermiticity_error": float(np.max(np.linalg.norm(rhos - rhos.conj().swapaxes(-1, -2), axis=(-2, -1)))),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(rhos).min()),
            "maximum_density_error_vs_exact": float(np.max(np.linalg.norm(rhos - exact, axis=(-2, -1)))),
            "maximum_Z_error_vs_exact": float(np.max(np.abs(magnetizations[name] - magnetizations["exact"]))),
            "final_Z": magnetizations[name][:, -1].tolist(),
            "final_current": flows[name][:, -1].tolist(),
            "final_distance_to_target": np.linalg.norm(rhos[:, -1] - target, axis=(-2, -1)).tolist(),
        }
        if name in {"lie", "strang"} and "unsplit" in trajectories:
            diagnostics[name]["maximum_density_error_vs_unsplit_dilation"] = float(
                np.max(np.linalg.norm(rhos - trajectories["unsplit"], axis=(-2, -1))))
    return dict(times=times, labels=labels, H=H, jumps=jumps, liouvillian=liouvillian,
                target=target, trajectories=trajectories, magnetizations=magnetizations,
                currents=flows, diagnostics=diagnostics)


def save_results(result, output_dir, stem, parameters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(times=result["times"], initial_states=np.asarray(result["labels"]),
                  target=result["target"])
    for name in result["trajectories"]:
        arrays[f"rho_{name}"] = result["trajectories"][name]
        arrays[f"Z_{name}"] = result["magnetizations"][name]
        arrays[f"I_{name}"] = result["currents"][name]
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    summary = dict(parameters=parameters, initial_states=result["labels"],
                   times=result["times"].tolist(), diagnostics=result["diagnostics"])
    (output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    styles = {"exact": "-", "unsplit": "--", "lie": ":", "strang": "-."}
    for state, bits in enumerate(result["labels"]):
        for name in result["trajectories"]:
            for site in range(parameters["number_of_qubits"]):
                axes[state, 0].plot(result["times"], result["magnetizations"][name][state, :, site],
                                    styles[name], color=f"C{site}", label=f"{name}, site {site + 1}")
            for bond in range(parameters["number_of_qubits"] - 1):
                axes[state, 1].plot(result["times"], result["currents"][name][state, :, bond],
                                    styles[name], color=f"C{bond}", label=f"{name}, bond {bond + 1}")
            distance = np.linalg.norm(result["trajectories"][name][state] - result["target"], axis=(-2, -1))
            axes[state, 2].plot(result["times"], np.maximum(distance, 1e-16), styles[name], label=name)
        axes[state, 0].axhline(1 - 2 * parameters["p"], color="black", lw=1, label="exact target")
        axes[state, 1].axhline(0, color="black", lw=1)
        axes[state, 2].set_yscale("log")
        for column, title in enumerate(("site magnetization", "excitation current", "distance to target")):
            axes[state, column].set_title(f"|{bits}>: {title}")
            axes[state, column].set_xlabel("Time")
            axes[state, column].grid(alpha=0.25)
            axes[state, column].legend(fontsize=7, ncol=2 if column < 2 else 1)
        axes[state, 0].set_ylabel("<Z>")
        axes[state, 1].set_ylabel("<I>")
        axes[state, 2].set_ylabel("Frobenius norm")
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=160)
    plt.close(figure)


def main(default_mode="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--times", type=float, nargs="+", help="strictly increasing elapsed times, e.g. 0 0.17 1.3 10")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--exchange-prefactor", type=float, choices=(0.25, 0.5), default=0.25)
    parser.add_argument("--mode", choices=("all", "exact", "unsplit", "lie", "strang"), default=default_mode)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.times is None and (args.num_times < 2 or not np.isfinite(args.t_final) or args.t_final <= 0):
        parser.error("require --num-times >= 2 and finite --t-final > 0")
    times = args.times if args.times is not None else np.linspace(0, args.t_final, args.num_times)
    methods = {"all": ("unsplit", "lie", "strang"), "exact": (),
               "unsplit": ("unsplit",), "lie": ("unsplit", "lie"),
               "strang": ("unsplit", "strang")}[args.mode]
    parameters = dict(number_of_qubits=args.n_qubits, J=args.J, delta=args.delta,
                      gamma=args.gamma, p=args.p, exchange_prefactor=args.exchange_prefactor,
                      max_dt=args.max_dt)
    try:
        result = run_experiment(times, **parameters, methods=methods)
    except ValueError as error:
        parser.error(str(error))
    dimension = 2**args.n_qubits
    print(f"System H and rho: {dimension} x {dimension}")
    print(f"Liouvillian: {dimension**2} x {dimension**2}; four boundary jumps")
    print(f"Dilation K: {5 * dimension} x {5 * dimension}; five ancilla levels (3 encoded ancilla qubits)")
    print(f"Hamiltonian prefactor: {args.exchange_prefactor} J; target Z: {1 - 2 * args.p}")
    print(json.dumps(result["diagnostics"], indent=2))
    stem = f"{args.mode}_N{args.n_qubits}_p{args.p:g}_c{args.exchange_prefactor:g}"
    save_results(result, args.output_dir, stem, parameters)
    print(f"Saved density matrices, observables, diagnostics and plot: {args.output_dir / stem}")


if __name__ == "__main__":
    main()
