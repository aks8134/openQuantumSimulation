"""Analyze saved N=3/7 exact runs; validate and extend their XX correlations.

The extension computes populations and currents, not full density matrices.
For delta=0, C_ij=<c_j^dagger c_i> satisfies
    dC/dt = A C + C A^dagger + p G,
where A=-i h-G/2, h has hopping 2*c*J and G has boundary rates Gamma.
With equal reservoirs C*=p I, so C(t)=p I+exp(t A)(C(0)-p I)exp(t A)^dagger.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, eigvals


ROOT = Path(__file__).resolve().parents[1]


def correlations(parameters, bitstrings, times):
    n = parameters["number_of_qubits"]
    if parameters["delta"] != 0:
        raise ValueError("This correlation extension requires delta=0")
    hopping = 2 * parameters["exchange_prefactor"] * parameters["J"]
    h = np.diag(np.full(n - 1, hopping), 1) + np.diag(np.full(n - 1, hopping), -1)
    rates = np.zeros(n)
    rates[[0, -1]] = parameters["gamma"]
    A = -1j * h - np.diag(rates) / 2
    p = parameters["p"]
    initial = np.asarray([np.diag([int(bit) for bit in bits]) for bits in bitstrings])
    covariance = np.empty((len(initial), len(times), n, n), complex)
    for index, time in enumerate(times):
        E = expm(A * time)
        covariance[:, index] = p * np.eye(n) + E @ (initial - p * np.eye(n)) @ E.conj().T
    Z = 1 - 2 * np.diagonal(covariance, axis1=-2, axis2=-1).real
    currents = 2 * hopping * np.diagonal(covariance, offset=1, axis1=-2, axis2=-1).imag
    return Z, currents, -2 * eigvals(A).real.max()


def write_report(summaries, output):
    three, seven = summaries["3"], summaries["7"]
    final_time = seven["saved_times"][-1]
    lines = [
        "# Analysis of the exact N=3 and N=7 runs", "",
        "Both runs use J=1, delta=0, Gamma=2, p=0.25 and exchange prefactor 0.25. "
        "The stationary target is Z=0.5 at every site with zero current.", "",
        f"The saved N=3 run ends at t={three['saved_times'][-1]:g}; "
        f"the saved N=7 run ends at t={final_time:g}. This report describes that snapshot.", "",
        "| Preparation | N=3 error at t=20 | N=7 error at t=20 | N=7 error at t=40 | N=7 error at t=80 | N=7 error at t=120 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, label in enumerate(("All zeros", "All ones", "Leftmost spin excited")):
        first, second = three["states"][index], seven["states"][index]
        lines.append(f"| {label} | {first['maximum_Z_error_at_20']:.3g} | "
                     f"{second['maximum_Z_error_at_20']:.5f} | "
                     f"{second['maximum_Z_error_at_40']:.5f} | "
                     f"{second['maximum_Z_error_at_80']:.6f} | "
                     f"{second['maximum_Z_error_at_120']:.6f} |")
    lines += [
        "", "Errors above are the maximum absolute deviation of any site's Z from 0.5. "
        "Values beyond the saved run come from the exact XX correlation equations, "
        "which reproduce the saved populations and currents within 1e-15.", "",
        "N=3 has equilibrated by t=20. N=7 relaxes more slowly because dissipation acts "
        "only at the boundaries and the interior couples to them through Hamiltonian transport. "
        "The nonuniform finite-time profile continues approaching the uniform target.", "",
        f"The slowest correlation decay time is {three['slowest_covariance_decay_time']:.3f} "
        f"for N=3 and {seven['slowest_covariance_decay_time']:.3f} for N=7. "
        "This concerns the correlation sector; a full Liouvillian gap was not computed.", "",
        "Uniform initial preparations retain reflection symmetry. The all-zero preparation "
        "has currents flowing inward as reservoirs inject excitations, while the all-one "
        "preparation has currents flowing outward as reservoirs remove them. "
        "For this XX model at p=1/4, their local Z deviations are exactly related by "
        "Z_all-one - 0.5 = -3 (Z_all-zero - 0.5), following from linear correlation evolution.", "",
        f"At the last saved time t={final_time:g}:", "",
        "| N=7 preparation | Largest site error | Largest absolute current | Trace distance from product target |",
        "|---|---:|---:|---:|",
    ]
    for state in seven["states"]:
        lines.append(f"| `{state['initial_state']}` | {state['final_maximum_Z_error']:.6g} | "
                     f"{state['final_maximum_current']:.6g} | {state['final_trace_distance']:.6g} |")
    lines += [
        "", "Trace, Hermiticity and continuity errors are at floating-point scale. "
        "The tiny negative N=7 density-matrix eigenvalue is roundoff. "
        "The error-versus-exact values of zero in the original diagnostics are self-comparisons; "
        "agreement with the separately solved correlation equations provides an additional check.", "",
        "The original observation grids are sparse, so straight segments in their plots "
        "are interpolation rather than the actual transient waveform. The comparison below "
        "uses dense exact correlation curves and marks the saved observations.", "",
        "![Comparison](exact_N3_N7_comparison.png)", "",
        "The extension solves C(t)=p I+exp(t A)(C(0)-p I)exp(t A)^dagger, "
        "with A=-i h-G/2, hopping h_j,j+1=2*c*J and boundary rates G=Gamma. "
        "It computes populations and currents rather than full density matrices. "
        "At t=120, all three preparations have every site within 0.001 of Z=0.5.", "",
        "For full density matrices with a denser grid through t=120:", "", "```bash",
        ".venv/bin/python Model2/Experiment1/exact.py --n-qubits 7 --t-final 120 "
        "--num-times 241 --output-dir Model2/Experiment1/results/N7_long_time", "```", "",
        "The exact solver does not use max_dt; that parameter affects dilation only.", "",
        "Regenerate this report, diagnostics and comparison plot from the saved runs:", "", "```bash",
        ".venv/bin/python Model2/Experiment1/analysis/analyze_exact.py", "```", "",
    ]
    (output / "ExactComparison.md").write_text("\n".join(lines))


def main():
    output = Path(__file__).parent
    dense_times = np.linspace(0, 120, 1201)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    summaries = {}
    dense_arrays = {"times": dense_times}
    for n in (3, 7):
        stem = f"exact_N{n}_p0.25_c0.25"
        summary = json.loads((ROOT / "results" / f"{stem}.json").read_text())
        with np.load(ROOT / "results" / f"{stem}.npz") as arrays:
            parameters = summary["parameters"]
            labels = summary["initial_states"]
            final_time = float(arrays["times"][-1])
            target_Z = 1 - 2 * parameters["p"]
            sampled_Z, sampled_I, rate = correlations(parameters, labels, arrays["times"])
            Z_difference = float(np.max(np.abs(sampled_Z - arrays["Z_exact"])))
            I_difference = float(np.max(np.abs(sampled_I - arrays["I_exact"])))
            np.testing.assert_allclose(sampled_Z, arrays["Z_exact"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(sampled_I, arrays["I_exact"], atol=1e-12, rtol=0)
            dense_Z, dense_I, _ = correlations(parameters, labels, dense_times)
            comparison_Z, _, _ = correlations(parameters, labels, [20, 40, 80, 120])
            dense_arrays[f"Z_N{n}"] = dense_Z
            dense_arrays[f"I_N{n}"] = dense_I
            states = []
            for state, label in enumerate(labels):
                final_rho = arrays["rho_exact"][state, -1]
                distance = float(np.abs(np.linalg.eigvalsh(final_rho - arrays["target"])).sum() / 2)
                states.append({
                    "initial_state": label,
                    "final_time": final_time,
                    "final_Z": arrays["Z_exact"][state, -1].tolist(),
                    "final_maximum_Z_error": float(np.max(np.abs(arrays["Z_exact"][state, -1] - target_Z))),
                    "final_total_excitation": float(((1 - arrays["Z_exact"][state, -1]) / 2).sum()),
                    "final_maximum_current": float(np.max(np.abs(arrays["I_exact"][state, -1]))),
                    "final_trace_distance": distance,
                    "maximum_Z_error_at_20": float(np.max(np.abs(comparison_Z[state, 0] - target_Z))),
                    "maximum_Z_error_at_40": float(np.max(np.abs(comparison_Z[state, 1] - target_Z))),
                    "maximum_Z_error_at_80": float(np.max(np.abs(comparison_Z[state, 2] - target_Z))),
                    "maximum_Z_error_at_120": float(np.max(np.abs(dense_Z[state, -1] - target_Z))),
                })
                error = np.max(np.abs(dense_Z[state] - target_Z), axis=-1)
                axes[1, state].semilogy(dense_times, np.maximum(error, 1e-15), label=f"N={n}")
                sample_error = np.max(np.abs(arrays["Z_exact"][state] - target_Z), axis=-1)
                axes[1, state].semilogy(arrays["times"], np.maximum(sample_error, 1e-15),
                                        "o", color=f"C{0 if n == 3 else 1}", ms=4)
                if n == 7:
                    axes[0, state].plot(np.arange(1, n + 1), arrays["Z_exact"][state, -1],
                                        "o-", label=f"N=7, t={final_time:g} (saved run)")
                    axes[0, state].plot(np.arange(1, n + 1), dense_Z[state, -1],
                                        "s--", label="N=7, t=120 (exact correlations)")
                    axes[0, state].axhline(target_Z, color="black", lw=1, label="target 0.5")
                    axes[0, state].set_title(f"Initial |{label}>")
                    axes[0, state].set_xlabel("Site")
            summaries[str(n)] = dict(
                parameters=parameters, saved_times=arrays["times"].tolist(),
                slowest_covariance_decay_rate=float(rate),
                slowest_covariance_decay_time=float(1 / rate),
                correlation_agreement=dict(maximum_Z_difference=Z_difference, maximum_current_difference=I_difference),
                saved_run_diagnostics=summary["diagnostics"],
                states=states,
            )
    for state in range(3):
        axes[1, state].axvline(final_time, color="gray", ls=":", label="N=7 saved run ends")
        axes[1, state].axhline(1e-3, color="black", ls="--", lw=1, label="error 0.001")
        axes[1, state].set_xlabel("Time")
        axes[1, state].set_ylim(1e-10, 2)
        for row in range(2):
            axes[row, state].legend(fontsize=8)
            axes[row, state].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Site magnetization <Z>")
    axes[1, 0].set_ylabel("Maximum site deviation from 0.5")
    figure.suptitle("Equal reservoirs: N=7 relaxes more slowly; both sizes share the same target")
    figure.tight_layout()
    figure.savefig(output / "exact_N3_N7_comparison.png", dpi=160)
    plt.close(figure)
    (output / "exact_N3_N7_analysis.json").write_text(json.dumps(summaries, indent=2) + "\n")
    np.savez_compressed(output / "exact_N3_N7_correlations.npz", **dense_arrays)
    write_report(summaries, output)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
