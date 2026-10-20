"""Measure the shared-ancilla Lie dilation's step error against saved exact data.

The optimized propagation preserves the same channel: the system unitary
acts within excitation-number sectors, and the four jump Kraus operators
are local bit flips with diagonal survival factors. A one-step comparison
against the original enlarged-Hamiltonian implementation checks equivalence.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Model2.Experiment1 import simulation
from Model2.Experiment1.analysis.analyze_exact import correlations


def prepare_step(H, gamma, p, dt, n):
    dimension = 2**n
    indices = np.arange(dimension)
    left = (indices >> (n - 1)) & 1
    right = indices & 1
    theta_plus = np.sqrt(gamma * p * dt)
    theta_minus = np.sqrt(gamma * (1 - p) * dt)
    survival_left = np.where(left, np.cos(theta_minus), np.cos(theta_plus))
    survival_right = np.where(right, np.cos(theta_minus), np.cos(theta_plus))
    survival = survival_left * survival_right
    transitions = []
    for mask, bits, factor in ((1 << (n - 1), left, np.ones(dimension)),
                               (1, right, survival_left)):
        for source_bit, theta in ((0, theta_plus), (1, theta_minus)):
            source = indices[bits == source_bit]
            target = source ^ mask
            amplitude = np.sin(theta) * factor[source]
            transitions.append((source, target, np.outer(amplitude, amplitude)))
    h = H.toarray()
    sectors = []
    for count in range(n + 1):
        sector = np.asarray([i for i in indices if int(i).bit_count() == count])
        unitary = expm(-1j * dt * h[np.ix_(sector, sector)])
        sectors.append((sector, unitary))
    return sectors, np.outer(survival, survival), transitions


def apply_step(rhos, prepared):
    sectors, survival, transitions = prepared
    coherent = np.zeros_like(rhos)
    for indices, U in sectors:
        coherent[:, indices[:, None], indices] = U @ rhos[:, indices[:, None], indices] @ U.conj().T
    evolved = coherent * survival
    for source, target, weight in transitions:
        evolved[:, target[:, None], target] += coherent[:, source[:, None], source] * weight
    return evolved


def write_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    data = json.loads((output / "sweep.json").read_text())
    parameters = data["parameters"]
    records = [dict(record) for record in data["records"]]
    early_path = output / "early_transient_check.json"
    early = json.loads(early_path.read_text()) if early_path.exists() else None
    if isinstance(early, dict) and early.get("parameters") == parameters:
        checks = {record["dt"]: record for record in early["records"]}
        for record in records:
            check = checks.get(record["max_dt"])
            if check and check["maximum_population_error_first_0p1"] > record["maximum_population_error"]:
                record["maximum_population_error"] = check["maximum_population_error_first_0p1"]
                record["maximum_Z_error"] = 2 * record["maximum_population_error"]
    population_tolerance = 0.001
    normalized_current_tolerance = 0.001
    passing = [r for r in records if r["maximum_population_error"] <= population_tolerance
               and r["maximum_current_error_divided_by_J"] <= normalized_current_tolerance]
    conservative = [r for r in passing if r["maximum_population_error"] <= population_tolerance / 2]
    selected = max(conservative or passing, key=lambda r: r["max_dt"]) if passing else None
    lines = [
        "# Lie dilation step selection", "",
        f"Parameters: N={parameters['number_of_qubits']}, J={parameters['J']:g}, "
        f"Gamma={parameters['gamma']:g}, p={parameters['p']:g}, delta={parameters['delta']:g}, "
        f"exchange prefactor={parameters['exchange_prefactor']:g}, final time={data['times'][-1]:g}.", "",
        "The error criterion is maximum absolute population error <=0.001 and maximum "
        "absolute error in I/J <=0.001, over all sites/bonds, the three initial states, "
        "and the tested observation grid. A factor-of-two margin in population error "
        "is used for the recommendation when a tested step achieves it.", "",
        "The tested grid resolves t=0 through t=1 every 0.01 and includes every saved "
        "exact observation time thereafter. This is a sampled error assessment, not "
        "a rigorous uniform-in-time or arbitrary-initial-state bound.", "",
        "When present, early_transient_check.json additionally samples every actual Lie "
        "step through t=0.1. The population and Z errors below include that finer check.", "",
        "| Maximum step | Number of steps | Max population error | Max Z error | Max current error | Max current error / J | Stationary population bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(f"| {r['max_dt']:g} | {r['total_steps']} | {r['maximum_population_error']:.6g} | "
                     f"{r['maximum_Z_error']:.6g} | {r['maximum_current_error']:.6g} | "
                     f"{r['maximum_current_error_divided_by_J']:.6g} | "
                     f"{r['constant_step_stationary_population_bias']:.6g} |")
    if selected:
        dt = selected["max_dt"]
        lines += ["", f"Recommended tested step: **delta t={dt:g}**. "
                  f"This uses {selected['total_steps']} ancilla trace/reset steps through the final time.", "",
                  "```bash", ".venv/bin/python Model2/Experiment1/lie_trotter.py \\",
                  f"  --n-qubits {parameters['number_of_qubits']} --J {parameters['J']:g} --gamma {parameters['gamma']:g} \\",
                  f"  --p {parameters['p']:g} --delta {parameters['delta']:g} --exchange-prefactor {parameters['exchange_prefactor']:g} \\",
                  f"  --t-final {data['times'][-1]:g} --num-times 121 --max-dt {dt:g} \\",
                  "  --output-dir Model2/Experiment1/results/N7_lie_step_selected", "```", "",
                  f"At the final time, the largest population deviation from the reservoir target is "
                  f"{selected['final_maximum_population_deviation_from_target']:.6g}. "
                  "Deviation from the target includes both finite-time relaxation and numerical channel error.", ""]
    lines += [
        "The shared-ancilla Lie formula contains both finite-step dilation error and "
        "unitary ordering error; all quoted transient errors compare the complete channel "
        "against exact Lindblad evolution, not merely against the unsplit dilation.", "",
        "For constant steps, define q_plus=sin^2(sqrt(Gamma*p*dt)) and "
        "q_minus=sin^2(sqrt(Gamma*(1-p)*dt)). The uniform product state with "
        "population p_eff=q_plus/(q_plus+q_minus) is exactly stationary for this Lie channel. "
        "It differs from the physical target p. Checking the plateau alone can underestimate "
        "the larger transient error.", "",
        "Scaling J and Gamma by a common factor s requires dividing dt by s to preserve "
        "the same dimensionless dilation step and accuracy. Thus dt=0.00025 for J=10, "
        "Gamma=20 corresponds to dt=0.0025 for J=1, Gamma=2. The number of steps needed "
        "for corresponding final times is unchanged.", "",
        "The sweep propagates excitation-number blocks and applies the analytical bit-flip "
        "Kraus actions of the same Lie channel. One-step and two-step comparisons with "
        "the original enlarged-Hamiltonian implementation pass at floating-point precision. "
        "The exact XX correlation reference is also checked against the saved full "
        "density-matrix run. This optimization applies to the number-block-diagonal "
        "states used in this benchmark.", "",
        "![Step errors](step_errors.png)", "",
        "Reproduce the sweep:", "", "```bash",
        ".venv/bin/python Model2/Experiment1/analysis/lie_step_sweep.py", "```", "",
    ]
    (output / "StepSelection.md").write_text("\n".join(lines))
    steps = np.asarray([r["max_dt"] for r in records])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(steps, [r["maximum_population_error"] for r in records], "o-", label="Population")
    axes[0].loglog(steps, [r["maximum_current_error_divided_by_J"] for r in records], "s-", label="Current / J")
    axes[0].axhline(0.001, color="black", ls="--", label="Tolerance 0.001")
    axes[0].set_ylabel("Maximum absolute error vs exact")
    axes[0].set_title("Transient errors across all three initial states")
    axes[1].loglog(steps, [r["constant_step_stationary_population_bias"] for r in records], "o-", label="Stationary bias")
    axes[1].loglog(steps, [r["final_maximum_population_deviation_from_target"] for r in records], "s-", label="Deviation from p at t=12")
    axes[1].set_ylabel("Population deviation from target")
    axes[1].set_title("Equilibration and finite-step bias")
    for ax in axes:
        ax.set_xlabel("Physical step delta t")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    figure.savefig(output / "step_errors.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", type=Path, default=ROOT / "Model2/Experiment1/results/N7_J10_G20")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.005, 0.002, 0.001, 0.0005, 0.00025])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "lie_step_selection")
    args = parser.parse_args()
    summary_path = args.exact_dir / "exact_N7_p0.25_c0.25.json"
    summary = json.loads(summary_path.read_text())
    params = summary["parameters"]
    n = params["number_of_qubits"]
    if params["delta"] != 0:
        parser.error("The dense-time exact correlation reference requires delta=0")
    H, jumps, Zops, _, Iops, _ = simulation.build_model(
        n, params["J"], params["delta"], params["gamma"], params["p"], params["exchange_prefactor"],
    )
    labels, initial = simulation.initial_states(n)
    with np.load(summary_path.with_suffix(".npz")) as saved:
        saved_times = saved["times"].copy()
        reference_Z, reference_I, _ = correlations(params, labels, saved_times)
        reference_Z_error = float(np.max(np.abs(reference_Z - saved["Z_exact"])))
        reference_I_error = float(np.max(np.abs(reference_I - saved["I_exact"])))
        np.testing.assert_allclose(reference_Z, saved["Z_exact"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(reference_I, saved["I_exact"], atol=1e-11, rtol=0)
    # Resolve the early transient more finely than the saved 0.1 spacing.
    times = np.unique(np.round(np.r_[np.arange(0, min(1, saved_times[-1]) + 0.0001, 0.01), saved_times], 12))
    exact_Z, exact_I, _ = correlations(params, labels, times)
    # Also verify a mixed state reached after a coherent/jump step, not only inputs.
    first = prepare_step(H, params["gamma"], params["p"], args.steps[0], n)
    rhos = apply_step(initial, first)
    kraus = simulation.dilation_kraus(H, jumps, args.steps[0], "lie")
    expected = sum(K @ initial @ K.conj().T for K in kraus)
    one_step_error = float(np.max(np.abs(rhos - expected)))
    np.testing.assert_allclose(rhos, expected, atol=2e-13, rtol=0)
    np.testing.assert_allclose(apply_step(rhos, first), sum(K @ rhos @ K.conj().T for K in kraus), atol=2e-13, rtol=0)
    Zdiag = np.asarray([operator.diagonal().real for operator in Zops])
    current_matrices = np.asarray([operator.toarray() for operator in Iops])
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for max_dt in args.steps:
        if not np.isfinite(max_dt) or max_dt <= 0:
            parser.error("steps must be finite and positive")
        started = time.monotonic()
        density = initial.copy()
        Z = np.empty_like(exact_Z)
        I = np.empty_like(exact_I)
        previous = 0.0
        cache = {}
        step_count_total = 0
        for index, t in enumerate(times):
            interval = t - previous
            if interval > 0:
                count = max(1, int(np.ceil(interval / max_dt - 1e-10)))
                dt = interval / count
                key = round(dt, 14)
                if key not in cache:
                    cache[key] = prepare_step(H, params["gamma"], params["p"], dt, n)
                for _ in range(count):
                    density = apply_step(density, cache[key])
                step_count_total += count
            Z[:, index] = np.diagonal(density, axis1=-2, axis2=-1).real @ Zdiag.T
            I[:, index] = np.einsum("sij,bji->sb", density, current_matrices).real
            previous = t
        qplus = np.sin(np.sqrt(params["gamma"] * params["p"] * max_dt))**2
        qminus = np.sin(np.sqrt(params["gamma"] * (1 - params["p"]) * max_dt))**2
        effective_p = qplus / (qplus + qminus)
        errors = np.abs(Z - exact_Z)
        worst_index = np.unravel_index(np.argmax(errors), errors.shape)
        record = dict(max_dt=max_dt, total_steps=step_count_total,
                      maximum_population_error=float(errors.max() / 2),
                      maximum_Z_error=float(errors.max()),
                      worst_Z_error_initial_state=labels[worst_index[0]],
                      worst_Z_error_time=float(times[worst_index[1]]),
                      worst_Z_error_site=int(worst_index[2] + 1),
                      maximum_current_error=float(np.max(np.abs(I - exact_I))),
                      maximum_current_error_divided_by_J=float(np.max(np.abs(I - exact_I)) / abs(params["J"])),
                      final_maximum_population_error_vs_exact=float(errors[:, -1].max() / 2),
                      final_maximum_population_deviation_from_target=float(np.max(np.abs((1 - Z[:, -1]) / 2 - params["p"]))),
                      final_populations=((1 - Z[:, -1]) / 2).tolist(),
                      constant_step_stationary_population=float(effective_p),
                      constant_step_stationary_population_bias=float(abs(effective_p - params["p"])),
                      final_trace_error=float(np.max(np.abs(np.trace(density, axis1=-2, axis2=-1) - 1))),
                      final_minimum_eigenvalue=float(np.linalg.eigvalsh(density).min()),
                      final_hermiticity_error=float(np.max(np.abs(density - density.conj().swapaxes(-1, -2)))),
                      elapsed_seconds=time.monotonic() - started)
        records.append(record)
        np.savez_compressed(args.output_dir / f"lie_dt{max_dt:g}.npz", times=times, Z_lie=Z, I_lie=I,
                            Z_exact=exact_Z, I_exact=exact_I, final_rho=density)
        result = dict(parameters=params, initial_states=labels, times=times.tolist(),
                      exact_reference_Z_agreement=reference_Z_error, exact_reference_current_agreement=reference_I_error,
                      optimized_channel_one_step_agreement=one_step_error, records=records)
        (args.output_dir / "sweep.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    write_report(args.output_dir)


if __name__ == "__main__":
    main()
