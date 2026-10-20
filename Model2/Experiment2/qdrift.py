"""Benchmark randomized single-jump dilation, including sequence fluctuations."""

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Model2.Experiment1.analysis.analyze_exact import correlations
from Model2.Experiment2 import randomized as method


def observation_counts(dt, t_final, dense_until=1.0):
    total = round(t_final / dt)
    if not np.isclose(total * dt, t_final, rtol=0, atol=1e-11):
        raise ValueError("t-final must be an integer multiple of each tested dt")
    # Every actual step during the early transient; every ~0.1 thereafter.
    early = np.arange(min(total, int(dense_until / dt)) + 1)
    late = np.rint(np.arange(dense_until, t_final, 0.1) / dt).astype(int)
    return np.unique(np.clip(np.r_[early, late, total], 0, total))


def reference(parameters, initial, times):
    return method.exact_covariances(initial, parameters["number_of_qubits"],
                                   parameters["J"], parameters["gamma"],
                                   parameters["p"], parameters["exchange_prefactor"], times)


def branches(parameters, dt, policy, order):
    return method.covariance_branches(parameters["number_of_qubits"], parameters["J"],
                                     parameters["gamma"], parameters["p"],
                                     parameters["exchange_prefactor"], dt, policy, order)


def compatible_baseline(parameters, t_final):
    path = ROOT / "Model2/Experiment1/analysis/strang_step_selection/comparison.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if any(parameters[key] != data["parameters"][key] for key in parameters):
        return []
    if t_final != data["times"][-1]:
        return []
    return [dict(method="shared-ancilla-lie", dt=record["max_dt"],
                 population_error=record["maximum_population_error"],
                 normalized_current_error=record["maximum_current_error_divided_by_J"],
                 resets=record["total_steps"], logical_unitary_factors=5 * record["total_steps"],
                 ancilla_qubits=3,
                 stationary_population_bias=record["stationary_maximum_population_bias"])
            for record in data["records"] if record["formula"] == "lie"]


def plot_comparison(records, baseline, output):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for policy in method.POLICIES:
        for order in method.ORDERS:
            group = sorted([r for r in records if r["policy"] == policy and r["order"] == order], key=lambda r: r["dt"])
            label = f"{policy}, {'H first' if order == 'hamiltonian-first' else 'jump first'}"
            axes[0].loglog([r["dt"] for r in group], [r["population_error"] for r in group], "o-", label=label, ms=4)
            axes[1].loglog([r["dt"] for r in group], [r["normalized_current_error"] for r in group], "o-", label=label, ms=4)
            axes[2].loglog([r["logical_unitary_factors"] for r in group], [r["population_error"] for r in group], "o-", label=label, ms=4)
    if baseline:
        baseline = sorted(baseline, key=lambda r: r["dt"])
        axes[0].loglog([r["dt"] for r in baseline], [r["population_error"] for r in baseline], "ks--", label="Shared-ancilla Lie", ms=4)
        axes[1].loglog([r["dt"] for r in baseline], [r["normalized_current_error"] for r in baseline], "ks--", label="Shared-ancilla Lie", ms=4)
        axes[2].loglog([r["logical_unitary_factors"] for r in baseline], [r["population_error"] for r in baseline], "ks--", label="Shared-ancilla Lie", ms=4)
    for index, axis in enumerate(axes):
        axis.axhline(0.001, color="gray", ls=":", label="Tolerance 0.001")
        if index != 1:
            axis.axhline(0.0005, color="gray", ls="--", label="Population margin 0.0005")
        axis.grid(alpha=0.25)
    axes[0].set(xlabel="Step size", ylabel="Maximum population error")
    axes[1].set(xlabel="Step size", ylabel="Maximum excitation-current error / |J|")
    axes[2].set(xlabel="Logical unitary factors per sequence", ylabel="Maximum population error")
    for axis in axes[:2]:
        axis.xaxis.set_major_locator(FixedLocator([0.000125, 0.0005, 0.002, 0.005]))
        axis.set_xticklabels(["1.25e-4", "5e-4", "2e-3", "5e-3"])
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=7)
    figure.suptitle("Average-channel accuracy and logical cost; measurement shots excluded")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=170)
    plt.close(figure)


def plot_sequences(times, mean_population, exact_population, population, labels, p, output):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, site in enumerate((0, population.shape[-1] // 2)):
        for state, axis in enumerate(axes[row]):
            for sequence in range(min(8, population.shape[1])):
                axis.plot(times, population[state, sequence, :, site], color="gray", alpha=0.35, lw=0.8)
            axis.plot(times, exact_population[state, :, site], "k--", label="Exact Lindblad")
            axis.plot(times, mean_population[state, :, site], color="C0", label="Exact sequence average")
            axis.plot(times, population[state, :, :, site].mean(axis=0), color="C1", label=f"{population.shape[1]}-sequence estimate")
            axis.axhline(p, color="gray", ls=":")
            axis.set(xlabel="Time", title=f"Initial |{labels[state]}>")
            axis.grid(alpha=0.25)
        axes[row, 0].set_ylabel(f"Excitation population, site {site + 1}")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "random_sequences.png", dpi=170)
    plt.close(figure)


def write_report(summary, output):
    pars = summary["parameters"]
    lines = ["# Randomized single-jump Ding–Li–Lin dilation", "",
             f"N={pars['number_of_qubits']}, J={pars['J']:g}, Gamma={pars['gamma']:g}, "
             f"p={pars['p']:g}, Delta=0, exchange prefactor={pars['exchange_prefactor']:g}, "
             f"T={summary['t_final']:g}. This is a numerical-method experiment on the "
             "same equal-reservoir problem as Experiment 1.", "",
             "The implementation follows the randomized Lindbladian framework and "
             "single-ancilla first-order implementation in Appendix B of "
             "[Chen, Li, Lu and Ying](https://arxiv.org/html/2407.06594v5). "
             "Its general convergence results distinguish the mean channel from one "
             "sampled realization. We benchmark observables here; we do not certify "
             "diamond-norm accuracy or apply the paper's Gibbs-sampler guarantees.", "",
             "## Construction", "",
             r"Write $\mathcal L=-i[H,\cdot]+\sum_{a=1}^4\mathcal D[V_a]$. "
             r"Draw $a$ independently with probability $\mu_a$ and use "
             r"$\mathcal L_a=-i[H,\cdot]+\mathcal D[V_a/\sqrt{\mu_a}]$. "
             r"Then $\sum_a\mu_a\mathcal L_a=\mathcal L$; the Hamiltonian is "
             "applied once per step, with its original strength.", "",
             r"The sampled interaction is $W_a=\exp[-i\sqrt{\tau}" 
             r"(|1\rangle\langle0|\otimes V_a/\sqrt{\mu_a}+" 
             r"|0\rangle\langle1|\otimes V_a^\dagger/\sqrt{\mu_a})]$. "
             "Prepare the ancilla in zero, apply this interaction and the separate "
             r"$U_H=\exp(-i\tau H)$ factor, trace/reset the ancilla. "
             "Both factor orders are tested. No exact boundary-reservoir exponential is used.", "",
             "Uniform sampling uses mu=(1/4,1/4,1/4,1/4). Rate-weighted sampling uses "
             "mu=(p/2,(1-p)/2,p/2,(1-p)/2), in left-pump, left-loss, right-pump, right-loss order.", "",
             r"For jump rate $r_a$, the boundary rotation angle is "
             r"$\theta_a=\sqrt{r_a\tau/\mu_a}$, and its transition probability is "
             r"$\sin^2\theta_a$, not $1-e^{-r_a\tau/\mu_a}$. "
             "Rate weighting equalizes all four angles. In the mean channel, this "
             "makes the pumping/loss transition ratio exactly p/(1-p), so the "
             "product target is fixed for either factor order. Individual sequences "
             "do not fix that target. Uniform sampling instead selects "
             r"$p_{\rm eff}=\mu_+\sin^2\theta_+/(\mu_+\sin^2\theta_++\mu_-\sin^2\theta_-)$.", "",
             "## Measured accuracy", "",
             "Maximum errors below cover all three initial preparations, all sites/bonds, "
             "every step through t=1, and observations approximately 0.1 apart afterwards. "
             "Late observations are rounded to actual step boundaries; no fractional "
             "steps are inserted. These are sampled-time errors, not uniform-time bounds. "
             "The previous Lie baseline checks every step through t=0.1 and a dense "
             "observation grid afterwards; its known maximum is in that early window.", "",
             "| Sampling | Order | Step | Population error | Current error / abs(J) | Steps | Logical factors |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for record in sorted(summary["records"], key=lambda r: (r["policy"], r["order"], -r["dt"])):
        lines.append(f"| {record['policy']} | {record['order']} | {record['dt']:g} | "
                     f"{record['population_error']:.6g} | {record['normalized_current_error']:.6g} | "
                     f"{record['resets']} | {record['logical_unitary_factors']} |")
    lines += ["", "Selection criteria: population error <=0.001 and current error/abs(J) "
              "<=0.001. The recommendation additionally requires population error <=0.0005.", "",
              "```json", json.dumps(summary["step_selections"], indent=2), "```", "",
              "![Mean error and cost](comparison.png)", "",
              "## Cost and sequence fluctuations", "",
              "A sampled step uses one Hamiltonian and one jump-interaction factor, one "
              "ancilla qubit, and one reset cycle. The previous shared-ancilla Lie step "
              "uses one Hamiltonian and four jump-interaction factors, three binary ancilla "
              "qubits encoding five levels, and one reset cycle. Counts are logical "
              "factors per sequence, not compiled gate counts, circuit depth, total "
              "measurement cost, or a hardware speedup. The full Hamiltonian factor is "
              "treated as an exact oracle in both methods.", ""]
    selection = summary["step_selections"]["rate-weighted/jump-first"]
    recommendation = selection["recommended_tested_step"]
    lines += [f"For rate-weighted jump-first sampling, the largest passing tested step "
              f"is {selection['largest_passing_tested_step']}; the largest tested step "
              f"with the additional population margin is {recommendation}.", ""]
    baseline_margin = [r for r in summary["lie_baseline"] if
                       r["population_error"] <= 0.0005 and r["normalized_current_error"] <= 0.001]
    if recommendation is not None and baseline_margin:
        old = max(baseline_margin, key=lambda r: r["dt"])
        new = next(r for r in summary["records"] if r["policy"] == "rate-weighted" and
                   r["order"] == "jump-first" and r["dt"] == recommendation)
        lines += [f"At that population margin, the old Lie dt={old['dt']:g} uses "
                  f"{old['resets']:,} steps and {old['logical_unitary_factors']:,} logical factors; "
                  f"the randomized dt={new['dt']:g} uses {new['resets']:,} steps and "
                  f"{new['logical_unitary_factors']:,} factors. This is "
                  f"{100 * (1 - new['resets'] / old['resets']):.1f}% fewer reset cycles and "
                  f"{old['logical_unitary_factors'] / new['logical_unitary_factors']:.2f} times "
                  "fewer logical factors per sequence.", ""]
    if summary["samples"] is not None:
        samples = summary["samples"]
        lines += [f"The {samples['realizations']}-sequence Monte Carlo mean has maximum "
                  f"population error {samples['maximum_sample_mean_population_error_vs_exact']:.4g}, "
                  f"while the largest measured sequence standard deviation is "
                  f"{samples['maximum_population_sequence_standard_deviation']:.4g}. "
                  "That finite collection of random sequences does not reach the 0.001 "
                  "accuracy target, even though the analytically evaluated ensemble does. "
                  "This is a material distinction for the cost comparison.", ""]
        lines += ["Random-sequence diagnostics (conditional channel expectations, no shot noise):", "",
                  "```json", json.dumps(summary["samples"], indent=2), "```", "",
                  "![Random sequences](random_sequences.png)", "",
                  "Reusing one random sequence across many shots measures that sequence's "
                  "channel. Resample the entire sequence per measurement shot to estimate "
                  "the average channel directly. For population measurements, the marginal "
                  "outcome remains Bernoulli with the averaged population, so the observed "
                  "sequence variance is not automatically an extra multiplier in the "
                  "fundamental shot cost. Averaging conditional channel expectations over "
                  "a fixed, small collection of seeds is a different estimator and can "
                  "have substantial sampling error. Standard errors here are pointwise, "
                  "not simultaneous confidence bounds over every observation.", ""]
    lines += ["## Numerical validation and scope", "",
              "XX covariance evolution exactly closes for these boundary channels, "
              "including non-Gaussian ensemble states. We use powers of its n^2-dimensional "
              "affine map to evaluate the exact sequence average to roundoff; this is "
              "a classical evaluation shortcut, not an extra quantum implementation. "
              "We do not reconstruct full density matrices or certify trace distances. "
              "The fast runner requires Delta=0 and equal reservoirs. Full two-level "
              "ancilla Kraus construction is also available in randomized.sampled_kraus.", "",
              "Tests compare all four branches with full density-matrix dilation on a "
              "coherent mixed three-spin state, enumerate all 16 two-step sequences, "
              "check the complete product target, and compare continuous reference "
              "covariances with the full Liouvillian. Eigenchannel powers are checked "
              "against 100 direct steps and their decomposition residual. Residual "
              "bounds concern exact-arithmetic propagation of the measured residual; "
              "they do not rigorously bound all floating-point operations.", "",
              "## Reproduce", "", "```bash",
              "MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
              f"  .venv/bin/python Model2/Experiment2/qdrift.py --n-qubits {pars['number_of_qubits']} \\",
              f"  --J {pars['J']:g} --gamma {pars['gamma']:g} --p {pars['p']:g} --t-final {summary['t_final']:g} \\",
              "  --realizations 64", "```", ""]
    (output / "Results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=7)
    parser.add_argument("--J", type=float, default=10)
    parser.add_argument("--gamma", type=float, default=20)
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--exchange-prefactor", type=float, default=0.25)
    parser.add_argument("--t-final", type=float, default=12)
    parser.add_argument("--dt", type=float, nargs="+", default=[0.005, 0.002, 0.001, 0.00075, 0.000625, 0.0005, 0.0004, 0.000375, 0.0003125, 0.00025, 0.000125])
    parser.add_argument("--sample-dt", type=float)
    parser.add_argument("--realizations", type=int, default=64, help="0 to skip sequence sampling; otherwise >=2")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.t_final <= 0 or not np.isfinite(args.t_final) or args.realizations == 1 or args.realizations < 0:
        parser.error("require positive finite t-final and realizations=0 or >=2")
    parameters = dict(number_of_qubits=args.n_qubits, J=args.J, delta=0.0,
                      gamma=args.gamma, p=args.p, exchange_prefactor=args.exchange_prefactor)
    output = args.output_dir or Path(__file__).parent / "results" / f"N{args.n_qubits}_J{args.J:g}_G{args.gamma:g}"
    output.mkdir(parents=True, exist_ok=True)
    labels, initial = method.initial_covariances(args.n_qubits)
    validation_times = np.array([0, 0.0357, 0.42, args.t_final])
    test_C = reference(parameters, initial, validation_times)
    test_pop, test_current = method.observables(test_C, 2 * args.exchange_prefactor * args.J)
    Z, current, _ = correlations(parameters, labels, validation_times)
    agreement = dict(population=float(np.max(np.abs(test_pop - (1 - Z) / 2))),
                     current=float(np.max(np.abs(test_current - current))))
    if max(agreement.values()) > 1e-10:
        raise RuntimeError("continuous XX reference validation failed")
    records = []
    started = time.perf_counter()
    for policy in method.POLICIES:
        for order in method.ORDERS:
            for dt in sorted(set(args.dt), reverse=True):
                channel = branches(parameters, dt, policy, order)
                counts = observation_counts(dt, args.t_final)
                times = counts * dt
                C, diagnostics = method.mean_trajectory(initial, channel, counts)
                population, currents = method.observables(C, channel["hopping"])
                exact_population, exact_current = method.observables(reference(parameters, initial, times), channel["hopping"])
                pop_error = np.abs(population - exact_population)
                worst_state, worst_time, worst_site = np.unravel_index(np.argmax(pop_error), pop_error.shape)
                record = dict(policy=policy, order=order, dt=dt, resets=int(counts[-1]),
                              logical_unitary_factors=2 * int(counts[-1]), ancilla_qubits=1,
                              probabilities=channel["mu"].tolist(),
                              population_error=float(pop_error.max()),
                              normalized_current_error=float(np.max(np.abs(currents - exact_current)) / abs(args.J)),
                              stationary_population=channel["effective_p"],
                              stationary_population_bias=abs(channel["effective_p"] - args.p),
                              final_maximum_population_deviation_from_target=float(np.max(np.abs(population[:, -1] - args.p))),
                              worst_population=dict(initial_state=labels[worst_state], time=float(times[worst_time]), site=int(worst_site + 1)),
                              validation=diagnostics)
                records.append(record)
                np.savez_compressed(output / f"{policy}_{order}_dt{dt:g}.npz", times=times,
                                    population=population, current=currents,
                                    exact_population=exact_population, exact_current=exact_current)
                print(f"{policy:13s} {order:17s} dt={dt:g}: pop={record['population_error']:.6g}, I/J={record['normalized_current_error']:.6g}", flush=True)
    selections = {}
    for policy in method.POLICIES:
        for order in method.ORDERS:
            group = [r for r in records if r["policy"] == policy and r["order"] == order]
            passing = [r["dt"] for r in group if r["population_error"] <= 0.001 and r["normalized_current_error"] <= 0.001]
            margin = [r["dt"] for r in group if r["population_error"] <= 0.0005 and r["normalized_current_error"] <= 0.001]
            selections[f"{policy}/{order}"] = dict(largest_passing_tested_step=max(passing, default=None),
                                                   recommended_tested_step=max(margin, default=None))
    baseline = compatible_baseline(parameters, args.t_final)
    sample_summary = None
    if args.realizations:
        sample_dt = args.sample_dt or selections["rate-weighted/jump-first"]["recommended_tested_step"] or min(args.dt)
        channel = branches(parameters, sample_dt, "rate-weighted", "jump-first")
        counts = observation_counts(sample_dt, args.t_final, dense_until=0.1)
        times = counts * sample_dt
        C, _ = method.mean_trajectory(initial, channel, counts)
        mean_population, mean_current = method.observables(C, channel["hopping"])
        exact_population, exact_current = method.observables(reference(parameters, initial, times), channel["hopping"])
        print(f"Sampling {args.realizations} independent sequences at dt={sample_dt:g} ...", flush=True)
        population, currents, sample_summary = method.sample_trajectories(initial, channel, counts, args.realizations, args.seed, "jump-first")
        std_pop = population.std(axis=1, ddof=1)
        std_current = currents.std(axis=1, ddof=1)
        sample_summary.update(dt=sample_dt, policy="rate-weighted", order="jump-first",
                              maximum_population_sequence_standard_deviation=float(std_pop.max()),
                              maximum_population_standard_error=float(std_pop.max() / np.sqrt(args.realizations)),
                              maximum_normalized_current_sequence_standard_deviation=float(std_current.max() / abs(args.J)),
                              maximum_sample_mean_population_error_vs_ensemble=float(np.max(np.abs(population.mean(axis=1) - mean_population))),
                              maximum_sample_mean_population_error_vs_exact=float(np.max(np.abs(population.mean(axis=1) - exact_population))),
                              final_population_standard_deviations=std_pop[:, -1].tolist(),
                              pointwise_seed_count_estimate_for_population_standard_error_0p001=int(np.ceil((std_pop.max() / 0.001)**2)))
        np.savez_compressed(output / "random_sequences.npz", times=times, population=population,
                            current=currents, ensemble_population=mean_population,
                            ensemble_current=mean_current, exact_population=exact_population,
                            exact_current=exact_current)
        plot_sequences(times, mean_population, exact_population, population, labels, args.p, output)
    summary = dict(parameters=parameters, t_final=args.t_final, initial_states=labels,
                   reference_agreement=agreement, records=records, lie_baseline=baseline,
                   step_selections=selections, samples=sample_summary,
                   elapsed_seconds=time.perf_counter() - started)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_comparison(records, baseline, output)
    write_report(summary, output)
    print(json.dumps(dict(step_selections=selections, samples=sample_summary, output_dir=str(output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
