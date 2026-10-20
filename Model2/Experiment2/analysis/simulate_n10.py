"""Simulate all N=10 population steps, then certify tolerance thresholds.

The XX covariance closure evaluates the exact ensemble mean of the actual
sampled Ding--Li--Lin channels, without a Gaussian-state approximation or
Monte Carlo sampling. Compare with continuous Lindblad covariances at every
step. Certificates use an independent real-gauge validated implementation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from Model2.Experiment2 import randomized
from Model2.Experiment2.analysis.population_bound import certificate
from Model2.Experiment2.analysis.tolerance_sweep import threshold, write_report


def simulate(steps):
    n, dt = 10, 12 / steps
    labels, initial = randomized.initial_covariances(n)
    branches = randomized.covariance_branches(n, 10, 20, .25, .25, dt,
                                               'rate-weighted', 'jump-first')
    counts = np.arange(steps + 1)
    times = counts * dt
    C, diagnostics = randomized.mean_trajectory(initial, branches, counts)
    exact = randomized.exact_covariances(initial, n, 10, 20, .25, .25, times)
    population, current = randomized.observables(C, 5)
    exact_population, exact_current = randomized.observables(exact, 5)
    error = np.abs(population - exact_population)
    state, step, site = np.unravel_index(error.argmax(), error.shape)
    # Independently iterate the original complex channel to check the late
    # endpoint and largest-error step, beyond the eigensolver's early check.
    direct = initial.copy()
    direct_difference = 0.0
    for m in range(steps + 1):
        if m in (int(step), steps):
            direct_difference = max(direct_difference, float(np.max(np.abs(direct - C[:, m]))))
        if m < steps:
            direct = randomized.apply_mean(direct, branches)
    if direct_difference > 1e-9:
        raise RuntimeError('Late direct channel check failed')
    diagnostics['late_direct_agreement'] = direct_difference
    measured = float(error.max())
    result = dict(n=n, steps=steps, dt=dt, observed_population_error=measured,
                  observed_normalized_current_error=float(np.abs(current-exact_current).max()/10),
                  largest_error=dict(initial_state=labels[state], site=int(site+1), step=int(step), time=float(times[step])),
                  exact_population_distance_from_steady_at_T=float(np.abs(exact_population[:, -1]-.25).max()),
                  simulated_population_distance_from_steady_at_T=float(np.abs(population[:, -1]-.25).max()),
                  diagnostics=diagnostics)
    arrays = dict(times=times, population=population, excitation_current=current,
                  exact_population=exact_population, exact_excitation_current=exact_current,
                  initial_labels=np.asarray(labels))
    return result, arrays


def plot_dynamics(output, strict):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with np.load(output / 'trajectories' / f"N10_M{strict['steps']}.npz") as arrays:
        times = arrays['times']
        population = arrays['population']
        exact = arrays['exact_population']
        labels = arrays['initial_labels']
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    colors = plt.get_cmap('tab10').colors
    for state, ax in enumerate(axes.flat[:3]):
        for site in range(10):
            ax.plot(times, exact[state, :, site], color=colors[site], lw=1)
            ax.plot(times, population[state, :, site], color=colors[site], lw=1, ls='--')
        ax.axhline(.25, color='gray', ls=':', lw=1)
        ax.set(title=f'Initial state {labels[state]}', xlabel='Time', ylabel='Excitation population',
               xlim=(0, 12), ylim=(-.025, 1.025))
        ax.grid(alpha=.2)
    axes[0, 0].legend(handles=[Line2D([], [], color='black', label='Exact Lindblad'),
                              Line2D([], [], color='black', ls='--', label='Mean dilation')], fontsize=8)
    ax = axes[1, 1]
    errors = np.max(np.abs(population-exact), axis=2)
    for state, label in enumerate(labels):
        ax.plot(times, errors[state], label=str(label))
    ax.axhline(.003, color='black', ls='--', label='Tolerance 0.003')
    ax.set(title='Early population error: maximum over sites', xlabel='Time',
           ylabel='Absolute population error', xlim=(0, .5))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    figure.suptitle(f"N=10, {strict['steps']:,} steps, dt={strict['dt']:.8g}; J=10, Gamma=20")
    figure.legend(handles=[Line2D([], [], color=colors[site], label=f'Site {site+1}') for site in range(10)],
                  loc='lower center', ncol=10, fontsize=8)
    figure.tight_layout(rect=(0, .055, 1, .95))
    figure.savefig(output / 'population_dynamics.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'N10_tolerance_sweep')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir
    saved_certificates = output / 'certificates'
    trajectories = output / 'trajectories'
    saved_certificates.mkdir(parents=True, exist_ok=True)
    trajectories.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / 'tolerance_sweep/sweep.json').read_text())['records']
    reference_n7 = sorted([r for r in reference if r['n'] == 7], key=lambda r: -r['tolerance'])
    cache = {}

    def evaluate(n, steps):
        if steps not in cache:
            path = trajectories / f'N10_M{steps}.json'
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record, arrays = simulate(steps)
                np.savez_compressed(trajectories / f'N10_M{steps}.npz', **arrays)
                path.write_text(json.dumps(record, indent=2)+'\n')
            cache[steps] = record
        record = cache[steps]
        # Numerical candidate search only: these equal point estimates are
        # replaced by actual independent bounds in the final report.
        return dict(record, population_upper_bound=record['observed_population_error'],
                    lower_witness=dict(error=record['observed_population_error']))

    numerical = []
    for row in reference_n7:
        tolerance, guess = row['tolerance'], row['steps']
        passing = evaluate(10, guess)
        failing = evaluate(10, guess-1)
        if not passing['observed_population_error'] <= tolerance < failing['observed_population_error']:
            passing, failing = threshold(10, tolerance, evaluate, guess)
        result = dict(cache[passing['steps']], tolerance=tolerance,
                      preceding_steps=failing['steps'],
                      preceding_observed_population_error=failing['observed_population_error'])
        numerical.append(result)
        (output / 'simulation.json').write_text(json.dumps(numerical, indent=2)+'\n')
        print(json.dumps(dict(phase='simulation', tolerance=tolerance, steps=result['steps'],
                              observed_error=result['observed_population_error'],
                              preceding_error=result['preceding_observed_population_error'])), flush=True)

    needed = sorted({m for r in numerical for m in (r['steps'], r['preceding_steps'])})
    certified = {}
    # Threads also work in restricted environments where multiprocessing's
    # forkserver cannot create a local socket. Certificate inputs are immutable.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for steps in needed:
            path = saved_certificates / f'N10_M{steps}.json'
            if path.exists():
                certified[steps] = json.loads(path.read_text())
            else:
                future = executor.submit(certificate, 10, steps, tolerance=0,
                                         block_steps=1, degree=1, bernstein_degree=1)
                pending[future] = steps
        for future in as_completed(pending):
            steps = pending[future]
            result = future.result()
            certified[steps] = result
            (saved_certificates / f'N10_M{steps}.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(phase='certificate', steps=steps,
                                  upper=result['population_upper_bound'],
                                  lower=result['lower_witness']['error'])), flush=True)

    records = list(reference)
    for result in numerical:
        tolerance = result['tolerance']
        passing = certified[result['steps']]
        failing = certified[result['preceding_steps']]
        if not passing['population_upper_bound'] <= tolerance < failing['lower_witness']['error']:
            raise RuntimeError('Numerical threshold requires certificate refinement')
        for cert in (passing, failing):
            specific = dict(cert, tolerance=tolerance,
                            certified_population=cert['population_upper_bound'] <= tolerance,
                            certified_violation=cert['lower_witness']['error'] > tolerance)
            (saved_certificates / f"eps{tolerance:g}_N10_M{cert['steps']}.json").write_text(json.dumps(specific, indent=2)+'\n')
        observed = result['observed_population_error']
        # Independently computed observed errors must lie inside the validated
        # envelope; also validate the preceding simulation against its bounds.
        for steps in (result['steps'], result['preceding_steps']):
            measured = cache[steps]['observed_population_error']
            cert = certified[steps]
            if not cert['lower_witness']['error'] <= measured <= cert['population_upper_bound']:
                raise RuntimeError('Simulation disagrees with validated enclosure')
        records.append(dict(tolerance=tolerance, n=10, steps=result['steps'], dt=result['dt'],
                            population_upper_bound=passing['population_upper_bound'],
                            preceding_steps=result['preceding_steps'],
                            preceding_population_lower_bound=failing['lower_witness']['error'],
                            logical_unitary_factors=2*result['steps'], observed_population_error=observed,
                            observed_preceding_population_error=result['preceding_observed_population_error']))
    write_report(records, output)
    columns = ['tolerance', 'steps', 'dt', 'observed_population_error', 'preceding_steps',
               'preceding_observed_population_error', 'observed_normalized_current_error',
               'exact_population_distance_from_steady_at_T', 'simulated_population_distance_from_steady_at_T']
    with (output / 'simulation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key:r[key] for key in columns} for r in numerical)
    strict = next(r for r in numerical if r['tolerance'] == .003)
    plot_dynamics(output, strict)
    _, initial_n7 = randomized.initial_covariances(7)
    exact_n7 = randomized.exact_covariances(initial_n7, 7, 10, 20, .25, .25, [12])
    n7_population, _ = randomized.observables(exact_n7, 5)
    n7_distance = float(np.abs(n7_population-.25).max())
    (output / 'equilibration_comparison.json').write_text(json.dumps(dict(
        T=12, stationary_population=.25, maximum_population_distance=dict(
            N7=n7_distance, N10=strict['exact_population_distance_from_steady_at_T'])), indent=2)+'\n')
    lines = ['', '## Independent N=10 simulation', '',
             'Every numerical check evaluates the ensemble-mean populations and excitation '
             'currents at all algorithm steps against the exact continuous XX covariance '
             'solution. A direct complex-channel iteration independently checks the '
             'largest-error step and final endpoint. No shot noise or finite random-seed '
             'sampling is included. The covariance closure is exact for these observables; '
             'it does not reconstruct the complete density matrix.', '',
             '![N=10 population dynamics and early errors](population_dynamics.png)', '',
             f"At tolerance 0.003, M={strict['steps']:,} gives measured maximum population "
             f"error {strict['observed_population_error']:.12g}. M-1 gives "
             f"{strict['preceding_observed_population_error']:.12g}. The peak is at "
             f"{strict['largest_error']}. All 13 selected counts coincide with N=7." if all(
                 a['steps'] == b['steps'] for a,b in zip(numerical,reference_n7)) else
             'See simulation.csv for measured errors at each selected and preceding count.', '',
             f"At T=12 the exact N=10 population differs from the steady value 0.25 by up to "
             f"{strict['exact_population_distance_from_steady_at_T']:.8g}. Thus the time-step "
             'accuracy criterion does not imply equilibration at this time. '
             f"For comparison, N=7 differs by at most {n7_distance:.8g} at the same time.", '',
             'N=10 certificates are saved here; N=3/N=7 certificates remain in '
             '[the previous sweep](../tolerance_sweep/Results.md). '
             '[Measured N=10 errors](simulation.csv) and per-count population/current '
             'trajectories in `trajectories/` are saved separately.', '',
             'Reproduce from the repository root:', '', '```bash',
             'MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
             '  .venv/bin/python Model2/Experiment2/analysis/simulate_n10.py --workers 2',
             '```', '']
    with (output / 'Results.md').open('a') as handle:
        handle.write('\n'.join(lines))
    print('All N=10 simulation results agree with their certificates.', flush=True)


if __name__ == '__main__':
    main()
