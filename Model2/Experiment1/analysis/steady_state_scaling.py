"""Exact population settling times for the equal-reservoir XX chain.

For C(t)-pI=E(t)(C(0)-pI)E(t)^dagger and E=exp(tA), the worst
population deviation of the three preparations is
max(p,1-p) max_j sum_k |E_jk|^2. It is monotone: E(t+s)=E(t)E(s)
and E(s) is a contraction. A threshold crossing therefore stays below
the threshold for all later times. This is exact Lindblad covariance
evolution, not a time-discretized dilation or full-state distance test.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, eigvals


ROOT = Path(__file__).resolve().parent


def generator(n, J=10., gamma=20., prefactor=.25):
    if n < 2 or not isinstance(n, (int, np.integer)):
        raise ValueError('N must be an integer of at least 2')
    if not np.all(np.isfinite([J, gamma, prefactor])) or J == 0 or gamma <= 0 or prefactor <= 0:
        raise ValueError('Invalid model parameters')
    hopping = 2 * prefactor * J
    h = np.diag(np.full(n-1, hopping), 1) + np.diag(np.full(n-1, hopping), -1)
    rates = np.zeros(n)
    rates[[0, -1]] = gamma
    return -1j*h - np.diag(rates)/2


def populations(E, p=.25):
    """Three preparations in order: all zeros, all ones, left spin excited."""
    row_norms_squared = np.sum(np.abs(E)**2, axis=1)
    return np.asarray([p-p*row_norms_squared,
                       p+(1-p)*row_norms_squared,
                       p-p*row_norms_squared+np.abs(E[:, 0])**2])


def population_distance(E, p=.25):
    return float(max(p, 1-p) * np.max(np.sum(np.abs(E)**2, axis=1)))


def settling_time(n, *, J=10., gamma=20., p=.25, prefactor=.25, tolerance=1e-3,
                  time_resolution=1e-9):
    if not np.isfinite(p) or not 0 < p < 1 or not 0 < tolerance < max(p, 1-p):
        raise ValueError('Require 0<p<1 and positive tolerance below the initial deviation')
    if not np.isfinite(time_resolution) or time_resolution <= 0:
        raise ValueError('Positive finite time resolution required')
    A = generator(n, J, gamma, prefactor)
    low, high = 0., 1./max(abs(J), gamma)
    while population_distance(expm(high*A), p) > tolerance:
        low, high = high, 2*high
    while high-low > time_resolution:
        middle = (low+high)/2
        if population_distance(expm(middle*A), p) > tolerance:
            low = middle
        else:
            high = middle
    E = expm(high*A)
    values = populations(E, p)
    row_norms_squared = np.sum(np.abs(E)**2, axis=1)
    labels = ['0'*n, '1'*n, '1'+'0'*(n-1)]
    gaps = -2*eigvals(A).real
    return dict(n=n, settling_time=high, lower_time=low, upper_time=high,
                bracket_width=high-low, population_error_at_lower=population_distance(expm(low*A), p),
                population_error_at_upper=population_distance(E, p),
                errors_by_initial_state=dict(zip(labels, np.max(np.abs(values-p), axis=1).tolist())),
                worst_initial_state=labels[1 if p <= .5 else 0],
                largest_error_site=int(np.argmax(row_norms_squared)+1),
                slowest_covariance_decay_rate=float(gaps.min()),
                population_distance_at_t12=population_distance(expm(12*A), p))


def write_outputs(records, output, parameters):
    output.mkdir(parents=True, exist_ok=True)
    columns = ['n', 'settling_time', 'ten_times_slower_rate_settling_time',
               'lower_time', 'upper_time', 'population_error_at_lower', 'population_error_at_upper',
               'slowest_covariance_decay_rate', 'population_distance_at_t12']
    for row in records:
        row['ten_times_slower_rate_settling_time'] = 10*row['settling_time']
    with (output/'steady_state_times.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key:r[key] for key in columns} for r in records)
    fit_group = [r for r in records if r['n'] >= 6]
    fit = None
    if len(fit_group) >= 3:
        slope, intercept = np.polyfit(np.log([r['n'] for r in fit_group]),
                                     np.log([r['settling_time'] for r in fit_group]), 1)
        fit = dict(exponent=float(slope), coefficient=float(np.exp(intercept)),
                   fitted_N_range=[fit_group[0]['n'], fit_group[-1]['n']],
                   interpretation='Empirical fit to exact settling times; not an asymptotic theorem')
    summary = dict(parameters=parameters, tolerance=parameters['tolerance'],
                   criterion='Maximum absolute population distance from p across all sites and three preparations, for every later time',
                   method='Continuous Lindblad XX covariance matrix exponential; floating-point numerical precision',
                   secondary_rate_set=dict(J=parameters['J']/10, gamma=parameters['gamma']/10),
                   empirical_power_fit=fit, records=records)
    (output/'steady_state_times.json').write_text(json.dumps(summary, indent=2)+'\n')
    ns = [r['n'] for r in records]
    times = [r['settling_time'] for r in records]
    figure, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(ns, times, 'o-', color='C0', ms=5, label='Exact population settling time')
    ax.axhline(12, color='gray', ls='--', lw=1, label='Previous observation horizon T=12')
    ax.set(xlabel='Chain length N', ylabel=f"Settling time: J={parameters['J']:g}, Gamma={parameters['gamma']:g}",
           title=f"Exact XX relaxation to population tolerance {parameters['tolerance']:g}")
    ax.set_xticks(ns)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(loc='upper left')
    secondary = ax.secondary_yaxis('right', functions=(lambda t:10*t, lambda t:t/10))
    secondary.set_ylabel(f"Settling time: J={parameters['J']/10:g}, Gamma={parameters['gamma']/10:g}")
    for row in records:
        if row['n'] in (7, 10, 20):
            ax.annotate(f"{row['settling_time']:.2f}", (row['n'], row['settling_time']),
                        xytext=(-6, 10), textcoords='offset points', ha='right', fontsize=9)
    figure.tight_layout()
    figure.savefig(output/'steady_state_time_vs_N.png', dpi=200)
    figure.savefig(output/'steady_state_time_vs_N.pdf')
    plt.close(figure)
    selected = [r for r in records if r['n'] in (3, 7, 10, 15, 20)]
    arrays = {}
    figure, ax = plt.subplots(figsize=(8.5, 5))
    for row in selected:
        n, crossing = row['n'], row['settling_time']
        A = generator(n, parameters['J'], parameters['gamma'], parameters['exchange_prefactor'])
        ts = np.geomspace(crossing*1e-4, crossing*1.25, 301)
        errors = np.asarray([population_distance(expm(t*A), parameters['p']) for t in ts])
        arrays[f'times_N{n}'], arrays[f'population_distance_N{n}'] = ts, errors
        line, = ax.loglog(ts, errors, label=f'N={n}')
        ax.plot(crossing, parameters['tolerance'], 'o', color=line.get_color(), ms=5)
    ax.axhline(parameters['tolerance'], color='black', ls='--', label='Convergence threshold')
    ax.set(xlabel='Time', ylabel='Maximum population distance from steady state',
           title='Exact relaxation curves', ylim=(parameters['tolerance']/10, 1))
    ax.grid(which='both', alpha=.2)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output/'population_convergence.png', dpi=180)
    plt.close(figure)
    np.savez_compressed(output/'population_convergence.npz', **arrays)
    lines = ['# Exact steady-state population convergence versus N', '',
             f"Parameters: J={parameters['J']:g}, Gamma={parameters['gamma']:g}, "
             f"p={parameters['p']:g}, Delta=0, H=(J/4) sum(XX+YY).", '',
             'Convergence means every site is within the population tolerance, for all '
             'three benchmark preparations, and remains within it at all later times:', '',
             r'$$t_{\rm ss}(N)=\inf\{t\ge0:\max_{s,j}|\langle n_j(u)\rangle_s-p|\le\varepsilon\ \text{for all }u\ge t\},\qquad\varepsilon=10^{-3}.$$', '',
             'This is a population convergence criterion. The full density-matrix distance '
             'and excitation current are not part of its threshold.', '',
             '![Steady-state time versus N](steady_state_time_vs_N.png)', '',
             '| N | Settling time | Time with J and Gamma both divided by 10 |', '|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['n']} | {row['settling_time']:.6f} | {10*row['settling_time']:.6f} |")
    if fit:
        lines += ['', f"A log-log fit over N={fit['fitted_N_range'][0]}–{fit['fitted_N_range'][1]} "
                  f"gives t_ss approximately {fit['coefficient']:.5g} N^{fit['exponent']:.5g}. "
                  'This describes the computed range and is not a universal scaling law.']
    lines += ['', '## Exact reduction and permanent convergence', '',
              r'For the XX model, $C_{ij}=\langle c_j^\dagger c_i\rangle$ obeys', '',
              r'$$C(t)=pI+E(t)[C(0)-pI]E(t)^\dagger,\quad E(t)=e^{tA},\quad A=-ih-\frac\Gamma2(\Pi_1+\Pi_N),\quad h_{j,j+1}=J/2.$$', '',
              'The three initial covariance matrices are 0, I, and diag(1,0,...,0). '
              'At p=1/4, the all-ones preparation gives the largest population deviation '
              'for every time. Consequently the exact worst-case deviation is', '',
              r'$$d_N(t)=\frac34\max_j\sum_k|E_{jk}(t)|^2.$$', '',
              r'Since $A+A^\dagger=-\Gamma(\Pi_1+\Pi_N)\le0$, $E(s)$ is a contraction. '
              r'The relation $E(t+s)=E(t)E(s)$ makes each row norm nonincreasing. '
              'Thus d_N(t) is nonincreasing: the root is permanent convergence, without '
              'a later recrossing. Bisection brackets each threshold crossing to 1e-9 '
              'in time; the reported value is the upper endpoint.', '',
              'Only an N-by-N matrix exponential is required. These populations are exact '
              'continuous Lindblad observables up to floating-point numerical error. '
              'There is no Trotter step or dilation approximation and no reconstruction '
              'of the exponentially large density matrix. The reduction requires Delta=0.', '',
              'Dividing both J and Gamma by 10 divides the entire generator by 10. '
              'Every settling time is therefore exactly multiplied by 10. The right '
              'plot axis and CSV provide that second rate convention.', '',
              '![Exact convergence curves](population_convergence.png)', '',
              'A population tolerance of 1e-3 corresponds to a Z-magnetization tolerance '
              'of 2e-3. A Z tolerance of 1e-3 would require population tolerance 5e-4.', '',
              'Reproduce:', '', '```bash',
              'MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
              '  .venv/bin/python Model2/Experiment1/analysis/steady_state_scaling.py', '```', '']
    (output/'Results.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-min', type=int, default=2)
    parser.add_argument('--n-max', type=int, default=20)
    parser.add_argument('--J', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=20)
    parser.add_argument('--p', type=float, default=.25)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'steady_state_scaling')
    args = parser.parse_args()
    if args.n_min < 2 or args.n_max < args.n_min:
        parser.error('Require 2 <= n-min <= n-max')
    parameters = dict(J=args.J, gamma=args.gamma, p=args.p, delta=0,
                      exchange_prefactor=.25, tolerance=args.tolerance)
    records = []
    for n in range(args.n_min, args.n_max+1):
        result = settling_time(n, J=args.J, gamma=args.gamma, p=args.p, tolerance=args.tolerance)
        records.append(result)
        print(json.dumps(result), flush=True)
    write_outputs(records, args.output_dir, parameters)
    print(args.output_dir/'Results.md', flush=True)


if __name__ == '__main__':
    main()
