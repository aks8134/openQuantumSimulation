"""Find and certify population-tolerance step counts for the fixed XX benchmark.

Candidate counts use interpolation of certificate results, not saved simulation
errors. Every selected count has an upper certificate and the preceding count
has a lower violating witness. This locates adjacent thresholds, without claiming
global monotonicity or optimality over all smaller counts.
"""

import argparse
import csv
import json
from math import ceil, floor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .population_bound import certificate
except ImportError:
    from population_bound import certificate


TOLERANCES = [.1, .08, .06, .05, .04, .03, .02, .015, .01, .0075, .005, .004, .003]
ROOT = Path(__file__).resolve().parent


def threshold(n, tolerance, evaluate, guess):
    """Search an adjacent certified pass/violation pair using reciprocal M."""
    low = high = None
    candidate = max(100, guess)
    for _ in range(80):
        result = evaluate(n, candidate)
        upper = result['population_upper_bound']
        lower = result['lower_witness']['error']
        if upper <= tolerance:
            high = (candidate, result)
        elif lower > tolerance:
            low = (candidate, result)
        else:
            raise RuntimeError(f'Unresolved arithmetic interval at N={n}, M={candidate}, eps={tolerance}')
        if low and high:
            if high[0] - low[0] == 1:
                return high[1], low[1]
            fraction = (low[1]['lower_witness']['error'] - tolerance) / (
                low[1]['lower_witness']['error'] - high[1]['population_upper_bound'])
            candidate = floor(1 / ((1 - fraction) / low[0] + fraction / high[0]))
            candidate = min(high[0] - 1, max(low[0] + 1, candidate))
        elif high:
            candidate = min(high[0] - 1, floor(high[0] * upper / tolerance))
            candidate = max(100, candidate)
            if candidate == high[0]:
                raise RuntimeError('Search reached its 100-step lower limit')
        else:
            candidate = max(low[0] + 1, ceil(low[0] * upper / tolerance))
    raise RuntimeError('Step-count search did not converge')


def write_report(records, output):
    columns = ['tolerance', 'n', 'steps', 'dt', 'population_upper_bound',
               'preceding_steps', 'preceding_population_lower_bound', 'logical_unitary_factors']
    with (output / 'sweep.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in records)
    metadata = dict(parameters=dict(J=10, Gamma=20, p=.25, Delta=0, T=12, hamiltonian_prefactor=.25),
                    method='rate-weighted jump-first Ding-Li-Lin dilation, ensemble mean',
                    certificate='full degree-one binomial polynomial, one-step blocks; outward arithmetic',
                    scope='all sites, all integer steps through T=12, preparations 000..., 111..., 100...',
                    records=records)
    (output / 'sweep.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lookup = {(r['n'], r['tolerance']): r for r in records}
    ns = sorted({r['n'] for r in records})
    tolerances = sorted({r['tolerance'] for r in records}, reverse=True)
    lines = ['# Population-error tolerance versus certified steps', '',
             'Fixed parameters: J=10, Gamma=20, p=1/4, Delta=0, T=12; H=(J/4) sum(XX+YY). '
             'Rate-weighted, jump-first randomized Ding–Li–Lin dilation.', '',
             'Each selected M has a certified upper bound below its requested tolerance. '
             'At M-1 a certified observation exceeds that tolerance. These are adjacent '
             'thresholds found by search; global monotonicity and optimality over all smaller '
             'counts are not proved.', '',
             '| Population tolerance | ' + ' | '.join(f'N={n} steps | N={n} dt' for n in ns) + ' |',
             '|---:|' + '---:|---:|' * len(ns)]
    for tolerance in tolerances:
        lines.append(f'| {tolerance:g} | ' + ' | '.join(
            (f"{lookup[n, tolerance]['steps']:,} | {lookup[n, tolerance]['dt']:.8g}"
             if (n, tolerance) in lookup else '— | —') for n in ns) + ' |')
    lines += ['', '![Tolerance versus steps](tolerance_steps.png)', '',
              'Smaller population-error budgets require more steps. For the selected settings, '
              'logical cost per sequence is 2M unitary factors and M ancilla resets, using one '
              'ancillary qubit. This excludes transpilation and measurement shots.', '',
              'The bound covers the ensemble-mean population bias for all sites, all three '
              'benchmark preparations, and every algorithm step through T=12. It does not '
              'bound individual sequence fluctuations or finite-shot errors.', '',
              'The certificates use the same validated maps and contraction propagation as '
              '[the derivation](../Proof.md). Here one-step blocks use the complete degree-one '
              'binomial polynomial. Its endpoints are the actual algorithm steps, so the '
              'Bernstein enclosure introduces no time-interpolation excess. Arithmetic radii '
              'and matrix Taylor remainders remain included under the stated IEEE assumptions.', '',
              'The cosine remainder now allows larger positive angles: only the omitted '
              'alternating terms must decrease. The physical reservoir channel has not changed.', '',
              'CSV contains the upper bound and the preceding count\'s lower witness. '
              'Full selected/preceding certificates are saved in `certificates/`.', '']
    (output / 'Results.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, n in enumerate(ns):
        group = sorted([r for r in records if r['n'] == n], key=lambda r: r['tolerance'], reverse=True)
        errors = [r['tolerance'] for r in group]
        steps = [r['steps'] for r in group]
        for ax in axes:
            ax.plot(errors, steps, ['o-', 's--', '^:'][index % 3], label=f'N={n}', markersize=4)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[0].set_title('Full tolerance range')
    axes[0].set_xticks([.1, .08, .06, .04, .02, .003], ['0.1', '0.08', '0.06', '0.04', '0.02', '0.003'])
    axes[1].set_title('Log scale: tighter tolerances')
    axes[1].set_xticks([.1, .05, .02, .01, .005, .003], ['0.1', '0.05', '0.02', '0.01', '0.005', '0.003'])
    for ax in axes:
        ax.set(xlabel='Allowed maximum population error', ylabel='Certified steps to T=12')
        ax.invert_xaxis()
        ax.grid(alpha=.25, which='both')
        ax.legend()
    fig.suptitle('Rate-weighted randomized Ding–Li–Lin dilation')
    fig.tight_layout()
    fig.savefig(output / 'tolerance_steps.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-qubits', type=int, nargs='+', default=[3, 7])
    parser.add_argument('--tolerances', type=float, nargs='+', default=TOLERANCES)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tolerance_sweep')
    args = parser.parse_args()
    output = args.output_dir
    certificates = output / 'certificates'
    certificates.mkdir(parents=True, exist_ok=True)
    cache = {}

    def evaluate(n, steps):
        key = n, steps
        if key not in cache:
            path = certificates / f'N{n}_M{steps}.json'
            if path.exists():
                result = json.loads(path.read_text())
            else:
                result = certificate(n, steps, tolerance=0, block_steps=1, degree=1, bernstein_degree=1)
                path.write_text(json.dumps(result, indent=2) + '\n')
            cache[key] = result
        return cache[key]

    records = []
    for n in args.n_qubits:
        previous = None
        for tolerance in sorted(set(args.tolerances), reverse=True):
            if tolerance <= 0:
                raise ValueError('Tolerances must be positive')
            guess = ceil(16.3 / tolerance) if previous is None else ceil(previous['steps'] * previous['tolerance'] / tolerance)
            passing, failing = threshold(n, tolerance, evaluate, guess)
            row = dict(tolerance=tolerance, n=n, steps=passing['steps'], dt=passing['dt'],
                       population_upper_bound=passing['population_upper_bound'],
                       preceding_steps=failing['steps'], preceding_population_lower_bound=failing['lower_witness']['error'],
                       logical_unitary_factors=2 * passing['steps'])
            records.append(row)
            previous = row
            # Add tolerance-specific decisions to the selected pair. Other
            # cached counts retain tolerance=0, since they serve several budgets.
            for original in (passing, failing):
                specific = dict(original, tolerance=tolerance,
                                certified_population=original['population_upper_bound'] <= tolerance,
                                certified_violation=original['lower_witness']['error'] > tolerance)
                (certificates / f"eps{tolerance:g}_N{n}_M{original['steps']}.json").write_text(json.dumps(specific, indent=2) + '\n')
            write_report(records, output)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
