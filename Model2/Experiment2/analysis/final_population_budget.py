"""Optimize a fixed 100-reset randomized DLL channel for final populations.

This optimizes final-state preparation, not fidelity to continuous Lindblad
dynamics. Results are ensemble means, not individual random circuits or shots.
The finite search is not a proof of a global optimum or impossibility.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Model2.Experiment2 import randomized


class FinalPopulationProblem:
    def __init__(self, n=20, steps=100, p=.25):
        self.n, self.steps, self.p = n, steps, p
        h = np.diag(np.full(n-1, .5), 1) + np.diag(np.full(n-1, .5), -1)
        self.values, self.vectors = eigh(h)
        self.phase = (1j)**np.arange(n)
        self.evaluations = 0

    def unitary(self, x):
        complex_u = (self.vectors*np.exp(-1j*x*self.values))@self.vectors.T
        return (self.phase.conj()[:, None]*complex_u*self.phase[None, :]).real

    def trajectory(self, x, cosine):
        """Powers of the centered channel on I, in a real fermion gauge.

        Rate-weighted sampling makes all four dilation angles equal. On
        C-pI, averaging pump/loss at a chosen boundary removes the source;
        the mean map is U [0.5*(R_L D R_L+R_R D R_R)] U^T.
        Starting from I gives the exact worst population envelope of the
        three preparations because the map preserves positive semidefiniteness.
        """
        U = self.unitary(x)
        left, right = np.ones(self.n), np.ones(self.n)
        left[0], right[-1] = cosine, cosine
        multiplier = .5*(np.outer(left, left)+np.outer(right, right))
        D = np.eye(self.n)
        distances = [max(self.p, 1-self.p)]
        for _ in range(self.steps):
            D = U@(multiplier*D)@U.T
            distances.append(max(self.p, 1-self.p)*np.diag(D).max())
        return D, np.asarray(distances)

    def objective(self, parameters):
        self.evaluations += 1
        return float(self.trajectory(*parameters)[1][-1])


def validate(problem, x, cosine):
    theta = np.arccos(cosine)
    branches = randomized.covariance_branches(
        problem.n, x, theta**2/2, problem.p, .25, 1.,
        'rate-weighted', 'jump-first')
    labels, initial = randomized.initial_covariances(problem.n)
    C = initial.copy()
    for _ in range(problem.steps):
        C = randomized.apply_mean(C, branches)
    D, distances = problem.trajectory(x, cosine)
    expected_ones = problem.p + (1-problem.p)*np.diag(D)
    discrepancies = np.max(np.abs(np.diag(C[1]).real-expected_ones))
    assert discrepancies < 2e-11, discrepancies
    errors = np.max(np.abs(np.diagonal(C, axis1=-2, axis2=-1).real-problem.p), axis=1)
    assert abs(errors.max()-distances[-1]) < 2e-11
    return dict(reference_agreement=float(discrepancies),
                final_errors_by_initial_state=dict(zip(labels, errors.tolist())),
                final_populations=np.diagonal(C, axis1=-2, axis2=-1).real.tolist(),
                stationary_covariance_residual=float(np.max(np.abs(
                    randomized.apply_mean(problem.p*np.eye(problem.n), branches)
                    -problem.p*np.eye(problem.n)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-search', action='store_true',
                        help='Revalidate and regenerate outputs from saved search results')
    args = parser.parse_args()
    output = Path(__file__).parent/'N20_final_population_budget'
    problem = FinalPopulationProblem()
    records = []
    if args.reuse_search:
        records = json.loads((output/'search.json').read_text())['searches']
    for upper in (() if args.reuse_search else (10., 100., 1000.)):
        for seed in (24, 71, 109):
            result = differential_evolution(problem.objective,
                [(.01, upper), (-1., 1.)], seed=seed,
                popsize=20, maxiter=160, tol=1e-9, polish=True)
            x, cosine = result.x
            row = dict(search_upper_J_tau=upper, seed=seed, J_tau=float(x),
                cosine=float(cosine), gamma_tau=float(np.arccos(cosine)**2/2),
                final_population_error=float(result.fun), evaluations=int(result.nfev))
            records.append(row)
            print(json.dumps(row), flush=True)
    best = min(records, key=lambda r:r['final_population_error'])
    validation = validate(problem, best['J_tau'], best['cosine'])
    # Several unrelated angles check the reduced formula beyond the optimizer.
    checks = []
    for n, x, cosine in ((3, .8, .7), (7, 7.3, -.4), (20, 23.4288781238, 0.)):
        checks.append(dict(n=n, J_tau=x, cosine=cosine,
            **validate(FinalPopulationProblem(n=n), x, cosine)))
    output.mkdir(parents=True, exist_ok=True)
    tau = best['gamma_tau']/20
    extended = FinalPopulationProblem(steps=3000).trajectory(best['J_tau'], best['cosine'])[1]
    passing = np.flatnonzero(extended <= .001)
    physical_parameters = dict(gamma=20., tau=tau, J=best['J_tau']/tau)
    report = dict(n=20, steps=100, p=.25, tolerance=.001,
        method='Rate-weighted independently randomized single-jump DLL; jump first; exact Hamiltonian factor',
        criterion='Final population error, all sites and all three initial preparations',
        meaning='Exact ensemble channel expectation, up to floating-point precision',
        search_limitations='Constant steps and parameters; finite numerical search, not a global impossibility proof',
        parameterization='x=J*tau, cosine=cos(sqrt(2*Gamma*tau)); first angle period covers every distinct channel',
        searches=records, best=best, validation=validation, independent_checks=checks,
        equivalent_physical_parameters=physical_parameters,
        first_passing_step_at_best_100_step_setting=int(passing[0]) if passing.size else None,
        passes_tolerance=best['final_population_error'] <= .001)
    (output/'search.json').write_text(json.dumps(report, indent=2)+'\n')
    D, distances = problem.trajectory(best['J_tau'], best['cosine'])
    np.savez_compressed(output/'best_trajectory.npz', steps=np.arange(101),
        population_distance=distances, final_centered_covariance=D)
    lines = ['# N=20 final-population preparation with 100 DLL resets', '',
        'Method: independently rate-weighted randomized single-jump Ding–Li–Lin dilation, '
        'jump first, followed by the exact XX Hamiltonian factor. p=1/4.', '',
        'Only the final populations are optimized. No transient accuracy is required. '
        'The results describe the exact ensemble mean up to floating-point roundoff; '
        'they do not include finite random-sequence sampling or measurement shots.', '',
        'The three preparations are all zeros, all ones, and a leftmost excitation. '
        'Positive-semidefinite ordering proves that the all-ones preparation realizes '
        'the largest population error at every step.', '',
        f"Best found: J*tau={best['J_tau']:.12g}, Gamma*tau={best['gamma_tau']:.12g}, "
        f"cos(theta)={best['cosine']:.12g}, final error={best['final_population_error']:.12g}.", '',
        f"Required error: 0.001. Pass: {report['passes_tolerance']}.", '',
        f"Equivalent physical parameters at Gamma=20: tau={tau:.12g}, "
        f"J={physical_parameters['J']:.12g}. This setting fails the 100-step target.", '',
        f"Continuing this same channel reaches 0.001 at step "
        f"{report['first_passing_step_at_best_100_step_setting']}. "
        'This is a measured count for that setting, not a minimum over all settings.', '',
        f"Reference implementation agreement: {validation['reference_agreement']:.3g}.", '',
        '| Upper J*tau | Seed | Best J*tau | Gamma*tau | Final population error |',
        '|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['search_upper_J_tau']:g} | {row['seed']} | {row['J_tau']:.8g} | "
                     f"{row['gamma_tau']:.8g} | {row['final_population_error']:.8g} |")
    lines += ['', 'This search does not establish a global minimum or rule out '
        'other DLL formulas, nonconstant timesteps, or engineered schedules. '
        'These large rotation angles are a final-state preparation experiment, '
        'not a validated approximation to the Lindblad transient.', '',
        'Uniformly scaling J and Gamma and inversely scaling tau leaves the channel '
        'unchanged and cannot improve this result.', '']
    (output/'Results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(best=best, validation=validation, report=str(output/'Results.md'))), flush=True)


if __name__ == '__main__':
    main()
