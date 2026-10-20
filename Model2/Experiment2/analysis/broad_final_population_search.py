"""Dense and multistart parameter search for a 100-reset N=20 DLL circuit.

Gamma=20 and J<=100. The rate-weighted independently randomized jump channel
is unchanged. Both Hamiltonian/jump orders and repeated reservoir angle
cycles are included. Search results are ensemble means, not shot guarantees.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Model2.Experiment2 import randomized
from Model2.Experiment2.analysis.final_population_budget import FinalPopulationProblem


class SymmetricProblem:
    """Exact reflection reduction of the mean covariance map to a 10x10 block.

    In reflection parity sectors, H+ has hopping 1/2 and last diagonal 1/2;
    H- is related to -H+ by a staggered gauge. Starting from I, the parity
    covariances are conjugate, up to that gauge, with equal populations.
    The mean jump map damps the first-row coherences by (1+c)/2 and the
    first diagonal by (1+c^2)/2. This is not an approximation or a new channel.
    """
    def __init__(self, steps=100, p=.25):
        h = np.diag(np.full(9, .5), 1)+np.diag(np.full(9, .5), -1)
        h[-1, -1] = .5
        self.values, self.vectors = eigh(h)
        self.steps, self.p = steps, p
        self.evaluations = 0

    def batch(self, xs, cs):
        xs, cs = np.broadcast_arrays(np.atleast_1d(xs), np.atleast_1d(cs))
        self.evaluations += xs.size
        U = (self.vectors[None]*np.exp(-1j*xs[:, None, None]*self.values[None, None]))@self.vectors.T
        Ud = U.conj().swapaxes(-1, -2)
        multiplier = np.ones(U.shape, dtype=float)
        multiplier[:, 0, :] = (1+cs[:, None])/2
        multiplier[:, :, 0] = (1+cs[:, None])/2
        multiplier[:, 0, 0] = (1+cs*cs)/2
        D = np.broadcast_to(np.eye(10, dtype=complex), U.shape).copy()
        for step in range(self.steps):
            reservoir = multiplier*D
            if step == self.steps-1:
                h_first = np.diagonal(reservoir, axis1=-2, axis2=-1).real.max(axis=1)
            D = U@reservoir@Ud
        jump_first = np.diagonal(D, axis1=-2, axis2=-1).real.max(axis=1)
        return max(self.p, 1-self.p)*np.column_stack((jump_first, h_first))

    def objective(self, z):
        return float(self.batch(z[0], z[1]).min())


def physical_parameters(x, cosine, gamma=20., max_J=100.):
    """Shortest angle cycle realizing this channel while respecting J<=max_J."""
    angle = float(np.arccos(np.clip(cosine, -1, 1)))
    required = np.sqrt(2*gamma*x/max_J)
    possibilities = []
    for sign in (1, -1):
        k = max(0, int(np.ceil((required-sign*angle)/(2*np.pi))))
        theta = sign*angle+2*np.pi*k
        if theta > 0 and theta+1e-12 >= required:
            possibilities.append((theta, k, sign))
    theta, cycles, sign = min(possibilities)
    tau = theta*theta/(2*gamma)
    return dict(J=x/tau, gamma=gamma, tau=tau, theta=theta,
                angle_cycle=cycles, angle_sign=sign, t_final=100*tau)


def reference_validation(x, cosine, order, physical):
    labels, C = randomized.initial_covariances(20)
    branches = randomized.covariance_branches(20, physical['J'], 20., .25, .25,
        physical['tau'], 'rate-weighted', order)
    for _ in range(100):
        C = randomized.apply_mean(C, branches)
    population = np.diagonal(C, axis1=-2, axis2=-1).real
    return dict(final_population_error=float(np.max(np.abs(population-.25))),
        errors_by_initial_state=dict(zip(labels, np.max(np.abs(population-.25), axis=1).tolist())),
        final_populations=population.tolist(), theta_cosine_agreement=float(abs(np.cos(branches['theta'][0])-cosine)),
        fixed_point_residual=float(np.max(np.abs(randomized.apply_mean(.25*np.eye(20), branches)-.25*np.eye(20)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'N20_broad_final_population_search')
    parser.add_argument('--reuse-scans', action='store_true')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    problem = SymmetricProblem()
    start = time.monotonic()
    # Validate reflection reduction against the original N-by-N reduction.
    for x,c in ((.5,0.),(3.14168147,-.43814958),(33.964,.3),(999.,-.9)):
        expected = FinalPopulationProblem().objective((x,c))
        assert abs(problem.batch(x,c)[0,0]-expected) < 2e-11
    candidates, scans = [], []
    candidate_file = output/'scan_candidates.json'
    if args.reuse_scans:
        saved = json.loads(candidate_file.read_text())
        candidates, scans = saved['candidates'], saved['scans']
    else:
        for name, xs, cs in (
            ('dense_0_20', np.linspace(0.001,20,2001), np.linspace(-1,1,81)),
            ('dense_20_1000', np.arange(20.1,1000.0001,.1), np.linspace(-1,1,41))):
            all_errors = np.empty((len(xs),len(cs),2))
            for first in range(0,len(xs),32):
                selected = xs[first:first+32]
                grid_x = np.repeat(selected,len(cs))
                grid_c = np.tile(cs,len(selected))
                all_errors[first:first+len(selected)] = problem.batch(grid_x,grid_c).reshape(len(selected),len(cs),2)
                if first % 1024 == 0:
                    print(json.dumps(dict(stage=name, completed_x=first, total_x=len(xs),
                        current_best=float(all_errors[:first+len(selected)].min()), seconds=time.monotonic()-start)),flush=True)
            np.savez_compressed(output/f'{name}.npz', J_tau=xs, cosines=cs, errors=all_errors)
            profile = all_errors.min(axis=(1,2))
            peaks = find_peaks(-profile,distance=3)[0]
            top = sorted(peaks,key=lambda i:profile[i])[:80]
            top.append(int(profile.argmin()))
            for index in top:
                ci,oi=np.unravel_index(all_errors[index].argmin(),all_errors[index].shape)
                candidates.append(dict(source=name, J_tau=float(xs[index]), cosine=float(cs[ci]),
                    error=float(all_errors[index,ci,oi]), order=['jump-first','hamiltonian-first'][oi]))
            scans.append(dict(name=name,evaluations=int(all_errors.size//2),
                x_range=[float(xs.min()),float(xs.max())],cosine_range=[-1.,1.],best_error=float(all_errors.min())))
        # Two independent space-filling scans of increasingly large phases.
        for name, power, seed, logarithmic in (
            ('sobol_log_1000_1e6',17,20261011,True),
            ('sobol_linear_1000_1e6',17,719,False)):
            points=qmc.Sobol(2,scramble=True,seed=seed).random_base2(power)
            xs=10**(3+3*points[:,0]) if logarithmic else 1000+999000*points[:,0]
            cs=2*points[:,1]-1
            errors=np.empty((len(xs),2))
            for first in range(0,len(xs),2048):
                errors[first:first+2048]=problem.batch(xs[first:first+2048],cs[first:first+2048])
                if first % 32768 == 0:
                    print(json.dumps(dict(stage=name, completed=first, total=len(xs),
                        current_best=float(errors[:first+2048].min()),seconds=time.monotonic()-start)),flush=True)
            np.savez_compressed(output/f'{name}.npz', J_tau=xs,cosines=cs,errors=errors)
            for index in np.argsort(errors.min(axis=1))[:100]:
                oi=int(errors[index].argmin())
                candidates.append(dict(source=name,J_tau=float(xs[index]),cosine=float(cs[index]),
                    error=float(errors[index,oi]),order=['jump-first','hamiltonian-first'][oi]))
            scans.append(dict(name=name,evaluations=len(xs),x_range=[1000.,1e6],
                cosine_range=[-1.,1.],best_error=float(errors.min())))
        candidate_file.write_text(json.dumps(dict(candidates=candidates,scans=scans),indent=2)+'\n')

    refinements=[]
    for i,candidate in enumerate(sorted(candidates,key=lambda c:c['error'])):
        x,c=candidate['J_tau'],candidate['cosine']
        width=.2 if x <=1000 else 2.
        result=minimize(problem.objective,[x,c],method='Nelder-Mead',
            bounds=[(max(.000001,x-width),x+width),(-1,1)],
            options=dict(maxiter=350,xatol=1e-9,fatol=1e-12,initial_simplex=np.array([[x,c],[x+.01,c],[x,min(1,c+.01)]])))
        refined=dict(source=candidate['source'],J_tau=float(result.x[0]),cosine=float(result.x[1]),
            error=float(result.fun),evaluations=int(result.nfev))
        refinements.append(refined)
        if i%40==0:
            print(json.dumps(dict(stage='local_refinements',completed=i,total=len(candidates),
                best=min(r['error'] for r in refinements),seconds=time.monotonic()-start)),flush=True)
    (output/'refinements.json').write_text(json.dumps(refinements,indent=2)+'\n')
    # Independent global population optimizations in overlapping phase windows.
    global_searches=[]
    windows=[(0.00001,20.)]+[(float(a),float(a+50)) for a in range(0,1000,50)]
    windows += [(1000.,10000.),(10000.,100000.),(100000.,1e6)]
    for i,window in enumerate(windows):
        def vectorized(z):
            return problem.batch(z[0],z[1]).min(axis=1)
        result=differential_evolution(vectorized,[window,(-1,1)],seed=300+i,
            popsize=32,maxiter=180,tol=1e-9,polish=False,vectorized=True,updating='deferred')
        polished=minimize(problem.objective,result.x,method='Nelder-Mead',bounds=[window,(-1,1)],
            options=dict(maxiter=350,xatol=1e-9,fatol=1e-12))
        row=dict(window=list(window),J_tau=float(polished.x[0]),cosine=float(polished.x[1]),
            error=float(polished.fun),generations=int(result.nit))
        global_searches.append(row)
        print(json.dumps(dict(stage='independent_DE',window=list(window),best=row['error'],
            overall_best=min(r['error'] for r in refinements+global_searches),seconds=time.monotonic()-start)),flush=True)
    best=min(refinements+global_searches,key=lambda r:r['error'])
    order_index=int(problem.batch(best['J_tau'],best['cosine'])[0].argmin())
    best['order']=['jump-first','hamiltonian-first'][order_index]
    best['physical_parameters']=physical_parameters(best['J_tau'],best['cosine'])
    best['reference']=reference_validation(best['J_tau'],best['cosine'],best['order'],best['physical_parameters'])
    assert abs(best['reference']['final_population_error']-best['error']) < 1e-9
    report=dict(N=20,steps=100,p=.25,tolerance=.001,gamma=20,J_max=100,
        policy='rate-weighted',initial_states=['0'*20,'1'*20,'1'+'0'*19],
        meaning='Exact ensemble-mean final populations; no transient-fidelity constraint',
        scans=scans,refinements=refinements,global_searches=global_searches,best=best,
        elapsed_seconds=time.monotonic()-start,evaluations=problem.evaluations,
        passes_tolerance=best['error']<=.001,
        limitations='Finite search: no proof of a global minimum over unbounded tau; constant steps and the stated randomized DLL formula only')
    (output/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    physical=best['physical_parameters']
    lines=['# Broader N=20, 100-reset final-population search','',
        'Fixed Gamma=20, J<=100, p=1/4, Delta=0. Rate-weighted independently randomized '
        'single-jump Ding–Li–Lin dilation; exact Hamiltonian factor. Both factor orders are searched. '
        'The criterion is the maximum final population deviation from 1/4 across all sites and '
        'the three benchmark preparations. These are ensemble means, not finite-shot guarantees.','',
        f"Dense/scattered scan candidates: {sum(s['evaluations'] for s in scans):,}. "
        f"Local refinements: {len(refinements)}. Independent differential-evolution runs: {len(global_searches)}.",'',
        f"Total evaluations including refinements and validation: {problem.evaluations:,}. "
        f"Elapsed: {report['elapsed_seconds']:.1f} seconds.",'',
        f"Best found final error: {best['error']:.12g}. Required: 0.001. Pass: {report['passes_tolerance']}.",'',
        f"J*tau={best['J_tau']:.12g}, cosine={best['cosine']:.12g}, order={best['order']}.",'',
        f"Physical realization: J={physical['J']:.12g}, Gamma=20, tau={physical['tau']:.12g}, "
        f"T={physical['t_final']:.12g}. Repeated reservoir cycles are allowed to honor J<=100.",'',
        f"Existing channel implementation agreement: "
        f"{abs(best['error']-best['reference']['final_population_error']):.3g}.",'',
        '| Scan | Evaluations | Best grid/sample error |','|---|---:|---:|']
    for scan in scans:
        lines.append(f"| {scan['name']} | {scan['evaluations']:,} | {scan['best_error']:.9g} |")
    lines += ['', 'The finite numerical search does not prove a global minimum. Tau has no specified '
        'upper bound, so a literal exhaustive search is not possible. The saved arrays include '
        'both factor orders and the full cosine range [-1,1], with Hamiltonian phase J*tau '
        'searched densely through 1000 and by independent space-filling samples through 1e6. '
        'The conversion to physical parameters chooses the shortest reservoir angle cycle '
        'satisfying J<=100. No exact-reservoir channel was substituted.','']
    (output/'Results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(best=best,report=str(output/'Results.md'))),flush=True)


if __name__=='__main__':
    main()
