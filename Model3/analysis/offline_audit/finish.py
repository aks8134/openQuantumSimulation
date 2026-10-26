"""Cheap supplementary checks and figures for the saved offline audit."""
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import chisquare, norm, fisher_exact

from library import reservoir_scaling as physics
from Model3.Experiment5.run_reset_survival import analyze as analyze_survival

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_comparison_20261026T005025_553744Z'


def augment():
    result = json.loads((OUT/'audit.json').read_text())
    survival_folder=RUN.parent/'ibm_phoenix_reset_survival_20261025T224328_184810Z'
    survival=analyze_survival(survival_folder)
    lookup={r['case_id']:r for r in survival['cases']};survival_p=[]
    for comparison in survival['comparisons']:
        a,b=(lookup[comparison[key]] for key in ('reset_case','idle_case'))
        ka,kb=round(a['system_population']*a['shots']),round(b['system_population']*b['shots'])
        survival_p.append(fisher_exact([[ka,a['shots']-ka],[kb,b['shots']-kb]]).pvalue)
    order=np.argsort(survival_p);adjusted=np.empty(24)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(np.array(survival_p)[order]*(24-np.arange(24))))
    result['existing_survival_recheck']=dict(run=survival_folder.name,
        verification=survival['verification'],circuits=len(survival['cases']),
        minimum_holm_p_reset_vs_idle=float(adjusted.min()),
        selected_cases=[dict(case_id=r['case_id'],system_population=r['system_population'],bath_population=r['bath_population'])
            for r in survival['cases'] if r['initial_system']==1 and
            (r['steps']==0 or (r['steps']==72 and r['idle_scale']==1.))])
    for label in ('baseline','candidate'):
        source = json.loads((RUN/f'{label}.json').read_bytes()); config = source['configuration']; p=config['physics']
        model = physics.Model(p['n_qubits'],p['gamma']/p['J'],p['p_left'],p['p_right'],p['exchange_prefactor'])
        dts = source['schedule']['step_dts']; choices = np.array(source['schedule']['jump_choices'])
        streams = [np.random.default_rng(np.random.SeedSequence([config['seed_trajectories'],i])) for i in range(1024)]
        regenerated = physics.draw_choice_block(streams,np.array(source['schedule']['jump_probabilities']))[:,:len(dts)]
        assert np.array_equal(regenerated,choices)
        counts = np.bincount(choices.ravel(),minlength=4)
        frequency = chisquare(counts,choices.size*np.array(source['schedule']['jump_probabilities']))
        state = np.diag(source['preparation']['populations'])
        counterfactuals = np.array([np.zeros((10,10)),np.eye(10)])
        target = physics.exact_stationary(model)
        def distance(C):
            e=physics.observable_error(C,target,model.prefactor)
            return max(e.values())
        for dt in dts:
            channel = physics.channel(model,p['J']*dt,config['sampling_policy'],config['order'],config['system_formula'])
            state = physics.mean_step(state,channel)
            counterfactuals = physics.mean_step(counterfactuals,channel)
        counterfactual_errors = [distance(x) for x in counterfactuals]
        stationary,_ = physics.mean_stationary(channel,model.n)
        saved=np.asarray(source['final_covariances_real_gauge'])[0]
        sample_error=physics.observable_error(saved.mean(axis=0),state,model.prefactor)
        all_two_se=max(2*np.diagonal(saved,axis1=-2,axis2=-1).std(axis=0,ddof=1).max()/np.sqrt(len(saved)),
            8*model.prefactor*np.diagonal(saved,offset=1,axis1=-2,axis2=-1).std(axis=0,ddof=1).max()/np.sqrt(len(saved)))
        # Bound the effect of numerical fixed-point residual via the smallest
        # singular value of I-L (Frobenius-vectorized covariance space).
        basis=np.eye(model.n**2).reshape(-1,model.n,model.n)
        linear=np.column_stack([physics.mean_step(e,channel,affine=False).ravel() for e in basis])
        smallest=float(np.linalg.svd(np.eye(model.n**2)-linear,compute_uv=False)[-1])
        residual=float(np.linalg.norm(physics.mean_step(stationary,channel)-stationary))
        fixed_error=residual/smallest
        # Each conditional A is a contraction; the convex mean is nonexpansive
        # in Frobenius norm. A future node's local error is at most b+2||C-C_dt,*||_F.
        # Check every earlier node, then certify the remaining infinite integer-node tail.
        tolerance = .016 if label=='baseline' else .0165
        bias = distance(stationary); maximum=distance(state);nodes=0
        while bias+max(1,4*model.prefactor)*(np.linalg.norm(state-stationary)+2*fixed_error)>tolerance and nodes*dts[-1]<200/p['J']:
            state=physics.mean_step(state,channel); nodes+=1; maximum=max(maximum,distance(state))
        bound=bias+max(1,4*model.prefactor)*(np.linalg.norm(state-stationary)+2*fixed_error)
        result['schedules'][label].update(seed_reproduction_exact=True,branch_counts=counts.tolist(),
            all_observable_sampling_distance=max(sample_error.values()),all_observable_maximum_two_se=float(all_two_se),
            branch_frequency_nominal_pvalue=float(frequency.pvalue),
            initial_state_counterfactual_errors=dict(all_zero=counterfactual_errors[0],all_one=counterfactual_errors[1]),
            extended_mean_check=dict(checked_future_nodes=nodes,checked_future_physical_time=nodes*dts[-1],
                maximum_checked_error=maximum,remaining_tail_bound=float(bound),mean_tolerance=tolerance,
                fixed_point_residual=residual,fixed_point_error_bound=fixed_error,
                permanent_mean_node_bound_passed=bool(maximum<=tolerance and bound<=tolerance),
                scope='Numerical bound in covariance sector for this half-mixed mean; constant final timestep; all future integer nodes. No measurement, hardware, continuous between-node or many-body trace-distance certificate.'))
    rows = list(csv.DictReader((OUT/'population_checks.csv').open()))
    # Forward control test: known fixed ideal paths, noisy independent controls/counts.
    ps = 2*norm.sf(abs(np.array([float(r['nominal_forward_residual_z']) for r in rows])))
    order=np.argsort(ps);adj=np.empty_like(ps)
    adj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    result['static_forward_control_tests'] = [dict(schedule=r['schedule'],site=int(r['site']),
        physical_qubit=int(r['physical_qubit']),residual=float(r['hardware_minus_forward']),
        nominal_z=float(r['nominal_forward_residual_z']),holm_p=float(a)) for r,a in zip(rows,adj)]
    test_log=(OUT/'test_results.txt').read_bytes()
    test_match=re.search(rb'Ran (\d+) tests in ([\d.]+)s',test_log)
    assert test_match is not None
    result['validation'] = dict(model3_unittest_count=int(test_match[1]),model3_unittest_passed=test_log.strip().endswith(b'OK'),
        command="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3_boundary_analysis/mpl .venv/bin/python -m unittest discover -s tests -p 'test_model3*.py' -q",
        runtime_seconds=float(test_match[2]),test_log_sha256=hashlib.sha256(test_log).hexdigest(),
        test_scope='Separately recorded test run; audit.py does not rerun the unittest suite.',
        numerical_environment={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit','qiskit-aer','qiskit-ibm-runtime')},
        metadata_fix='Hard-coded R=8192 provenance sentence replaced with actual replay count in verify.py and prepare_pair_inputs.py; historical inputs/results preserved.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plots(rows,result)
    return result


def plots(rows,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained',sharey=True)
    for ax,label,title in zip(axes,('baseline','candidate'),('72-step cosine','48-step warped cosine')):
        selected=[r for r in rows if r['schedule']==label];x=np.arange(1,11)
        ax.plot(x,result['target']['populations'],'k--',label='Physical stationary populations')
        ax.plot(x,[float(r['ideal']) for r in selected],'o-',color='#00897b',label='Ideal, same 1024 branch paths')
        ax.plot(x,[float(r['forward_control_prediction']) for r in selected],'s:',color='#8e24aa',label='Ideal through static zero/one control response')
        ax.plot(x,[float(r['hardware']) for r in selected],'d-',color='#c62828',label='Saved hardware')
        ax.set(title=title,xlabel='Site',ylabel='Population',xticks=x,ylim=(.38,.63));ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='upper right')
    fig.savefig(OUT/'static_control_populations.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    nodes=result['calibration_chain_nodes'];x=np.arange(12)
    axes[0].plot(x,[r['t1_us'] for r in nodes],'o-',label='Saved T1 (μs)')
    axes[0].plot(x,[r['t2_us'] for r in nodes],'s-',label='Saved T2 (μs)')
    axes[0].set(ylabel='Microseconds',title='Saved chain calibration: baths at the two ends');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].bar(np.arange(11)+.5,[100*r['cz_error'] for r in result['calibration_chain_edges']],width=.7,color='#1565c0')
    axes[1].set(ylabel='CZ calibration error (%)',xlabel='Physical qubit along the connected chain',xticks=x,
        xticklabels=[str(r['physical_qubit']) for r in nodes]);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(OUT/'saved_chain_calibration.png',dpi=170);plt.close(fig)


if __name__=='__main__':augment()
