"""Reproducible offline audit: saved files, exact circuits and calibration only.

No IBM service or account is constructed. Original run files are read-only.
Run from the repository root with .venv/bin/python -m Model3.analysis.offline_audit.audit.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from scipy.linalg import eigvals, expm
from scipy.stats import chi2
from qiskit import transpile
from qiskit.circuit import Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import InstructionProperties, QubitProperties, Target
from qiskit_aer import AerSimulator

from library import reservoir_scaling as physics
from Model3.Experiment5 import circuit_depth, quality_path, run_schedule_comparison
from Model3.Experiment5.analyze_hardware import serializable
from Model3.Experiment5.check_aer_populations import reduced_density_circuit
from Model3.analysis.hardware_cost_redesign.search import evaluate

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_comparison_20261026T005025_553744Z'


def save(name, value):
    (OUT/name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+'\n')


def table(name, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def spin_stationary(gamma):
    """Independent full 8x8 spin master equation, without covariance routines."""
    n = 3; dimension = 2**n
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    lower = np.array([[0, 1], [0, 0]], complex)
    def local(a, j):
        value = np.ones((1, 1), complex)
        for k in reversed(range(n)):
            value = np.kron(value, a if j == k else np.eye(2))
        return value
    H = sum((local(X,j)@local(X,j+1)+local(Y,j)@local(Y,j+1))/2 for j in range(n-1))
    identity = np.eye(dimension)
    generator = -1j*(np.kron(identity,H)-np.kron(H.T,identity))
    for site, probability in ((0,.9),(2,.1)):
        for amplitude, rate in ((lower.conj().T,gamma*probability),(lower,gamma*(1-probability))):
            V = np.sqrt(rate)*local(amplitude,site); VV = V.conj().T@V
            generator += np.kron(V.conj(),V)-.5*(np.kron(identity,VV)+np.kron(VV.T,identity))
    matrix = generator.copy(); rhs = np.zeros(dimension**2, complex)
    matrix[0] = identity.ravel(order='F'); rhs[0] = 1
    rho = np.linalg.solve(matrix,rhs).reshape((dimension,dimension),order='F')
    occupations = [local(lower.conj().T@lower,j) for j in range(n)]
    # Current leaving site 0 follows directly from its Hamiltonian continuity equation.
    current = -1j*(H@occupations[0]-occupations[0]@H)
    eigenvalues = eigvals(generator)
    return dict(gamma=gamma, populations=[float(np.trace(op@rho).real) for op in occupations],
        left_current=float(np.trace(current@rho).real), trace=float(np.trace(rho).real),
        minimum_density_eigenvalue=float(np.linalg.eigvalsh(rho).min()),
        lindblad_residual=float(np.linalg.norm(generator@rho.ravel(order='F'))),
        zero_eigenvalues=int(np.sum(abs(eigenvalues)<1e-9)))


def calibration_target(snapshot):
    properties = snapshot['backend_properties']
    qubits = [{item['name']:item for item in row} for row in properties['qubits']]
    target = Target(num_qubits=len(qubits), qubit_properties=[QubitProperties(
        t1=q['T1']['value']*1e-6 if 'T1' in q else None,
        t2=q['T2']['value']*1e-6 if 'T2' in q else None) for q in qubits])
    edges = {}
    for gate in properties['gates']:
        if gate['gate'] == 'cz':
            params = {p['name']:p['value'] for p in gate['parameters']}
            pair = tuple(sorted(gate['qubits']))
            edges[pair] = max(edges.get(pair,0.),params['gate_error'])
    target.add_instruction(CZGate(), {pair:InstructionProperties(error=error) for pair,error in edges.items()})
    target.add_instruction(Measure(), {(j,):InstructionProperties(error=q['readout_error']['value'])
        for j,q in enumerate(qubits) if 'readout_error' in q})
    return SimpleNamespace(target=target,name='ibm_phoenix_saved_only'), qubits, edges


def compact_compile(body, snapshot, cfg):
    """Regenerate the saved fixed-layout compilation, never claim binary identity."""
    target = Target(num_qubits=120)
    for name, operation in [('cz',CZGate()),('x',XGate()),('sx',SXGate()),('measure',Measure()),('reset',Reset())]:
        props = {tuple(x['qubits']):InstructionProperties(duration=x['duration_seconds'],error=x['error'])
            for x in snapshot['instructions'] if x['name']==name}
        target.add_instruction(operation,props)
    target.add_instruction(RZGate(Parameter('phi')), {(q,):InstructionProperties(duration=0,error=0) for q in cfg['physical_path']})
    mapping = cfg['physical_path'][1:-1]+[cfg['physical_path'][0],cfg['physical_path'][-1]]
    native = transpile(body,target=target,initial_layout=mapping,optimization_level=cfg['optimization_level'],seed_transpiler=cfg['seed_transpiler'])
    assert native.layout.final_index_layout(filter_ancillas=True)==mapping
    native = circuit_depth.append_readout(native,mapping[:-2],'z')
    return circuit_depth.metrics(native)


def main(reuse_density=False):
    manifest = json.loads((RUN/'manifest.json').read_bytes()); cfg = manifest['configuration']
    snapshot = json.loads((RUN/'backend_snapshot.json').read_bytes())
    sources = {label:json.loads((RUN/f'{label}.json').read_bytes()) for label in ('baseline','candidate')}
    report = run_schedule_comparison.analyze(RUN)
    cached = json.loads((RUN/'hardware_results.json').read_bytes())
    result = dict(scope='Offline only; no IBM connection, query or job', run=RUN.name,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()},
        verification=report['verification'], physics=[spin_stationary(2.),spin_stationary(1.)])
    previous = json.loads((OUT/'audit.json').read_text()) if reuse_density else None
    if previous is not None:
        assert previous['input_sha256']==result['input_sha256'], 'cached density inputs changed'
    # Independent integer-mask interpretation, not the existing reversed-string parser.
    independent = {k:[] for k in sources}; total = 0
    mapping = cfg['physical_path'][1:-1]+[cfg['physical_path'][0],cfg['physical_path'][-1]]
    for batch in manifest['batches']:
        for record, observation in zip(manifest['circuits'][batch['start']:batch['stop']],batch['observations'],strict=True):
            counts = observation['outcome_counts']; shots = sum(counts.values()); total += shots
            assert shots == record['shots']; assert record['final_logical_to_physical']==mapping
            pop = np.array([sum(count*((int(key,2)>>j)&1) for key,count in counts.items())/shots for j in range(10)])
            if record['kind']=='trajectory': independent[record['schedule']].append((record['trajectory'],pop))
    for label, values in independent.items():
        values = np.array([v for _,v in sorted(values,key=lambda pair:pair[0])])
        np.testing.assert_allclose(values,report['schedules'][label]['per_path_populations'],atol=0,rtol=0)
        np.testing.assert_allclose(values.mean(axis=0),cached['schedules'][label]['hardware_populations'],atol=0,rtol=0)
    assert total == cfg['total_shots']
    with zipfile.ZipFile(RUN/'results_bundle.zip') as archive:
        assert archive.testzip() is None
        equal = [name for name in archive.namelist() if Path(name).name in {'baseline.json','candidate.json','manifest.json','hardware_results.json','backend_snapshot.json'}]
        for name in equal:
            assert json.loads(archive.read(name))==json.loads((RUN/Path(name).name).read_bytes())
    result['integrity'] = dict(total_circuits=len(manifest['circuits']),total_shots=total,
        integer_mask_count_parser_matches=True,all_recorded_layouts_match=True,bundle_json_verified=equal)
    # Independent full-spin steady states above and exact covariance target at N=10.
    p = sources['baseline']['configuration']['physics']
    model = physics.Model(p['n_qubits'],p['gamma']/p['J'],p['p_left'],p['p_right'],p['exchange_prefactor'])
    target = physics.exact_stationary(model)
    np.testing.assert_allclose(np.diag(target),sources['baseline']['stationary_observables']['populations'],atol=1e-13)
    result['target'] = dict(populations=np.diag(target),currents=4*model.prefactor*p['J']*np.diag(target,1))
    result['schedules'] = {}; rows = []
    zero, one, mixture = (next(c for c in report['controls'] if np.allclose(c['expected_populations'],x)) for x in (0,1,.5))
    a, b = np.array(zero['populations']),np.array(one['populations'])
    result['controls'] = report['control_assessment']
    density = AerSimulator(method='density_matrix',max_parallel_threads=2,max_parallel_experiments=1,max_memory_mb=1200,
        fusion_threshold=1,fusion_max_qubit=2)
    for label, source in sources.items():
        print(f'{label}: replaying all saved covariance paths',flush=True)
        settings = source['configuration']; dts = source['schedule']['step_dts']; R = cfg['trajectory_count']
        choices = np.array(source['schedule']['jump_choices'])[:R]
        C = np.broadcast_to(np.diag(source['preparation']['populations']),(1,R,10,10)).copy()
        for k,dt in enumerate(dts):
            branch = physics.channel(model,p['J']*dt,settings['sampling_policy'],settings['order'],settings['system_formula'])
            C = physics.sampled_step(C,choices[:,k],branch)
        saved = np.asarray(source['final_covariances_real_gauge'])[:,:R]
        discrepancy = float(np.max(abs(C-saved))); assert discrepancy<1e-12
        replay = evaluate(model,p['J'],dts,10/p['J'],.016 if label=='baseline' else .0165)
        np.testing.assert_allclose(np.diag(replay['output_covariance']),report['schedules'][label]['ideal_mean_channel'],atol=1e-13)
        ideal_paths = C[0].diagonal(axis1=-2,axis2=-1); ideal = ideal_paths.mean(axis=0)
        path_se = ideal_paths.std(axis=0,ddof=1)/np.sqrt(R)
        shot_se = np.sqrt(np.sum(ideal_paths*(1-ideal_paths),axis=0)/(R**2*cfg['shots']))
        hw = np.array(report['schedules'][label]['hardware_populations']); residual = hw-ideal
        controls_forward = a+(b-a)*ideal
        control_variance = ((1-ideal)**2*a*(1-a)+ideal**2*b*(1-b))/(cfg['control_shots']-1)
        fixed_hw_variance = np.array(report['schedules'][label]['fixed_paths_shot_se'])**2
        control_z = (hw-controls_forward)/np.sqrt(control_variance+fixed_hw_variance)
        # Hoeffding union bound: independent ideal shots, fixed sampled branch paths.
        hoeffding = min(1.,20*np.exp(-2*R*cfg['shots']*max(abs(residual))**2))
        best_permutation = min((max(abs(hw-np.array([.58 if j==left else .42 if j==right else .5 for j in range(10)]))),left+1,right+1)
            for left in range(10) for right in range(10) if left!=right)
        info = dict(steps=len(dts),physical_time=sum(dts),covariance_replay_max_difference=discrepancy,
            mean_output_error=replay['output_error'],mean_forward_maximum=replay['forward_maximum'],
            final_step_stationary_bias=replay['final_dt_stationary_bias'],
            ideal_population_sampling_distance=float(max(abs(ideal-np.diag(replay['output_covariance'])))),
            maximum_ideal_path_two_se=float(2*max(path_se)),maximum_ideal_shot_two_se=float(2*max(shot_se)),
            maximum_ideal_combined_two_se=float(2*max(np.sqrt(path_se**2+shot_se**2))),
            ideal_sampling_allowance=.004 if label=='baseline' else .0035,
            shot_only_hoeffding_union_bound=hoeffding,
            best_target_relabeling=dict(max_error=best_permutation[0],left_site=best_permutation[1],right_site=best_permutation[2]),
            exact_density_checks=[])
        for j,q in enumerate(cfg['physical_path'][1:-1]):
            rows.append(dict(schedule=label,site=j+1,physical_qubit=q,ideal=ideal[j],hardware=hw[j],
                ideal_path_se=path_se[j],ideal_shot_se=shot_se[j],hardware_minus_ideal=residual[j],
                forward_control_prediction=controls_forward[j],hardware_minus_forward=hw[j]-controls_forward[j],
                nominal_forward_residual_z=control_z[j]))
        for index in (0,1023):
            if previous is not None:
                matches=[row for row in previous['schedules'][label]['exact_density_checks'] if row['trajectory']==index]
                assert len(matches)==1 and matches[0]['max_population_discrepancy']<1e-11
                info['exact_density_checks'].append(matches[0]);continue
            print(f'{label}: exact N=10 density matrix and saved-target recompilation, path {index}',flush=True)
            body = circuit_depth.build_body(source,index)
            reduced = reduced_density_circuit(body,10)
            simulated = density.run(reduced,shots=1).result(); assert simulated.success
            populations = np.array([(1-simulated.data()[f'z{j}'])/2 for j in range(10)])
            error = float(max(abs(populations-ideal_paths[index]))); assert error<1e-11
            native = compact_compile(body,snapshot,cfg)
            recorded = next(r['metrics'] for r in manifest['circuits'] if r.get('schedule')==label and r.get('trajectory')==index)
            info['exact_density_checks'].append(dict(trajectory=index,max_population_discrepancy=error,
                regenerated_native_metrics=native,saved_metrics=recorded,metrics_match=native==recorded))
            print(f'{label} path {index}: density discrepancy {error:.3g}',flush=True)
        result['schedules'][label] = info
        save('audit.json',result)
    table('population_checks.csv',rows)
    print('Reranking connected paths from full saved device calibration only',flush=True)
    backend,qubits,edges = calibration_target(snapshot)
    nodes,eligible_edges = quality_path.calibration(backend,{7},80e-6,60e-6,.005,.02)
    weights = quality_path.gate_weights(sources['baseline'],list(range(cfg['trajectory_count'])))
    ranking,metrics = quality_path.score(cfg['physical_path'],nodes,eligible_edges,weights)
    selection = quality_path.select(backend,sources['baseline'],list(range(cfg['trajectory_count'])))
    result['mapping'] = dict(current_path=cfg['physical_path'],current_metrics=metrics,current_passes_original_filters=True,
        current_ranking=ranking,offline_selection=selection,calibration_captured=snapshot['captured_at_utc'],
        note='Static calibration heuristic only; cannot predict observable accuracy or prove a better hardware mapping.')
    node_rows = []
    for position,q in enumerate(cfg['physical_path']):
        row = qubits[q]; prop = backend.target.qubit_properties[q]
        node_rows.append(dict(physical_qubit=q,role='left_bath' if position==0 else 'right_bath' if position==11 else f'site_{position}',
            t1_us=prop.t1*1e6,t2_us=prop.t2*1e6,readout_error=row['readout_error']['value'],
            readout_date=row['readout_error']['date'],t1_date=row['T1']['date'],t2_date=row['T2']['date'],
            initialization_error_date=row.get('init_error',{}).get('date','missing')))
    edge_rows = [dict(left=a,right=b,cz_error=edges[tuple(sorted((a,b)))],baseline_cz_uses=weights[k],
        percent_of_device_edges_with_lower_error=100*np.mean(np.array(list(edges.values()))<edges[tuple(sorted((a,b)))]))
        for k,(a,b) in enumerate(zip(cfg['physical_path'],cfg['physical_path'][1:]))]
    table('chain_qubits.csv',node_rows); table('chain_edges.csv',edge_rows)
    result['calibration_chain_nodes'] = node_rows;result['calibration_chain_edges'] = edge_rows
    # Examine heterogeneity of residuals over the four large batches; last batch has only two pairs.
    batch_tests = {}
    for label in sources:
        values = np.asarray(report['schedules'][label]['per_path_populations'])-np.asarray(report['schedules'][label]['per_path_ideal_populations'])
        groups = []
        for batch in manifest['batches']:
            indices = [r['trajectory'] for r in manifest['circuits'][batch['start']:batch['stop']] if r.get('schedule')==label]
            if len(indices)>=30: groups.append(values[indices])
        means = np.array([g.mean(axis=0) for g in groups]);variances = np.array([g.var(axis=0,ddof=1)/len(g) for g in groups])
        w = 1/variances; center = (means*w).sum(axis=0)/w.sum(axis=0)
        statistic = np.sum(w*(means-center)**2,axis=0); raw_p = chi2.sf(statistic,len(groups)-1)
        batch_tests[label] = dict(group_sizes=[len(g) for g in groups],means=means,nominal_pvalues=raw_p,
            note='Exploratory heterogeneity, stationary-independent-path approximation, not causal evidence of drift; final two-pair batch omitted.')
    all_p = np.concatenate([np.array(batch_tests[k]['nominal_pvalues']) for k in sources])
    order=np.argsort(all_p); adjusted=np.empty_like(all_p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(all_p[order]*(len(all_p)-np.arange(len(all_p)))))
    for i,label in enumerate(sources):batch_tests[label]['holm_adjusted_pvalues']=adjusted[10*i:10*(i+1)]
    result['batch_heterogeneity'] = batch_tests
    save('audit.json',result)
    from Model3.analysis.offline_audit.finish import augment
    augment()
    print('Audit complete:',OUT/'audit.json',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-density',action='store_true',help='Reuse completed exact simulations only when every saved input hash matches')
    main(parser.parse_args().reuse_density)
