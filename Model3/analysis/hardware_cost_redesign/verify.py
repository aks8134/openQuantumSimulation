"""Replay and compile the best bounded-search candidate entirely offline.

The target is reconstructed from a saved calibration snapshot. Timing is a
block-duration model, not a wall-clock guarantee. No IBM service is created.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit import Delay, Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.transpiler.passes.scheduling import ASAPScheduleAnalysis

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from library import reservoir_scaling as physics
from Model3.Experiment4 import budgeted_steady_state as replay_engine
from Model3.Experiment7.compare_schedules import parse_arguments as replay_options
from Model3.Experiment5 import circuit_depth
from Model3.Experiment5.export_hardware_input import export
from Model3.analysis.hardware_cost_redesign.search import evaluate

DIRECTORY=Path(__file__).parent
RUN=ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_reset_survival_20261025T224328_184810Z'


def saved_target():
    manifest=json.loads((RUN/'manifest.json').read_bytes())
    raw=(RUN/'backend_snapshot.json').read_bytes();snapshot=json.loads(raw)
    cfg=manifest['configuration'];timing=cfg['timing_model']
    target=Target(num_qubits=cfg['num_qubits'],dt=timing['dt_seconds'],
        granularity=1,min_length=timing['min_length_dt'],pulse_alignment=timing['alignment_dt'],acquire_alignment=timing['alignment_dt'])
    path=cfg['physical_path']
    for name,operation in [('cz',CZGate()),('x',XGate()),('sx',SXGate()),('measure',Measure()),('reset',Reset())]:
        props={tuple(i['qubits']):InstructionProperties(duration=i['duration_seconds'],error=i['error'])
               for i in snapshot['instructions'] if i['name']==name}
        target.add_instruction(operation,props)
    target.add_instruction(RZGate(Parameter('phi')),{(q,):InstructionProperties(duration=0.,error=0.) for q in path})
    target.add_instruction(Delay(Parameter('duration')),{(q,):None for q in path})
    return target,path,dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(),captured_at_utc=snapshot['captured_at_utc'],
        backend=snapshot['backend'],dt_seconds=target.dt,
        scope='Offline target reconstructed only on the saved chain; RZ is virtual. Dynamic reset-block overhead unknown. No fresh backend query.')


def as_plan(evaluation,preparation):
    dts=evaluation['step_dts'];steps=len(dts);time0=evaluation['physical_time']
    stages=[]
    for dt in dts:
        if stages and stages[-1]['dt']==dt:stages[-1]['steps']+=1
        else:stages.append(dict(dt=dt,steps=1))
    selected=dict(step_dts=dts,steps=steps,checked_from_step=steps,tail_window_covered=evaluation['passed'],
        window_start=time0,window_end=evaluation['forward_times'][-1],estimated_total_depth=None,stages=stages,
        final_mean_covariance_real_gauge=evaluation['output_covariance'],
        forward_validation=dict(step_dts=[dts[-1]]*evaluation['validation_steps'],times=evaluation['forward_times'],
            mean_errors=evaluation['forward_errors'],interpretation='Classical constant-final-step continuation at every node; output-only trajectory validation.'))
    return dict(selected=selected,selected_preparation=preparation)


def compile_sources(source,target,path,count=4):
    records=[]
    initial=list(path[1:-1])+[path[0],path[-1]]
    for index in range(count):
        logical=circuit_depth.build_body(source,index)
        native=transpile(logical,target=target,initial_layout=initial,optimization_level=3,seed_transpiler=11)
        mapping=native.layout.final_index_layout(filter_ancillas=True)
        if mapping != initial:raise ValueError('fixed-chain compilation changed layout')
        native=circuit_depth.append_readout(native,mapping[:10],'z')
        metrics=circuit_depth.metrics(native)
        if metrics['operation_counts'].get('reset',0)!=len(source['schedule']['step_dts']):raise ValueError('reset count mismatch')
        for inst in native.data:
            qargs=tuple(native.find_bit(q).index for q in inst.qubits)
            if not target.instruction_supported(operation_name=inst.operation.name,qargs=qargs):
                raise ValueError('unsupported native instruction')
        manager=PassManager([ASAPScheduleAnalysis(target=target)]);manager.run(native)
        ends={};final=[]
        outcome=next(r for r in native.cregs if r.name=='outcome')
        for node,(block,start) in manager.property_set['node_start_time'].items():
            if node.name=='barrier':duration=0
            elif node.name=='delay':duration=int(node.op.duration)
            else:
                qargs=tuple(native.find_bit(q).index for q in node.qargs)
                duration=target.seconds_to_dt(target[node.name][qargs].duration)
            ends[block]=max(ends.get(block,0),int(start)+duration)
            if node.name=='measure' and node.cargs[0] in outcome:final.append((block,int(start)))
        starts=[sum(v for k,v in ends.items() if k<block)+start for block,start in final]
        if len(starts)!=10:raise ValueError('missing final system readout')
        timing=dict(dynamic_blocks=len(ends),block_durations_dt={str(k):v for k,v in ends.items()},
            summed_pre_measure_dt=max(starts),summed_total_dt=sum(ends.values()),
            final_readout_start_range_dt=[min(starts),max(starts)],unknown_reset_block_latency=True)
        records.append(dict(trajectory=index,metrics=metrics,block_timing=timing,
            modeled_pre_readout_us=timing['summed_pre_measure_dt']*target.dt*1e6,
            modeled_total_us=timing['summed_total_dt']*target.dt*1e6))
    return records


def main(arguments=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',type=Path,default=DIRECTORY/'search_results.json')
    p.add_argument('--output',type=Path,default=DIRECTORY/'verification.json')
    p.add_argument('--trajectories',type=int,default=8192)
    p.add_argument('--seeds',type=int,nargs='+',default=[29,30,31])
    p.add_argument('--candidate-steps',type=int)
    p.add_argument('--mean-tolerance',type=float,default=.016)
    p.add_argument('--sampling-tolerance',type=float,default=.004)
    p.add_argument('--labels',nargs='+',choices=['baseline','candidate'],default=['baseline','candidate'])
    p.add_argument('--compile-count',type=int,default=4)
    a=p.parse_args(arguments)
    if (a.trajectories<128 or not 1<=a.compile_count<=128 or a.mean_tolerance<=0 or a.sampling_tolerance<=0
            or not np.isclose(a.mean_tolerance+a.sampling_tolerance,.02,rtol=0.,atol=1e-14)):
        raise ValueError('need R>=128, positive allowances summing to 0.02 and compile count 1..128')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    search=json.loads(a.search.read_bytes());selected=search['selection']
    if selected is None:raise ValueError('bounded search found no passing mean candidate; no replay or hardware input generated')
    candidate=next(t for t in search['trials'] if t['family']==selected['family'] and t['steps']==selected['steps']
                   and t['forward_maximum']==selected['forward_maximum'])
    if a.candidate_steps:
        candidate=min([t for t in search['trials'] if t['steps']==a.candidate_steps],key=lambda t:t['forward_maximum'])
    baseline_source=json.loads((ROOT/search['source']).read_bytes())
    opts=replay_options(['--n-qubits','10','--J','1','--gamma','1','--steady-tolerance','.02',
                        '--sampling-tolerance',str(a.sampling_tolerance),'--no-plots'])
    target,path,provenance=saved_target()
    result=dict(scope='Offline arithmetic, covariance replay and fixed-chain native compilation only; no hardware execution or noisy simulation.',
        target=provenance,search_sha256=hashlib.sha256(a.search.read_bytes()).hexdigest(),runs={},compact_inputs={},
        total_tolerance=.02,mean_tolerance=a.mean_tolerance,sampling_tolerance=a.sampling_tolerance)
    all_passed=True
    for label,evaluation in [('baseline',search['baseline']),('candidate',candidate)]:
        if label not in a.labels:continue
        independent=evaluate(opts.model,1.,evaluation['step_dts'],10.,a.mean_tolerance)
        if not independent['passed'] or not np.isclose(independent['forward_maximum'],evaluation['forward_maximum'],atol=1e-12,rtol=1e-10):
            raise ValueError('independent direct mean validation disagrees')
        plan=as_plan(independent,baseline_source['preparation']);plan['physical_stationary_real_gauge']=physics.exact_stationary(opts.model).tolist()
        result['runs'][label]=dict(steps=len(evaluation['step_dts']),forward_maximum=independent['forward_maximum'],replays={})
        for seed in a.seeds:
            options=copy.copy(opts);options.seed_trajectories=seed
            start=time.perf_counter();replay=replay_engine.replay(options,plan,a.trajectories)
            replay.update(experiment='Model3/Experiment7',schedule_method='hardware_cost_redesign_'+label)
            full_bytes=json.dumps(replay,sort_keys=True,separators=(',',':')).encode()
            sha=hashlib.sha256(full_bytes).hexdigest()
            assessment=replay['assessment'];all_passed &= replay['run_status']=='passed'
            result['runs'][label]['replays'][str(seed)]=dict(status=replay['run_status'],assessment=assessment,
                trajectories=a.trajectories,transient_full_replay_sha256=sha,wall_seconds=time.perf_counter()-start,
                sampled_final_populations=replay['observables']['populations'][0][-1],
                sampled_final_currents=replay['observables']['excitation_flows'][0][-1])
            if seed==a.seeds[0] and replay['run_status']=='passed':
                compact=export(replay,list(range(128)),sha)
                # Hash refers to canonical transient bytes; retain exact compact paths and audit summaries.
                compact['provenance']['full_replay_storage']=f'Computed in memory; hash is canonical sorted JSON. Full R={a.trajectories} covariance dump not retained to avoid bulky artifacts.'
                output=a.output.parent/f'{label}_seed{seed}_paths128.json'
                physics.atomic_json(output,compact)
                result['compact_inputs'][label]=output.name
                result['runs'][label]['native_circuits']=compile_sources(compact,target,path,count=a.compile_count)
            physics.atomic_json(a.output,result)
            print(f"{label} seed {seed}: {replay['run_status']}, sampled={assessment['output_sampled_error']:.8f}, sampling={assessment['max_sampling_distance']:.8f}",flush=True)
    result['all_sampled_replays_passed']=bool(all_passed)
    result['candidate_ready_for_review']=bool(all_passed and candidate['steps']<=.8*search['baseline']['steps'])
    result['interpretation']='Readiness requires sampled checks and >=20% CZ reduction in the current implementation; it is not a prediction of 0.02 noisy-hardware accuracy.'
    physics.atomic_json(a.output,result);print(f'Saved: {a.output}',flush=True)
    return result


if __name__=='__main__':main()
