"""Paired schedule or mapping Z-population comparison; explicit CLI submission.

Defaults: 1024 paths x 8 shots for each schedule, three 2048-shot controls,
22528 shots total in five pair-preserving batches. Mapping mode uses the
identical candidate schedule on two paths, with three controls per path:
2054 circuits and 28672 shots at these defaults. No new schedule search.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from qiskit import qpy,transpile

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from library.reservoir_scaling import atomic_json
from Model3.Experiment5 import circuit_depth as circuits,run_hardware as hardware,control_quality,quality_path
from Model3.Experiment5.analyze_hardware import moments,serializable,source_integrity
from Model3.Experiment5.analyze_populations import population_error

EXPERIMENT='Model3/Experiment5/schedule_comparison'
DEFAULT_INPUT=DIRECTORY/'inputs/paired_cosine72_vs48_seed29_R1024.zip'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def parse_arguments(arguments=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-bundle',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--backend',default='ibm_phoenix')
    p.add_argument('--comparison',choices=['schedule','mapping'],default='schedule',
        help='mapping uses the same candidate schedule and branch paths on two chains')
    p.add_argument('--alternative-path',type=int,nargs='+',
        help='second connected bath-system-bath chain for --comparison mapping')
    p.add_argument('--verify-previous-run',type=Path,
        help='verify every previous native QPY chunk locally before any new backend connection or submission')
    p.add_argument('--account-file',type=Path)
    p.add_argument('--trajectory-count',type=int,default=1024)
    p.add_argument('--shots',type=int,default=8)
    p.add_argument('--control-shots',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--order-seed',type=int,default=29)
    p.add_argument('--seed-transpiler',type=int,default=11)
    p.add_argument('--optimization-level',type=int,choices=range(4),default=3)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--compile-only',action='store_true')
    action.add_argument('--resume',type=Path)
    action.add_argument('--analyze-only',type=Path)
    p.add_argument('--output-directory',type=Path,default=DIRECTORY/'hardware_runs')
    p.add_argument('--analysis-output-directory',type=Path)
    a=p.parse_args(arguments)
    if (min(a.trajectory_count,a.shots,a.control_shots,a.batch_size)<2 or min(a.order_seed,a.seed_transpiler)<0
            or not a.backend.strip()):raise ValueError('need >=2 trajectories/shots/control-shots/batch-size and nonnegative seeds')
    if not (a.resume or a.analyze_only):
        if a.comparison=='mapping' and (a.alternative_path is None or len(set(a.alternative_path))!=len(a.alternative_path) or min(a.alternative_path)<0):
            raise ValueError('mapping comparison requires a distinct nonnegative --alternative-path')
        if a.comparison!='mapping' and a.alternative_path is not None:raise ValueError('alternative-path requires mapping comparison')
    return a


def validate_sources(sources,comparison='schedule'):
    if set(sources)!={'baseline','candidate'}:raise ValueError('need baseline and candidate inputs')
    for source in sources.values():
        hardware.require_validated_source(source)
        n=source['configuration']['physics']['n_qubits'];R=source['configuration']['trajectories']
        C=np.asarray(source['final_covariances_real_gauge'])
        if C.shape!=(1,R,n,n) or not np.allclose(C,C.swapaxes(-1,-2),atol=1e-12):
            raise ValueError('invalid saved ideal covariance shape/symmetry')
        eigenvalues=np.linalg.eigvalsh(C)
        if eigenvalues.min() < -1e-9 or eigenvalues.max()>1+1e-9:raise ValueError('unphysical saved ideal covariances')
    a,b=sources['baseline'],sources['candidate']
    for key in ['physics','sampling_policy','order','system_formula','seed_trajectories','trajectories']:
        if a['configuration'].get(key)!=b['configuration'].get(key):raise ValueError('paired physics or trajectory streams differ')
    if a['preparation']['populations']!=b['preparation']['populations']:raise ValueError('paired preparations differ')
    if comparison=='mapping':
        if a!=b:raise ValueError('mapping comparison requires identical sources, schedules and ideal paths')
        return
    if comparison!='schedule':raise ValueError('unknown comparison mode')
    count=len(b['schedule']['step_dts'])
    if count>=len(a['schedule']['step_dts']):raise ValueError('candidate must have fewer steps')
    if any(x[:count]!=y for x,y in zip(a['schedule']['jump_choices'],b['schedule']['jump_choices'])):
        raise ValueError('paired trajectories must share branch-choice prefixes')


def load_inputs(path):
    raw_bundle=path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_bundle)) as archive:
        if archive.testzip() is not None:raise ValueError('input bundle CRC failed')
        metadata=json.loads(archive.read('input_manifest.json'))
        raw={label:archive.read(metadata['sources'][label]['file']) for label in ['baseline','candidate']}
    if any(sha(value)!=metadata['sources'][label]['sha256'] for label,value in raw.items()):
        raise ValueError('input source hash mismatch')
    sources={label:json.loads(value) for label,value in raw.items()};validate_sources(sources)
    n=sources['baseline']['configuration']['physics']['n_qubits'];path=metadata['physical_path']
    if len(path)!=n+2 or len(set(path))!=len(path) or min(path)<0:raise ValueError('invalid saved physical chain')
    return sources,raw,metadata,sha(raw_bundle)


def plan(cfg,sources):
    rng=np.random.default_rng(cfg['order_seed']);count=cfg['trajectory_count'];n=len(sources['baseline']['preparation']['populations'])
    controls={'all_zero':[0.]*n,'all_one':[1.]*n,'initial_mixture':sources['baseline']['preparation']['populations']}
    def control(name):
        if cfg.get('comparison','schedule')=='mapping':
            return [dict(case_id=f'{label}_{name}',kind='control',control=name,schedule=label,basis='z',
                shots=cfg['control_shots'],expected_populations=controls[name]) for label in rng.permutation(['baseline','candidate'])]
        return [dict(case_id=name,kind='control',control=name,basis='z',shots=cfg['control_shots'],expected_populations=controls[name])]
    units=[control('all_zero')]
    for k,index in enumerate(rng.permutation(count)):
        if k==count//2:units.append(control('all_one'))
        pair=[]
        for label in rng.permutation(['baseline','candidate']):
            pair.append(dict(case_id=f'{label}_{index}',kind='trajectory',schedule=str(label),trajectory=int(index),basis='z',shots=cfg['shots']))
        units.append(pair)
    units.append(control('initial_mixture'))
    records=[];batches=[];start=0
    for unit in units:
        if len(records)-start+len(unit)>cfg['batch_size']:
            batches.append(dict(start=start,stop=len(records),status='not_submitted',job_id=None,compiled=False));start=len(records)
        for row in unit:records.append(dict(row,pub_index=len(records)))
    batches.append(dict(start=start,stop=len(records),status='not_submitted',job_id=None,compiled=False))
    return records,batches


def record_path(cfg,record):
    if cfg.get('comparison','schedule')=='mapping':return cfg['physical_paths'][record['schedule']]
    return cfg['physical_path']


def verify_plan(manifest,sources):
    if manifest.get('experiment')!=EXPERIMENT:raise ValueError('need paired schedule manifest')
    cfg=manifest['configuration'];R=cfg['trajectory_count']
    if cfg.get('comparison','schedule')=='mapping':
        validate_sources(sources,'mapping')
        paths=cfg['physical_paths'];size=len(sources['baseline']['preparation']['populations'])+2
        if set(paths)!={'baseline','candidate'} or paths['baseline']==paths['candidate'] or paths['candidate']==paths['baseline'][::-1]:
            raise ValueError('need two different physical chains, not a reversal')
        if cfg['physical_path']!=paths['baseline']:raise ValueError('baseline path changed')
        if any(len(path)!=size or len(set(path))!=size or min(path)<0 for path in paths.values()):raise ValueError('invalid comparison chains')
    if R<2 or R>sources['baseline']['configuration']['trajectories'] or min(cfg['shots'],cfg['control_shots'],cfg['batch_size'])<2:
        raise ValueError('invalid saved workload')
    expected,batches=plan(cfg,sources)
    if len(expected)!=len(manifest['circuits']) or len(batches)!=len(manifest['batches']):raise ValueError('incomplete paired plan')
    for a,b in zip(expected,manifest['circuits']):
        if any(b.get(k)!=v for k,v in a.items()):raise ValueError('paired case order/settings changed')
        if b.get('final_logical_to_physical') is not None:
            path=record_path(cfg,b);desired=path[1:-1]+[path[0],path[-1]]
            if b['final_logical_to_physical']!=desired:raise ValueError('compiled chain mapping changed')
    for a,b in zip(batches,manifest['batches']):
        if a['start']!=b['start'] or a['stop']!=b['stop']:raise ValueError('paired batch coverage changed')
    if cfg['total_circuits']!=len(expected) or cfg['total_shots']!=sum(r['shots'] for r in expected):
        raise ValueError('paired circuit/shot budget mismatch')


def read_sources(folder,cfg):
    sources={};integrity={}
    for label in ['baseline','candidate']:
        raw=(folder/f'{label}.json').read_bytes()
        integrity[label]=source_integrity(raw,cfg['source_sha256'][label]);sources[label]=json.loads(raw)
    validate_sources(sources,cfg.get('comparison','schedule'))
    return sources,integrity


def compiled_batch(folder,manifest,batch,sources,backend=None):
    cfg=manifest['configuration']
    filename=folder/f"circuits_{batch['start']:06d}_{batch['stop']:06d}.qpy"
    records=manifest['circuits'][batch['start']:batch['stop']]
    if batch.get('compiled'):
        raw=filename.read_bytes()
        if sha(raw)!=batch['isa_sha256']:raise ValueError('compiled batch hash mismatch')
        pubs=qpy.load(io.BytesIO(raw))
        if len(pubs)!=len(records):raise ValueError('compiled batch length mismatch')
        for pub,record in zip(pubs,records):
            path=record_path(cfg,record)
            if pub.metadata.get('comparison_case_id')!=record['case_id'] or circuits.metrics(pub)!=record['metrics']:
                raise ValueError('compiled case metadata/metrics mismatch')
            measured=[(pub.find_bit(x.qubits[0]).index,pub.find_bit(x.clbits[0]).index)
                      for x in pub.data if x.operation.name=='measure' and x.clbits[0] in next(r for r in pub.cregs if r.name=='outcome')]
            outcome=next(r for r in pub.cregs if r.name=='outcome')
            if measured!=[(q,pub.find_bit(outcome[j]).index) for j,q in enumerate(path[1:-1])]:
                raise ValueError('compiled final physical readout changed')
    else:
        if backend is None:raise ValueError('backend required to compile a missing batch')
        pubs=[]
        for record in records:
            path=record_path(cfg,record);mapping=path[1:-1]+[path[0],path[-1]]
            if record['kind']=='control':
                source=sources[record.get('schedule','baseline')]
                source={**source,'preparation':{**source['preparation'],'populations':record['expected_populations']}}
                logical=circuits.build_body(source,0,steps=0)
            else:logical=circuits.build_body(sources[record['schedule']],record['trajectory'])
            native=transpile(logical,backend=backend,initial_layout=mapping,optimization_level=cfg['optimization_level'],seed_transpiler=cfg['seed_transpiler'])
            if native.layout.final_index_layout(filter_ancillas=True)!=mapping:raise ValueError('transpiler changed fixed chain')
            pub=circuits.append_readout(native,mapping[:-2],'z')
            pub.metadata={**(pub.metadata or {}),'comparison_case_id':record['case_id']}
            circuits.assert_native(pub,backend)
            record.update(metrics=circuits.metrics(pub),final_logical_to_physical=mapping)
            pubs.append(pub)
            if len(pubs)%16==0 or len(pubs)==len(records):
                print(f"  Compiled {len(pubs)}/{len(records)} circuits in current batch",flush=True)
        with filename.open('wb') as stream:qpy.dump(pubs,stream)
        batch.update(compiled=True,isa_file=filename.name,isa_sha256=sha(filename.read_bytes()))
        atomic_json(folder/'manifest.json',manifest)
    if backend is not None:
        for pub in pubs:circuits.assert_native(pub,backend)
    return pubs


def analyze(folder):
    manifest=json.loads((folder/'manifest.json').read_bytes());cfg=manifest['configuration']
    sources,integrity=read_sources(folder,cfg);verify_plan(manifest,sources)
    if manifest['status']!='collected':raise ValueError('need completely collected paired results')
    n=sources['baseline']['configuration']['physics']['n_qubits'];R=cfg['trajectory_count']
    paths={label:{} for label in sources};controls=[];jobs=[];qpy_checked=[];missing=[];total=0
    for batch in manifest['batches']:
        if batch['status']!='collected' or not batch.get('job_id') or not batch.get('compiled'):
            raise ValueError('uncollected or uncompiled paired batch')
        records=manifest['circuits'][batch['start']:batch['stop']]
        if len(batch.get('observations',[]))!=len(records):raise ValueError('missing paired observations')
        if (folder/batch['isa_file']).exists():
            compiled_batch(folder,manifest,batch,sources);qpy_checked.append(batch['isa_file'])
        else:missing.append(batch['isa_file'])
        jobs.append(batch['job_id'])
        for record,obs in zip(records,batch['observations']):
            population,covariance,shots=moments(obs['outcome_counts'],n)
            if shots!=record['shots']:raise ValueError('paired raw shot count mismatch')
            total+=shots
            if record['kind']=='control':
                controls.append(dict(control=record['control'],populations=population.tolist(),expected_populations=record['expected_populations'],
                    shots=shots,physical_sites=record_path(cfg,record)[1:-1],pub_index=record['pub_index'],
                    **({'mapping':record['schedule']} if cfg.get('comparison')=='mapping' else {})))
            else:
                if record['trajectory'] in paths[record['schedule']]:raise ValueError('duplicate paired path')
                paths[record['schedule']][record['trajectory']]=(population,covariance)
    if total!=cfg['total_shots']:raise ValueError('paired total shots differ')
    summaries={};residuals={}
    physical=np.asarray(sources['baseline']['stationary_observables']['populations'])
    for label,source in sources.items():
        if set(paths[label])!=set(range(R)):raise ValueError('missing paired trajectories')
        values=np.asarray([paths[label][i][0] for i in range(R)])
        shot_cov=sum(paths[label][i][1] for i in range(R))/R**2
        ideal_paths=np.diagonal(np.asarray(source['final_covariances_real_gauge'])[0,:R],axis1=-2,axis2=-1)
        ideal=ideal_paths.mean(axis=0);mean=values.mean(axis=0);residuals[label]=values-ideal_paths
        covariance=np.cov(values,rowvar=False,ddof=1)/R
        summaries[label]=dict(hardware_populations=mean,ideal_same_paths=ideal,physical_stationary=physical,
            ideal_mean_channel=np.asarray(source['mean_channel_observables']['populations'])[0,-1],
            per_path_populations=values,per_path_ideal_populations=ideal_paths,
            across_path_se=np.sqrt(np.maximum(0.,np.diag(covariance))),fixed_paths_shot_se=np.sqrt(np.maximum(0.,np.diag(shot_cov))),
            errors_vs_physical=population_error(mean,physical),errors_vs_own_ideal=population_error(mean,ideal),
            ideal_error_vs_physical=population_error(ideal,physical))
        records=[r for r in manifest['circuits'] if r.get('schedule')==label and r['kind']=='trajectory']
        summaries[label]['resources']=dict(steps=len(source['schedule']['step_dts']),
            cz_counts=sorted({r['metrics']['operation_counts'].get('cz',0) for r in records}),
            reset_counts=sorted({r['metrics']['operation_counts'].get('reset',0) for r in records}),
            depth_range=[min(r['metrics']['depth'] for r in records),max(r['metrics']['depth'] for r in records)])
    paired=residuals['candidate']-residuals['baseline']
    paired_cov=np.cov(paired,rowvar=False,ddof=1)/R
    assessment=control_quality.assess_controls(controls) if cfg.get('comparison')!='mapping' else None
    if assessment is None:
        per_mapping={label:control_quality.assess_controls([c for c in controls if c['mapping']==label]) for label in sources}
        assessment=dict(passed=all(a['passed'] for a in per_mapping.values()),by_mapping=per_mapping,
            failed_sites_1based={label:a['failed_sites_1based'] for label,a in per_mapping.items()})
    return dict(experiment=EXPERIMENT,run_directory=str(folder.resolve()),configuration=cfg,job_ids=jobs,
        schedules=summaries,controls=controls,control_assessment=assessment,
        paired_residual_difference=paired.mean(axis=0),paired_residual_se=np.sqrt(np.maximum(0.,np.diag(paired_cov))),
        verification=dict(source_integrity=integrity,compiled_batches_verified=qpy_checked,compiled_batches_not_available=missing,
                          raw_counts_reconstructed=True),
        scope='Populations only. No hardware currents, readout correction or accuracy guarantee. Across-path errors include shots but omit drift; adjacent PUB order is requested, not verified hardware dispatch order.')


def save(report,output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.resolve().is_relative_to(Path(report['run_directory']).resolve()):raise ValueError('analysis must be outside raw run')
    output.mkdir(parents=True,exist_ok=True)
    mapping_comparison=report['configuration'].get('comparison')=='mapping'
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');rows=[]
    for ax,(label,summary) in zip(axes,report['schedules'].items()):
        sites=np.arange(1,len(summary['hardware_populations'])+1)
        ax.errorbar(sites,summary['hardware_populations'],yerr=2*summary['across_path_se'],fmt='o-',capsize=3,label='Hardware ±2 across-path SE')
        ax.plot(sites,summary['ideal_same_paths'],'s--',label='Ideal: same saved paths')
        ax.plot(sites,summary['physical_stationary'],'k:',label='Physical stationary target')
        name=('current chain' if label=='baseline' else 'alternative chain') if mapping_comparison else label
        ax.set(xlabel='Site',ylabel='Excitation population',title=f"{name}: {summary['resources']['steps']} steps",xticks=sites)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        for j,site in enumerate(sites):rows.append(dict(schedule=label,site=int(site),hardware=float(summary['hardware_populations'][j]),
            ideal_same_paths=float(summary['ideal_same_paths'][j]),physical_target=float(summary['physical_stationary'][j]),
            across_path_se=float(summary['across_path_se'][j])))
    for ext in ['png','pdf']:fig.savefig(output/f'populations.{ext}',dpi=170)
    plt.close(fig)
    with (output/'populations.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Paired '+('mapping' if mapping_comparison else 'schedule')+' population comparison','',f"Jobs: {', '.join(report['job_ids'])}. Total shots: {report['configuration']['total_shots']}.",'',
        '![Population profiles](populations.png)','',
        '| Schedule | Steps | CZ count | Hardware max error vs physical | Hardware max error vs own ideal | Ideal max error vs physical |',
        '|---|---:|---|---:|---:|---:|']
    for label,s in report['schedules'].items():text.append(f"| {label} | {s['resources']['steps']} | {s['resources']['cz_counts']} | {s['errors_vs_physical']['maximum']:.6f} | {s['errors_vs_own_ideal']['maximum']:.6f} | {s['ideal_error_vs_physical']['maximum']:.6f} |")
    primary=('Primary mapping comparison: hardware discrepancy from the common ideal saved paths. '
        'Also report population error to the physical stationary target; 0.02 is not certified by this workload. ' if mapping_comparison else
        'Primary comparison: all-site population error to the physical stationary target (target 0.02). ')
    text+=['',primary+
        'Own-ideal errors separate finite-path preparation bias from device discrepancy. Report both; do not infer improvement merely from circuit depth. '
        'Means use the same paths and shots. Controls are interleaved for context; they do not automatically stop the experiment or apply mitigation.','',
        f"Preparation-control screen passed: **{report['control_assessment']['passed']}**. Failed sites: {report['control_assessment']['failed_sites_1based']}.",'',
        report['scope'],'','Compiled QPY batches are retained locally. The small results bundle omits them; verification states which batches were available. '
        'Retain those local files for later circuit inspection. No permanent settling or noisy-hardware accuracy is certified by the input large-ensemble checks.','']
    if mapping_comparison:
        text+=['','Mapping test: baseline is the current chain; candidate is the alternative chain. Both use the identical candidate schedule, preparation and branch paths. '
            'The primary mapping comparison is discrepancy from that common ideal; physical-target errors remain a secondary reported outcome. '
            'Per-chain controls are diagnostic context, not an automatic stopping rule or mitigation.','',
            f"Physical paths: `{report['configuration']['physical_paths']}`.",'',
            'Sampling precision here does not certify the original 0.02 error budget. Paired estimates condition on the saved branch paths and omit drift.','']
    (output/'Analysis.md').write_text('\n'.join(text),encoding='utf-8')
    atomic_json(output/'analysis.json',serializable(report))


def bundle(folder,output=None):
    with zipfile.ZipFile(folder/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['manifest.json','baseline.json','candidate.json','input_manifest.json','backend_snapshot.json','hardware_results.json']:
            if (folder/name).exists():archive.write(folder/name,name)
        if output:
            for path in Path(output).iterdir():
                if path.is_file():archive.write(path,'analysis/'+path.name)


def main(options,*,backend=None,service=None,sampler_factory=None):
    if options.analyze_only:
        report=analyze(options.analyze_only)
        kind='mapping_comparison' if report['configuration'].get('comparison')=='mapping' else 'schedule_comparison'
        output=options.analysis_output_directory or DIRECTORY/'analysis/hardware'/options.analyze_only.name/kind
        save(report,output);print(f"Analysis: {output/'Analysis.md'}");return report
    previous_verification=None
    if options.verify_previous_run:
        previous=analyze(options.verify_previous_run)
        if previous['verification']['compiled_batches_not_available']:
            raise ValueError('previous native QPY chunks are missing; verify on the original execution machine with its circuits_*.qpy files')
        previous_verification=dict(run=options.verify_previous_run.name,
            manifest_sha256=sha((options.verify_previous_run/'manifest.json').read_bytes()),
            compiled_batches=previous['verification']['compiled_batches_verified'])
        print('Previous native QPY chunks verified locally:',len(previous_verification['compiled_batches']),flush=True)
    if options.resume:
        folder=options.resume.resolve().parent;manifest=json.loads(options.resume.read_bytes());cfg=manifest['configuration']
        sources,_=read_sources(folder,cfg);verify_plan(manifest,sources)
    else:
        sources,raw,metadata,input_hash=load_inputs(options.input_bundle)
        if options.comparison=='mapping':
            selected_raw=raw['candidate'];selected_source=sources['candidate']
            sources={label:copy.deepcopy(selected_source) for label in ['baseline','candidate']}
            raw={label:selected_raw for label in sources}
            metadata=copy.deepcopy(metadata)
            metadata['comparison']='mapping';metadata['original_input_source']='candidate'
            metadata['sources']={label:dict(file=label+'.json',sha256=sha(selected_raw)) for label in sources}
        if options.backend!=metadata['backend']:raise ValueError('backend must match paired input chain')
        if options.trajectory_count>sources['baseline']['configuration']['trajectories']:raise ValueError('trajectory count exceeds saved paired inputs')
        cfg=dict(backend=options.backend,physical_path=metadata['physical_path'],source_sha256={k:sha(v) for k,v in raw.items()},
            input_bundle_sha256=input_hash,trajectory_count=options.trajectory_count,shots=options.shots,control_shots=options.control_shots,
            batch_size=options.batch_size,order_seed=options.order_seed,seed_transpiler=options.seed_transpiler,optimization_level=options.optimization_level)
        cfg['comparison']=options.comparison
        if previous_verification is not None:cfg['previous_qpy_verification']=previous_verification
        cfg['software_versions']={name:importlib.metadata.version(name) for name in ['qiskit','qiskit-ibm-runtime','numpy']}
        if options.comparison=='mapping':
            cfg['physical_paths']=dict(baseline=metadata['physical_path'],candidate=options.alternative_path)
            cfg['sampling_scope']='Fixed common saved paths; mapping diagnostic only. No 0.02 sampling or hardware accuracy certification.'
        records,batches=plan(cfg,sources);cfg.update(total_circuits=len(records),total_shots=sum(r['shots'] for r in records))
        manifest=dict(experiment=EXPERIMENT,status='planned',configuration=cfg,circuits=records,batches=batches)
        verify_plan(manifest,sources)
    pending=not options.resume or any(b.get('observations') is None for b in manifest['batches'])
    if pending:
        if backend is None:
            service=service or hardware.make_service(options);backend=service.backend(cfg['backend'],use_fractional_gates=False)
        if backend.name!=cfg['backend']:raise ValueError('backend differs from saved paired run')
        for path in cfg.get('physical_paths',{'baseline':cfg['physical_path']}).values():
            circuits.chain_layout(backend,sources['baseline']['configuration']['physics']['n_qubits'],'boundary',path)
    if not options.resume:
        refresh=getattr(backend,'refresh',None)
        if callable(refresh):refresh()
        if cfg.get('comparison')=='mapping':
            nodes,edges=quality_path.calibration(backend,{7},80e-6,60e-6,.005,.02)
            weights=quality_path.gate_weights(sources['baseline'],list(range(cfg['trajectory_count'])))
            cfg['path_calibration']={}
            for label,path in cfg['physical_paths'].items():
                if not set(path)<=nodes.keys() or any(tuple(sorted(pair)) not in edges for pair in zip(path,path[1:])):
                    raise ValueError(f'{label} chain fails current coherence/CZ/readout filters; no jobs submitted')
                _,cfg['path_calibration'][label]=quality_path.score(path,nodes,edges,weights)
            print('Current calibration metrics for both chains:',cfg['path_calibration'],flush=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        name='mapping_comparison' if cfg.get('comparison')=='mapping' else 'comparison'
        folder=options.output_directory/f"{cfg['backend']}_{name}_{stamp}";folder.mkdir(parents=True,exist_ok=False)
        for label,value in raw.items():(folder/f'{label}.json').write_bytes(value)
        atomic_json(folder/'input_manifest.json',metadata)
        active=sorted({q for path in cfg['physical_paths'].values() for q in path}) if cfg.get('comparison')=='mapping' else cfg['physical_path']
        hardware.save_backend_snapshot(backend,active,manifest['circuits'],folder)
        if cfg.get('comparison')=='mapping':
            snapshot=json.loads((folder/'backend_snapshot.json').read_bytes())
            snapshot.update(physical_path=None,physical_paths=cfg['physical_paths'],active_qubits=active)
            atomic_json(folder/'backend_snapshot.json',snapshot)
        atomic_json(folder/'manifest.json',manifest)
    print(f"Paired experiment: {cfg['total_circuits']} circuits, {cfg['total_shots']} shots, {len(manifest['batches'])} batches. {folder}",flush=True)
    sampler=None
    for number,batch in enumerate(manifest['batches']):
        if batch.get('observations') is not None:continue
        if options.compile_only:
            compiled_batch(folder,manifest,batch,sources,backend)
            print(f"Compiled batch {number+1}/{len(manifest['batches'])}",flush=True);continue
        if batch.get('job_id'):
            if service is None:raise ValueError('service required to recover submitted job')
            job=service.job(batch['job_id'])
        else:
            if batch.get('status')=='submitting':
                raise ValueError('submission interrupted before job ID was saved; find that job on IBM Quantum and add its job_id before resuming')
            print(f"Compiling batch {number+1}/{len(manifest['batches'])} before submission...",flush=True)
            pubs=compiled_batch(folder,manifest,batch,sources,backend)
            if sampler is None:
                if sampler_factory is None:
                    from qiskit_ibm_runtime import SamplerV2
                    sampler_factory=SamplerV2
                sampler=sampler_factory(mode=backend,options={'dynamical_decoupling':{'enable':False},
                    'twirling':{'enable_gates':False,'enable_measure':False},'execution':{'init_qubits':True}})
            records=manifest['circuits'][batch['start']:batch['stop']]
            selected=[(pub,None,r['shots']) for pub,r in zip(pubs,records)]
            batch['status']='submitting';atomic_json(folder/'manifest.json',manifest)
            job=sampler.run(selected,shots=cfg['shots'])
            batch.update(job_id=job.job_id(),status='submitted');atomic_json(folder/'manifest.json',manifest)
            print(f"Submitted batch {number+1}: {batch['job_id']}",flush=True)
        result=job.result();records=manifest['circuits'][batch['start']:batch['stop']]
        if len(result)!=len(records):raise ValueError('unexpected Runtime result count')
        observations=[]
        for pub,record in zip(result,records):
            counts={str(k):int(v) for k,v in pub.data.outcome.get_counts().items()}
            _,_,shots=moments(counts,len(cfg['physical_path'])-2)
            if shots!=record['shots']:raise ValueError('Runtime shot count mismatch')
            observations.append(dict(outcome_counts=counts))
        batch.update(observations=observations,status='collected',collected_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            metrics=job.metrics()
            if isinstance(metrics,dict):batch['job_metrics']=json.loads(json.dumps(metrics,default=str))
        except Exception as error:batch['job_metrics_unavailable']=str(error)
        atomic_json(folder/'manifest.json',manifest)
    if options.compile_only:
        manifest['status']='compiled';atomic_json(folder/'manifest.json',manifest);bundle(folder)
        print('Compile only: no jobs submitted. Resume this manifest to execute.',flush=True);return manifest
    manifest['status']='collected';atomic_json(folder/'manifest.json',manifest)
    report=analyze(folder);atomic_json(folder/'hardware_results.json',serializable(report))
    kind='mapping_comparison' if cfg.get('comparison')=='mapping' else 'schedule_comparison'
    output=options.analysis_output_directory or DIRECTORY/'analysis/hardware'/folder.name/kind
    save(report,output);bundle(folder,output)
    print(f"Analysis: {output/'Analysis.md'}\nBring back: {folder/'results_bundle.zip'}\nRetain local circuits_*.qpy files; they are omitted from the small results bundle.",flush=True)
    return report


if __name__=='__main__':
    try:main(parse_arguments())
    except (ValueError,OSError) as error:raise SystemExit(str(error)) from error
