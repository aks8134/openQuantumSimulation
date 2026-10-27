"""First-order Model3 mitigation pilot; default is an entirely offline plan.

Explicit --execute or --resume is required to submit. CZ twirling/folding is
client-side; Runtime twirling and DD remain disabled. Reservoir resets are
never folded. No account is accessed at import, planning or analysis time.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import io
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
from Model3.Experiment5 import circuit_depth as circuits,mitigation as mit,run_hardware as hardware,quality_path
from Model3.Experiment5.run_schedule_comparison import DEFAULT_INPUT,load_inputs
from Model3.Experiment5.analyze_hardware import moments,serializable,source_integrity

EXPERIMENT='Model3/Experiment5/mitigation_pilot_v1'
DEFAULT_PATH=[49,39,38,48,58,57,56,66,76,86,85,84]
SAMPLER_OPTIONS={'dynamical_decoupling':{'enable':False},'twirling':{'enable_gates':False,'enable_measure':False},'execution':{'init_qubits':True}}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def parse_arguments(arguments=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-bundle',type=Path,default=DEFAULT_INPUT);p.add_argument('--backend',default='ibm_phoenix')
    p.add_argument('--account-file',type=Path);p.add_argument('--physical-path',type=int,nargs='+',default=DEFAULT_PATH)
    p.add_argument('--trajectory-count',type=int,default=64)
    p.add_argument('--shots',type=int,default=256,help='total shots per trajectory per arm/noise factor')
    p.add_argument('--randomizations',type=int,default=4,help='even count; balanced terminal X masks')
    p.add_argument('--control-shots',type=int,default=2048,help='total per preparation per raw/twirled treatment')
    p.add_argument('--controls-per-batch',action='store_true',help='complete zero/one/mixture controls and local calibration inside every job')
    p.add_argument('--calibration-policy',choices=('screen','record-only'),default='screen',
        help='screen applies chain quality thresholds; record-only retains the specified chain and records its calibration')
    p.add_argument('--fold-factors',type=float,nargs='+',default=[1.,1.25,1.5])
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=29)
    p.add_argument('--seed-transpiler',type=int,default=11);p.add_argument('--optimization-level',type=int,choices=range(4),default=3)
    p.add_argument('--bootstrap',type=int,default=2000)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--plan-only',action='store_true');action.add_argument('--compile-only',action='store_true')
    action.add_argument('--execute',action='store_true');action.add_argument('--resume',type=Path);action.add_argument('--analyze-only',type=Path)
    p.add_argument('--output-directory',type=Path,default=DIRECTORY/'hardware_runs')
    p.add_argument('--analysis-output-directory',type=Path)
    a=p.parse_args(arguments)
    if not (a.execute or a.compile_only or a.resume or a.analyze_only):a.plan_only=True
    if min(a.seed,a.seed_transpiler)<0 or a.bootstrap<100:raise ValueError('invalid seed/bootstrap count')
    if not (a.resume or a.analyze_only):
        if a.trajectory_count<2 or a.randomizations<2 or a.randomizations%2:raise ValueError('need >=2 paths and even randomizations >=2')
        if min(a.shots,a.control_shots)<2*a.randomizations or a.shots%a.randomizations or a.control_shots%a.randomizations:
            raise ValueError('shots/control-shots must be divisible by randomizations, with >=2 per variant')
        factors=a.fold_factors
        if len(factors)<3 or factors[0]!=1 or any(not np.isfinite(f) or f<1 or f>3 for f in factors) or any(b<=a for a,b in zip(factors,factors[1:])):
            raise ValueError('need >=3 strictly increasing factors starting at 1, each <=3')
        if a.batch_size<max(1+a.randomizations*len(factors),1+a.randomizations):raise ValueError('batch too small to keep a whole matched trajectory group')
        if a.controls_per_batch and a.batch_size<3*(1+a.randomizations)+2*(1+a.randomizations*len(factors)):
            raise ValueError('batch needs complete controls and at least two matched trajectories')
        if not a.backend.strip() or min(a.physical_path)<0 or len(set(a.physical_path))!=len(a.physical_path):raise ValueError('invalid backend/path')
    return a


def plan(cfg):
    rng=np.random.default_rng(cfg['seed']);R=cfg['trajectory_count'];K=cfg['randomizations'];n=len(cfg['physical_path'])-2
    masks=mit.balanced_masks(n,K,cfg['seed']);base=2*n*cfg['steps']
    folds=[mit.fold_count(f,base) for f in cfg['fold_factors']]
    actual=[1+2*k/base for k in folds];mit.zne_weights(actual)
    def control(name,prefix=''):
        expected={'all_zero':[0.]*n,'all_one':[1.]*n,'initial_mixture':[.5]*n}[name]
        rows=[dict(kind='control',control=name,arm='raw',randomization=None,measurement_mask=0,
            shots=cfg['control_shots'],expected_populations=expected,case_id=f'{prefix}raw_{name}')]
        rows.extend(dict(kind='control',control=name,arm='twirled',randomization=v,measurement_mask=masks[v],
            shots=cfg['control_shots']//K,expected_populations=expected,case_id=f'{prefix}twirled_{name}_{v}') for v in range(K))
        return [rows[i] for i in rng.permutation(len(rows))]
    def trajectory(index):
        rows=[dict(kind='trajectory',trajectory=int(index),arm='raw',factor_index=0,requested_factor=1.,actual_factor=1.,
            folds=0,randomization=None,measurement_mask=0,shots=cfg['shots'],case_id=f'raw_{index}')]
        for k,(f,c,x) in enumerate(zip(cfg['fold_factors'],folds,actual)):
            for v in range(K):rows.append(dict(kind='trajectory',trajectory=int(index),arm='twirled',factor_index=k,
                requested_factor=f,actual_factor=x,folds=c,randomization=v,measurement_mask=masks[v],shots=cfg['shots']//K,
                case_id=f'twirled_{index}_{k}_{v}'))
        return [rows[i] for i in rng.permutation(len(rows))]
    if cfg.get('controls_per_batch',False):
        capacity=(cfg['batch_size']-3*(1+K))//(1+K*len(folds))
        if capacity<2:raise ValueError('batch needs complete controls and at least two matched trajectories')
        groups=np.array_split(rng.permutation(R),(R+capacity-1)//capacity)
        if any(len(g)<2 for g in groups):raise ValueError('need at least two paths per job; increase batch-size or adjust trajectory-count')
        records=[];batches=[]
        for job,indices in enumerate(groups):
            start=len(records);prefix=f'job{job}_'
            units=control('all_zero',prefix)
            for pos,index in enumerate(indices):
                if pos==len(indices)//2:units.extend(control('all_one',prefix))
                units.extend(trajectory(index))
            units.extend(control('initial_mixture',prefix))
            records.extend(dict(row,pub_index=start+i) for i,row in enumerate(units))
            batches.append(dict(start=start,stop=len(records),status='not_submitted',job_id=None,compiled=False))
        return records,batches
    units=[control('all_zero')]
    for pos,index in enumerate(rng.permutation(R)):
        if pos==R//2:units.append(control('all_one'))
        units.append(trajectory(index))
    units.append(control('initial_mixture'));records=[];batches=[];start=0
    for unit in units:
        if len(records)-start+len(unit)>cfg['batch_size']:
            batches.append(dict(start=start,stop=len(records),status='not_submitted',job_id=None,compiled=False));start=len(records)
        for row in unit:records.append(dict(row,pub_index=len(records)))
    batches.append(dict(start=start,stop=len(records),status='not_submitted',job_id=None,compiled=False))
    return records,batches


def verify_manifest(manifest,source):
    if manifest.get('experiment')!=EXPERIMENT:raise ValueError('not a mitigation manifest')
    cfg=manifest['configuration'];hardware.require_validated_source(source)
    if cfg.get('calibration_policy','screen') not in ('screen','record-only'):raise ValueError('unknown calibration policy')
    if cfg['sampler_options']!=SAMPLER_OPTIONS:raise ValueError('saved Runtime options differ from explicit manual mitigation protocol')
    n=source['configuration']['physics']['n_qubits']
    if (cfg['trajectory_count']>source['configuration']['trajectories'] or cfg['steps']!=len(source['schedule']['step_dts'])
        or len(cfg['physical_path'])!=n+2 or source['preparation']['populations']!=[.5]*n):raise ValueError('source/plan mismatch')
    expected,batches=plan(cfg)
    if len(expected)!=len(manifest['circuits']) or len(batches)!=len(manifest['batches']):raise ValueError('incomplete mitigation plan')
    desired=cfg['physical_path'][1:-1]+[cfg['physical_path'][0],cfg['physical_path'][-1]]
    for a,b in zip(expected,manifest['circuits']):
        if any(b.get(k)!=v for k,v in a.items()):raise ValueError('mitigation cases/shot budget changed')
        if 'final_logical_to_physical' in b and b['final_logical_to_physical']!=desired:raise ValueError('mapping changed')
    for a,b in zip(batches,manifest['batches']):
        if (a['start'],a['stop'])!=(b['start'],b['stop']):raise ValueError('batch coverage changed')
        if b['status'] not in ('not_submitted','submitting','submitted','collected'):raise ValueError('unknown saved batch state')
        if b['status'] in ('submitted','collected') and not b.get('job_id'):raise ValueError('submitted/collected batch lacks a job ID; reconcile with IBM')
    if cfg['total_circuits']!=len(expected) or cfg['total_shots']!=sum(r['shots'] for r in expected):raise ValueError('workload changed')


def read_source(folder,manifest):
    raw=(folder/'source.json').read_bytes();integrity=source_integrity(raw,manifest['configuration']['source_sha256'])
    source=json.loads(raw);verify_manifest(manifest,source);return source,integrity


def template_key(record):return (record['kind'],record.get('trajectory',record.get('control')))


def transform_seed(cfg,record):
    # Stable across noise factors; raw/control variants are kept distinct.
    index=record.get('trajectory',{'all_zero':0,'all_one':1,'initial_mixture':2}.get(record.get('control'),0))
    return int(np.random.SeedSequence([cfg['seed'],index,record['randomization'],0 if record['kind']=='trajectory' else 1]).generate_state(1)[0])


def assert_supported(circuit,backend):
    for item in circuit.data:
        if item.operation.name=='barrier':continue
        args=tuple(circuit.find_bit(q).index for q in item.qubits)
        if not backend.target.instruction_supported(operation_name=item.operation.name,qargs=args):
            raise ValueError(f'unsupported native mitigation operation {item.operation.name}{args}')


def make_variant(native,cfg,record):
    result=native.copy() if record['arm']=='raw' else mit.transformed(native,seed=transform_seed(cfg,record),
        folds=record.get('folds',0),measurement_mask=record['measurement_mask'])
    result.metadata={**(result.metadata or {}),'mitigation_case_id':record['case_id']}
    return result


def verify_circuits(pubs,records,cfg):
    if len(pubs)!=len(records):raise ValueError('native PUB length mismatch')
    raw={template_key(r):p for p,r in zip(pubs,records) if r['arm']=='raw'}
    for pub,record in zip(pubs,records):
        if pub.metadata.get('mitigation_case_id')!=record['case_id'] or circuits.metrics(pub)!=record['metrics']:
            raise ValueError('native case/metrics mismatch')
        expected=make_variant(raw[template_key(record)],cfg,record)
        if pub!=expected:raise ValueError('twirl/folding/preparation/reset contents changed')
        outcome=next(r for r in pub.cregs if r.name=='outcome')
        measured=[(pub.find_bit(i.qubits[0]).index,list(outcome).index(i.clbits[0])) for i in pub.data
            if i.operation.name=='measure' and i.clbits[0] in outcome]
        if measured!=list(zip(cfg['physical_path'][1:-1],range(len(outcome)))):raise ValueError('final readout mapping changed')


def compiled_batch(folder,manifest,batch,source,backend=None):
    cfg=manifest['configuration'];records=manifest['circuits'][batch['start']:batch['stop']]
    path=folder/f"circuits_{batch['start']:06d}_{batch['stop']:06d}.qpy"
    if batch['compiled']:
        raw=path.read_bytes()
        if sha(raw)!=batch['isa_sha256']:raise ValueError('native QPY hash mismatch')
        pubs=qpy.load(io.BytesIO(raw));verify_circuits(pubs,records,cfg)
    else:
        if backend is None:raise ValueError('backend required for compilation')
        templates={};n=source['configuration']['physics']['n_qubits'];layout=cfg['physical_path'][1:-1]+[cfg['physical_path'][0],cfg['physical_path'][-1]]
        for record in records:
            key=template_key(record)
            if key in templates:continue
            control=record['kind']=='control'
            data=source if not control else {**source,'preparation':{**source['preparation'],'populations':record['expected_populations']}}
            logical=circuits.build_body(data,record.get('trajectory',0),steps=0 if control else None)
            native=transpile(logical,backend=backend,initial_layout=layout,optimization_level=cfg['optimization_level'],seed_transpiler=cfg['seed_transpiler'])
            if native.layout.final_index_layout(filter_ancillas=True)!=layout:raise ValueError('fixed layout changed during compilation')
            native=circuits.append_readout(native,layout[:n],'z')
            if not control and (native.count_ops().get('cz',0)!=2*n*cfg['steps'] or native.count_ops().get('reset',0)!=cfg['steps']):
                raise ValueError('compiled CZ/reset count differs from planned fixed-channel circuit')
            templates[key]=native
        pubs=[]
        for j,record in enumerate(records):
            pub=make_variant(templates[template_key(record)],cfg,record);assert_supported(pub,backend)
            record.update(metrics=circuits.metrics(pub),final_logical_to_physical=layout)
            pubs.append(pub)
            if (j+1)%32==0 or j==len(records)-1:print(f'Compiled {j+1}/{len(records)} mitigation variants',flush=True)
        verify_circuits(pubs,records,cfg)
        with path.open('wb') as stream:qpy.dump(pubs,stream)
        batch.update(compiled=True,isa_file=path.name,isa_sha256=sha(path.read_bytes()))
        atomic_json(folder/'manifest.json',manifest)
    if backend is not None:
        for pub in pubs:assert_supported(pub,backend)
    return pubs


def combine_job_calibrations(statistics,values,covariances,controls_by_job,groups,factors,ideal,target,*,replicates,seed,job_ids):
    """Calibrate each job separately, then average with its trajectory weight.

    Independent local bootstrap draws preserve shared calibration uncertainty
    across noise factors inside a job. Job weights are fixed, matching the
    stratified trajectory bootstrap. No pooled response inversion is used.
    """
    local=[];draws=[];ideal_draws=[];fractions=[]
    for k,(indices,controls) in enumerate(zip(groups,controls_by_job)):
        part=mit.estimates(values[indices],covariances[indices],controls,factors,ideal[indices],target,
            replicates=replicates,seed=int(np.random.SeedSequence([seed,k,9183]).generate_state(1)[0]),return_samples=True)
        draws.append(part.pop('_bootstrap_samples'));ideal_draws.append(part.pop('_bootstrap_ideal'))
        fractions.append(len(indices)/len(values));local.append(part)
    names=list(statistics['profiles'])
    samples={name:sum(f*d[name] for f,d in zip(fractions,draws)) for name in names}
    bootstrap_ideal=sum(f*d for f,d in zip(fractions,ideal_draws))
    mean_ideal=ideal.mean(axis=0)
    for name in names:
        point=sum(f*s['profiles'][name]['populations'] for f,s in zip(fractions,local))
        shot_se=np.sqrt(sum(f*f*s['profiles'][name]['fixed_path_delta_method_se']**2 for f,s in zip(fractions,local)))
        error=abs(point-mean_ideal);physical=abs(point-target)
        statistics['profiles'][name]=dict(populations=point,bootstrap_ci95=np.quantile(samples[name],[.025,.975],axis=0),
            fixed_path_delta_method_se=shot_se,outside_probability_range=bool(np.any((point<0)|(point>1))),
            ideal_mae=float(error.mean()),ideal_maximum=float(error.max()),physical_mae=float(physical.mean()),physical_maximum=float(physical.max()))
    for comparison in statistics['comparisons']:
        name,reference= comparison['method'],comparison['reference']
        metric=comparison['metric'];reducer=np.mean if metric=='ideal_mae' else np.max
        differences=reducer(abs(samples[name]-bootstrap_ideal),axis=1)-reducer(abs(samples[reference]-bootstrap_ideal),axis=1)
        comparison.update(difference=statistics['profiles'][name][metric]-statistics['profiles'][reference][metric],
            bootstrap_ci95=np.quantile(differences,[.025,.975]))
    statistics['calibration_scope']='job_local: each job corrected before trajectory-weighted averaging; pooled calibration and mixture fields are diagnostics only'
    statistics['job_local_controls']=[dict(job_id=job_id,trajectory_count=len(indices),weight=f,
        calibration=part['calibration'],heldout_mixture=part['heldout_mixture'],profiles=part['profiles'])
        for job_id,indices,f,part in zip(job_ids,groups,fractions,local)]
    statistics['bootstrap']['scope']+=' SPAM inversion uses independent job-local calibration draws, shared across factors inside each job.'
    return statistics


def analyze(folder,*,replicates=2000,seed=29):
    manifest=json.loads((folder/'manifest.json').read_bytes());source,integrity=read_source(folder,manifest);cfg=manifest['configuration']
    if manifest['status']!='collected':raise ValueError('need completely collected mitigation results')
    n=len(cfg['physical_path'])-2;R=cfg['trajectory_count'];L=len(cfg['fold_factors']);K=cfg['randomizations']
    paths={};controls={arm:{name:[] for name in ('all_zero','all_one','initial_mixture')} for arm in ('raw','twirled')}
    verified=[];missing=[];jobs=[];shots=0;groups=[];controls_by_job=[]
    for batch in manifest['batches']:
        local_controls={arm:{name:[] for name in ('all_zero','all_one','initial_mixture')} for arm in ('raw','twirled')}
        records=manifest['circuits'][batch['start']:batch['stop']]
        groups.append([r['trajectory'] for r in records if r['kind']=='trajectory' and r['arm']=='raw'])
        if batch['status']!='collected' or not batch.get('job_id') or not batch.get('compiled') or len(batch.get('observations',[]))!=len(records):raise ValueError('incomplete mitigation batch')
        if (folder/batch['isa_file']).exists():compiled_batch(folder,manifest,batch,source);verified.append(batch['isa_file'])
        else:missing.append(batch['isa_file'])
        jobs.append(batch['job_id'])
        for record,obs in zip(records,batch['observations']):
            counts=mit.undo_measurement_twirl(obs['outcome_counts'],n,record['measurement_mask'])
            if sum(counts.values())!=record['shots']:raise ValueError('shot mismatch')
            shots+=record['shots']
            if record['kind']=='control':
                controls[record['arm']][record['control']].append(counts)
                local_controls[record['arm']][record['control']].append(counts)
            else:
                key=(record['trajectory'],0 if record['arm']=='raw' else record['factor_index']+1)
                paths.setdefault(key,[]).append(counts)
        controls_by_job.append(local_controls)
    if shots!=cfg['total_shots'] or len(paths)!=R*(L+1):raise ValueError('missing trajectory/factor data')
    values=np.zeros((R,L+1,n));covariances=np.zeros((R,L+1,n,n))
    for (index,arm),pub_groups in paths.items():
        if len(pub_groups)!=(1 if arm==0 else K):raise ValueError('missing randomization')
        values[index,arm],covariances[index,arm]=mit.group_moments(pub_groups,n)
    base=2*n*cfg['steps'];factors=[1+2*mit.fold_count(f,base)/base for f in cfg['fold_factors']]
    ideal=np.asarray(source['final_covariances_real_gauge'])[0,:R].diagonal(axis1=-2,axis2=-1)
    target=np.asarray(source['stationary_observables']['populations'])
    statistics=mit.estimates(values,covariances,controls,factors,ideal,target,replicates=replicates,seed=seed,groups=groups)
    if cfg.get('controls_per_batch',False):
        statistics=combine_job_calibrations(statistics,values,covariances,controls_by_job,groups,factors,ideal,target,
            replicates=replicates,seed=seed,job_ids=jobs)
    return dict(experiment=EXPERIMENT,run_directory=str(folder.resolve()),configuration=cfg,job_ids=jobs,
        statistics=statistics,ideal_same_paths=ideal.mean(axis=0),physical_target=target,
        per_path_populations=values,per_path_ideal_populations=ideal,
        verification=dict(source_integrity=integrity,compiled_batches_verified=verified,compiled_batches_not_available=missing,raw_counts_reconstructed=True))


def save_analysis(report,output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.resolve().is_relative_to(Path(report['run_directory']).resolve()):raise ValueError('analysis must be outside raw run')
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'analysis.json',serializable(report))
    stats=report['statistics'];fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharey=True);x=np.arange(1,len(report['ideal_same_paths'])+1)
    rows=[]
    for name,profile in stats['profiles'].items():
        ax=axes[1 if name.endswith('_spam_corrected') else 0]
        ax.plot(x,profile['populations'],'o-',label=name.replace('_',' '))
        for j in range(len(x)):rows.append(dict(method=name,site=j+1,population=profile['populations'][j],
            ci95_low=profile['bootstrap_ci95'][0,j],ci95_high=profile['bootstrap_ci95'][1,j]))
    for ax,title in zip(axes,('No SPAM inversion','Exploratory SPAM inversion')):
        ax.plot(x,report['ideal_same_paths'],'k--',label='Ideal: same pilot branch paths');ax.plot(x,report['physical_target'],'k:',label='Physical stationary target')
        ax.set(xlabel='Site',ylabel='Population',xticks=x,title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(output/f'populations.{ext}',dpi=170)
    plt.close(fig)
    selected=sorted(set([0,max(0,(len(x)-1)//2),max(0,len(x)-2)]))
    fig,axes=plt.subplots(1,len(selected),figsize=(12,4),layout='constrained',squeeze=False)
    scale=stats['linear_zne'];factors=np.array(scale['factors']);line=np.linspace(0,max(factors),100)
    for ax,j in zip(axes.flat,selected):
        ax.errorbar(factors,scale['raw_scale_populations'][:,j],yerr=2*scale['raw_scale_fixed_path_se'][:,j],fmt='o',capsize=3,label='Twirled scales ±2 shot SE')
        slope,intercept=np.polyfit(factors,scale['raw_scale_populations'][:,j],1)
        ax.plot(line,intercept+slope*line,'--',label='Fixed linear extrapolation')
        ax.axhline(report['ideal_same_paths'][j],color='#00897b',linestyle=':',label='Same-path ideal')
        ax.set(title=f'Site {j+1}',xlabel='CZ amplification factor',ylabel='Population');ax.grid(alpha=.2);ax.legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(output/f'noise_scaling.{ext}',dpi=170)
    plt.close(fig)
    with (output/'populations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# First-order mitigation pilot','',stats['scope'],'','![Populations](populations.png)','',
        '| Method | Ideal MAE | Ideal maximum | Physical maximum |','|---|---:|---:|---:|']
    for name,p in stats['profiles'].items():lines.append(f"| {name} | {p['ideal_mae']:.6f} | {p['ideal_maximum']:.6f} | {p['physical_maximum']:.6f} |")
    lines+=['','![CZ noise scaling for three preset sites](noise_scaling.png)','',
        'This is a fixed subset pilot, not a 0.02 steady-state certification. Actual CZ factors, linear-intercept weights, '
        'held-out mixture controls, uncertainty amplification, all raw scale populations and descriptive fit residuals are in analysis.json. '
        'A favorable intercept alone does not validate noise scaling; inspect the scale dependence and uncertainty before interpreting mitigation as improvement.','',
        stats['bootstrap']['scope'],'']
    if stats.get('job_local_controls'):
        lines+=['## Preparation controls inside each evolution job','',stats['calibration_scope'],'',
            '| Job | Readout | Largest corrected-mixture deviation from 0.5 | Site |',
            '|---|---|---:|---:|']
        for part in stats['job_local_controls']:
            for arm,control in part['heldout_mixture'].items():
                deviations=abs(np.asarray(control['corrected'])-.5);j=int(np.argmax(deviations))
                lines.append(f"| {part['job_id']} | {arm} | {deviations[j]:.6f} | {j+1} |")
        lines+=['','Per-job calibration, held-out mixture intervals and evolved populations are saved in analysis.json. '
            'The mixture is held out of calibration. Same-job controls do not prove absence of within-job drift or state/context-dependent noise.','']
    (output/'Analysis.md').write_text('\n'.join(lines))


def bundle(folder,output=None):
    with zipfile.ZipFile(folder/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('source.json','manifest.json','backend_snapshot.json','hardware_results.json'):
            if (folder/name).exists():archive.write(folder/name,name)
        for path in folder.glob('resume_calibration/*/backend_snapshot.json'):
            archive.write(path,path.relative_to(folder).as_posix())
        if output:
            for p in output.iterdir():
                if p.is_file():archive.write(p,'analysis/'+p.name)


def main(options,*,backend=None,service=None,sampler_factory=None):
    if options.analyze_only:
        report=analyze(options.analyze_only,replicates=options.bootstrap,seed=options.seed)
        output=options.analysis_output_directory or DIRECTORY/'analysis/hardware'/options.analyze_only.name/'mitigation'
        save_analysis(report,output);print(output/'Analysis.md');return report
    if options.resume:
        folder=options.resume.resolve().parent;manifest=json.loads(options.resume.read_bytes());source,_=read_source(folder,manifest)
    else:
        sources,raw,metadata,input_hash=load_inputs(options.input_bundle);source=sources['candidate'];source_raw=raw['candidate']
        if options.backend!=metadata['backend']:raise ValueError('backend differs from saved input platform')
        cfg={k:getattr(options,k) for k in ('backend','physical_path','trajectory_count','shots','randomizations','control_shots','fold_factors',
            'batch_size','seed','seed_transpiler','optimization_level','controls_per_batch','calibration_policy')}
        cfg.update(source_sha256=sha(source_raw),input_bundle_sha256=input_hash,steps=len(source['schedule']['step_dts']),
            sampler_options=SAMPLER_OPTIONS,software_versions={k:importlib.metadata.version(k) for k in ('qiskit','qiskit-ibm-runtime','numpy')},
            sampling_scope='Fixed first R saved branch paths, analytically averaged ideal mixture; pilot only, no original sampling-budget certification.')
        if cfg['trajectory_count']<=source['configuration']['trajectories']:
            populations=np.asarray(source['final_covariances_real_gauge'])[0,:cfg['trajectory_count']].diagonal(axis1=-2,axis2=-1).mean(axis=0)
            cfg['pilot_ideal_populations']=populations.tolist()
            cfg['pilot_ideal_maximum_population_error']=float(max(abs(populations-np.asarray(source['stationary_observables']['populations']))))
        records,batches=plan(cfg);cfg.update(total_circuits=len(records),total_shots=sum(r['shots'] for r in records))
        manifest=dict(experiment=EXPERIMENT,configuration=cfg,circuits=records,batches=batches,status='planned');verify_manifest(manifest,source)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        folder=options.output_directory/f'{options.backend}_mitigation_{stamp}';folder.mkdir(parents=True,exist_ok=False)
        (folder/'source.json').write_bytes(source_raw);atomic_json(folder/'manifest.json',manifest)
    cfg=manifest['configuration'];print(f"Mitigation plan: {cfg['total_circuits']} circuits, {cfg['total_shots']} shots, {len(manifest['batches'])} batches. {folder}",flush=True)
    if options.plan_only:
        print('Offline plan only; no account/backend connection or submission.',flush=True);return manifest
    pending=any(b['status']!='collected' for b in manifest['batches'])
    if pending:
        # Reject ambiguous submissions before any connection or compilation.
        if any(b['status']=='submitting' and not b.get('job_id') for b in manifest['batches']):
            raise ValueError('submission has no recorded job ID; reconcile with IBM before resuming; no automatic retry')
    needs_compilation=any(b['status']!='collected' and not b.get('job_id') for b in manifest['batches'])
    if needs_compilation:
        if backend is None:
            service=service or hardware.make_service(options);backend=service.backend(cfg['backend'],use_fractional_gates=False)
        if backend.name!=cfg['backend']:raise ValueError('backend mismatch')
        refresh=getattr(backend,'refresh',None)
        if callable(refresh):refresh()
        circuits.chain_layout(backend,len(cfg['physical_path'])-2,'boundary',cfg['physical_path'])
        path=cfg['physical_path']
        if cfg.get('calibration_policy','screen')=='screen':
            nodes,edges=quality_path.calibration(backend,{7},80e-6,60e-6,.005,.02)
            if not set(path)<=nodes.keys() or any(tuple(sorted(e)) not in edges for e in zip(path,path[1:])):
                raise ValueError('chain fails current calibration filters; no new job submitted')
        else:
            print('Calibration policy: record-only; fixed reference chain retained without quality-threshold rejection.',flush=True)
        snapshot_folder=folder
        if (folder/'backend_snapshot.json').exists():
            snapshot_folder=folder/'resume_calibration'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            snapshot_folder.mkdir(parents=True,exist_ok=False)
        hardware.save_backend_snapshot(backend,path,manifest['circuits'],snapshot_folder)
        snapshot=snapshot_folder/'backend_snapshot.json'
        manifest.setdefault('calibration_snapshots',[]).append(dict(file=snapshot.relative_to(folder).as_posix(),sha256=sha(snapshot.read_bytes())))
        atomic_json(folder/'manifest.json',manifest)
    sampler=None
    for batch in manifest['batches']:
        if batch['status']=='collected':continue
        if options.compile_only:
            compiled_batch(folder,manifest,batch,source,backend);continue
        if batch.get('job_id'):
            service=service or hardware.make_service(options)
            job=service.job(batch['job_id'])
        else:
            pubs=compiled_batch(folder,manifest,batch,source,backend)
            if sampler is None:
                if sampler_factory is None:
                    from qiskit_ibm_runtime import SamplerV2
                    sampler_factory=SamplerV2
                sampler=sampler_factory(mode=backend,options=SAMPLER_OPTIONS)
            records=manifest['circuits'][batch['start']:batch['stop']]
            batch['status']='submitting';atomic_json(folder/'manifest.json',manifest)
            job=sampler.run([(pub,None,r['shots']) for pub,r in zip(pubs,records)],shots=cfg['shots'])
            job_id=job.job_id()
            if not isinstance(job_id,str) or not job_id.strip():raise ValueError('submission returned no usable job ID; reconcile with IBM before resuming')
            batch.update(job_id=job_id,status='submitted');atomic_json(folder/'manifest.json',manifest)
            print('Submitted mitigation job:',batch['job_id'],flush=True)
        result=job.result();records=manifest['circuits'][batch['start']:batch['stop']]
        if len(result)!=len(records):raise ValueError('Runtime result length mismatch')
        observations=[]
        for pub,record in zip(result,records):
            counts={str(k):int(v) for k,v in pub.data.outcome.get_counts().items()}
            if moments(counts,len(cfg['physical_path'])-2)[2]!=record['shots']:raise ValueError('Runtime shot mismatch')
            observations.append(dict(outcome_counts=counts))
        batch.update(status='collected',observations=observations,collected_at_utc=datetime.now(timezone.utc).isoformat())
        try:batch['job_metrics']=json.loads(json.dumps(job.metrics(),default=str))
        except Exception as error:batch['job_metrics_unavailable']=str(error)
        atomic_json(folder/'manifest.json',manifest)
    if options.compile_only:
        manifest['status']='compiled';atomic_json(folder/'manifest.json',manifest);bundle(folder)
        print('Compiled only; no jobs submitted. Resume this manifest to execute.',flush=True);return manifest
    manifest['status']='collected';atomic_json(folder/'manifest.json',manifest)
    report=analyze(folder,replicates=options.bootstrap,seed=options.seed);atomic_json(folder/'hardware_results.json',serializable(report))
    output=options.analysis_output_directory or DIRECTORY/'analysis/hardware'/folder.name/'mitigation'
    save_analysis(report,output);bundle(folder,output);print(f"Analysis: {output/'Analysis.md'}\nTransfer: {folder/'results_bundle.zip'}",flush=True)
    return report


if __name__=='__main__':
    try:main(parse_arguments())
    except (ValueError,OSError) as error:raise SystemExit(str(error)) from error
