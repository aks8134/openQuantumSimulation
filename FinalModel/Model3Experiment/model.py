"""Inputs, deterministic paths and matched plans; no account access."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import os
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from FinalModel.Experiment1 import phoenix_resources as store,study
from FinalModel.Experiment2 import sampling
from FinalModel.Experiment3 import readout
from Model3.Experiment5 import mitigation as original

DIRECTORY=Path(__file__).resolve().parent
BASES=('z','xy','yx')
PROTOCOL='Model3/Experiment5/manuscript-mitigation; populations and normalized currents'


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    validate_config(cfg)
    return cfg


def execution_options(cfg,execution_mode=None,jobs_per_batch=None,batch_max_time=None):
    """Submission policy; CLI overrides do not change the frozen path groups."""
    h=cfg['hardware']
    values=dict(execution_mode=h.get('execution_mode','job') if execution_mode is None else execution_mode,
                jobs_per_batch=h.get('jobs_per_batch',4) if jobs_per_batch is None else jobs_per_batch,
                batch_max_time=h.get('batch_max_time',28800) if batch_max_time is None else batch_max_time)
    if values['execution_mode'] not in ('job','batch'):raise ValueError('execution_mode must be job or batch')
    for name in ('jobs_per_batch','batch_max_time'):
        if type(values[name]) is not int or values[name]<1:raise ValueError(f'{name} must be a positive integer')
    return values


def validate_config(cfg):
    p,s,m,h=cfg['physics'],cfg['sampling'],cfg['mitigation'],cfg['hardware']
    if type(p['N']) is not int or not 2<=p['N']<=100:raise ValueError('N must be an integer in [2,100]')
    if any(not np.isfinite(p[k]) or p[k]<=0 for k in ('J','gamma','prefactor')):raise ValueError('J, gamma and prefactor must be positive')
    if any(not np.isfinite(p[k]) or not 0<=p[k]<=1 for k in ('p_left','p_right')):raise ValueError('reservoir populations must be in [0,1]')
    if p['preparation'] not in ('half_mixed','vacuum','filled','center_excitation'):raise ValueError('unsupported diagonal preparation')
    K=m['randomizations']
    if type(K) is not int or K<2 or K%2:raise ValueError('randomizations must be even and >=2')
    if type(s['trajectories']) is not int or s['trajectories']<2:raise ValueError('at least two paths required')
    if any(type(v) is not int or v<2*K or v%K for v in (s['shots'],m['control_shots'])):raise ValueError('shots must divide among variants with >=2 shots each')
    if any(type(v) is not int or v<0 for v in (s['trajectory_seed'],m['seed'],h['transpiler_seed'])):raise ValueError('seeds must be nonnegative integers')
    factors=m['fold_factors']
    if len(factors)!=3 or factors!=[1.,1.25,1.5]:raise ValueError('preserve manuscript CZ factors [1,1.25,1.5]')
    if h['paths_per_job']<2 or not h['backend'].strip():raise ValueError('invalid job grouping or backend')
    execution_options(cfg)
    path=h.get('physical_path')
    if path is not None and (len(path)!=p['N']+2 or len(set(path))!=len(path) or min(path)<0):raise ValueError('physical_path must be [left bath, N sites, right bath]')
    if cfg['analysis']['bootstrap']<100:raise ValueError('bootstrap requires >=100 replicates')
    if not 0<cfg['analysis']['minimum_calibration_contrast']<1:raise ValueError('invalid calibration contrast')


def make_case(cfg):
    physics=copy.deepcopy(cfg['physics']);n=physics.pop('N');schedule=cfg['schedule'];mode=schedule['mode']
    if mode=='saved':
        payload=store.read_inputs(ROOT/schedule['input'])
        key=f"model3_N{n}_{schedule['family']}"
        case=next((c for c in payload['cases'] if c['id']==key),None)
        if case is None or case['physics']!=physics:
            raise ValueError('Saved schedule physics differ. Use constant, warped_cosine or manual mode for changed parameters.')
        return copy.deepcopy(case)
    if mode=='constant':dts=[float(schedule['J_dt'])/physics['J']]*int(schedule['steps'])
    elif mode=='warped_cosine':
        M=int(schedule['steps']);lo,hi,q=(float(schedule[k]) for k in ('J_tau_min','J_tau_max','q'))
        if M<2 or not 0<lo<=hi or q<=0:raise ValueError('invalid warped-cosine parameters')
        dts=((lo+(hi-lo)*(1+np.cos(np.pi*(np.arange(M)/(M-1))**q))/2)/physics['J']).tolist()
    elif mode=='manual':dts=[float(v) for v in schedule['step_dts']]
    else:raise ValueError('schedule mode must be saved, constant, warped_cosine or manual')
    if not dts or any(not np.isfinite(v) or v<=0 for v in dts):raise ValueError('timesteps must be positive and finite')
    return dict(id=f'model3_N{n}_{mode}',model='model3',N=n,physics=physics,steps=len(dts),
                step_dts=dts,schedule_sha256=store.digest(dts),physical_time=sum(dts),family=mode)


def reference_engine(cfg):
    engine=cfg['sampling'].get('reference_engine','auto')
    if engine not in ('auto','cpp','portable'):raise ValueError('reference_engine must be auto, cpp or portable')
    if engine!='auto':return engine
    cpuinfo=Path('/proc/cpuinfo')
    supported=(os.name!='nt' and Path('/usr/bin/g++').is_file() and cpuinfo.is_file()
               and 'avx512f' in cpuinfo.read_text()
               and (ROOT/'FinalModel/Experiment3/readout.cpp').is_file())
    return 'cpp' if supported else 'portable'


def references(cfg,case):
    n=case['N'];p=case['physics'];dts=np.ascontiguousarray(store.steps(case),dtype=float)
    initial=np.ascontiguousarray(study.initial(n,p['preparation']))
    if reference_engine(cfg)=='portable':
        from FinalModel.Model3Experiment import portable
        seed=int(sampling.physical_identity(case)[:16],16)^cfg['sampling']['trajectory_seed']
        return portable.references(case,cfg['sampling']['trajectories'],seed,initial)
    digital=np.empty((1,2*n-1))
    sampling.kernel().trajectory_reference(n,p['prefactor'],p['J'],p['gamma'],p['p_left'],p['p_right'],initial,dts,len(dts),0,digital)
    targetC=study.stationary(study.model_for(n,p))
    target=np.r_[np.diag(targetC),4*p['prefactor']*np.diag(targetC,1)]
    R=cfg['sampling']['trajectories'];choices=np.empty((R,len(dts)),dtype=np.int32);ideals=np.empty((R,2*n-1))
    C=np.empty((n,n));seed=int(sampling.physical_identity(case)[:16],16)^cfg['sampling']['trajectory_seed']
    lib=readout.kernel()
    for index in range(R):
        lib.trajectory_snapshot(n,p['prefactor'],p['J'],p['gamma'],p['p_left'],p['p_right'],
            initial,dts,len(dts),0,seed,index,0,C,choices[index])
        ideals[index]=np.r_[np.diag(C),4*p['prefactor']*np.diag(C,1)]
    return dict(choices=choices,ideal_paths=ideals,digital=digital[0],target=target)


def batches(cfg):
    R=cfg['sampling']['trajectories'];size=cfg['hardware']['paths_per_job']
    groups=np.array_split(np.random.default_rng(cfg['mitigation']['seed']).permutation(R),(R+size-1)//size)
    if min(map(len,groups))<2:raise ValueError('each job needs >=2 paths for stratified bootstrap')
    return [dict(index=i,indices=g.tolist(),status='not_submitted',job_id=None) for i,g in enumerate(groups)]


def records(cfg,batch):
    K=cfg['mitigation']['randomizations'];S=cfg['sampling']['shots'];CS=cfg['mitigation']['control_shots']
    n=cfg['physics']['N'];seed=cfg['mitigation']['seed'];masks=original.balanced_masks(n,K,seed)
    rng=np.random.default_rng(np.random.SeedSequence([seed,batch['index']]))
    result=[]
    def controls(name):
        group=[dict(kind='control',control=name,basis='z',arm='raw',variant=None,mask=0,shots=CS)]
        group.extend(dict(kind='control',control=name,basis='z',arm='twirled',variant=v,mask=masks[v],shots=CS//K) for v in range(K))
        result.extend(group[j] for j in rng.permutation(len(group)))
    controls('all_zero')
    for pos,index in enumerate(batch['indices']):
        if pos==len(batch['indices'])//2:controls('all_one')
        group=[]
        for basis in BASES:
            group.append(dict(kind='trajectory',trajectory=index,basis=basis,arm='raw',factor_index=0,variant=None,mask=0,shots=S))
            for factor in range(3):
                group.extend(dict(kind='trajectory',trajectory=index,basis=basis,arm='twirled',factor_index=factor,
                                  variant=v,mask=masks[v],shots=S//K) for v in range(K))
        result.extend(group[j] for j in rng.permutation(len(group)))
    controls('initial_mixture')
    return [dict(row,pub_index=i,case_id=store.digest(row)[:20]) for i,row in enumerate(result)]


def workload(cfg):
    groups=batches(cfg);R=cfg['sampling']['trajectories'];K=cfg['mitigation']['randomizations']
    result=dict(paths=R,shots_per_path_basis_factor=cfg['sampling']['shots'],bases=list(BASES),jobs=len(groups),
        evolution_circuits=R*3*(1+3*K),control_circuits=len(groups)*3*(1+K),
        evolution_shots=R*3*4*cfg['sampling']['shots'],control_shots=len(groups)*6*cfg['mitigation']['control_shots'])
    result['total_circuits']=result['evolution_circuits']+result['control_circuits']
    result['total_shots']=result['evolution_shots']+result['control_shots']
    if result['total_shots']>cfg['hardware']['max_total_shots']:raise ValueError('declared total shot ceiling is exceeded')
    return result


def prepare(config_path,output=None):
    from datetime import datetime,timezone
    cfg=load_config(config_path);case=make_case(cfg);work=workload(cfg)
    folder=Path(output) if output else DIRECTORY/'hardware_runs'/datetime.now(timezone.utc).strftime('model3_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    data=references(cfg,case);np.savez_compressed(folder/'paths.npz',**data)
    source=store.sealed(dict(configuration=cfg,case=case,protocol=PROTOCOL,reference_engine=reference_engine(cfg),
        paths_sha256=store.file_digest(folder/'paths.npz'),versions=store.versions(),
        code={str(path.relative_to(ROOT)):store.file_digest(path) for path in
              (Path(__file__),DIRECTORY/'circuits.py',DIRECTORY/'run.py',DIRECTORY/'portable.py',Path(original.__file__),Path(readout.__file__))},
        maximum_mean_output_error=float(abs(data['digital']-data['target']).max()),
        maximum_selected_path_output_error=float(abs(data['ideal_paths'].mean(axis=0)-data['target']).max())))
    store.atomic_json(folder/'source.json',source)
    manifest=dict(experiment='FinalModel/Model3Experiment/v1',source_sha256=source['sha256'],
                  workload=work,status='planned',batches=batches(cfg))
    store.atomic_json(folder/'manifest.json',manifest)
    print(json.dumps(dict(run=str(folder),**work),indent=2),flush=True)
    return folder,manifest


def load_run(folder):
    folder=Path(folder);source=store.read_sealed(folder/'source.json')
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['experiment']!='FinalModel/Model3Experiment/v1' or manifest['source_sha256']!=source['sha256']:
        raise ValueError('run/source identity mismatch')
    if store.file_digest(folder/'paths.npz')!=source['paths_sha256']:raise ValueError('saved paths changed')
    expected=batches(source['configuration'])
    if len(expected)!=len(manifest['batches']) or manifest['workload']!=workload(source['configuration']):raise ValueError('workload changed')
    for want,got in zip(expected,manifest['batches']):
        if got['indices']!=want['indices'] or got['index']!=want['index']:raise ValueError('batch membership changed')
        if got['status'] not in ('not_submitted','submitting','submitted','collected'):raise ValueError('invalid batch state')
        if got['status'] in ('submitted','collected') and not got.get('job_id'):raise ValueError('missing job ID')
    with np.load(folder/'paths.npz',allow_pickle=False) as arrays:data={k:arrays[k].copy() for k in arrays.files}
    return source,manifest,data
