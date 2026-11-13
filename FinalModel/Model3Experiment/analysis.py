"""Paired population/current analysis with the manuscript's local SPAM model.

Zero/one Z controls calibrate the same marginal affine response as before.
For currents the pair parity is corrected under the explicitly assumed
product of those local responses. No extra calibration states are introduced.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from FinalModel.Model3Experiment import model
from Model3.Experiment5 import mitigation as original

METHODS=('raw','twirled','zne')


def features(counts,n):
    if not counts or any(type(c) is not int or c<0 for c in counts.values()):raise ValueError('invalid counts')
    labels=list(counts);shots=sum(counts.values())
    if shots<2:raise ValueError('at least two shots required')
    if any(len(k)!=n or set(k)-{'0','1'} for k in labels):raise ValueError('incorrect outcome-register bit width')
    bits=np.array([[(int(k,2)>>j)&1 for j in range(n)] for k in labels])
    z=1-2*bits;values=np.c_[z,z[:,:-1]*z[:,1:]]
    weights=np.array([counts[k] for k in labels])/shots
    mean=weights@values
    variance=np.maximum(weights@(values*values)-mean*mean,0)/(shots-1)
    return mean,variance


def correct_features(values,zero,one,minimum=.5):
    """Correct singles and neighbouring parities using local affine responses.

    E[z_obs]=c+d E[z_true]. For pairs use
    E[(z_obs,i-c_i)(z_obs,j-c_j)]/(d_i*d_j), retaining singles offsets.
    """
    values=np.asarray(values);zero=np.asarray(zero);one=np.asarray(one);n=zero.shape[-1]
    d=one-zero;c=1-zero-one
    if np.any(~np.isfinite(d)) or np.any(d<minimum):raise ValueError('insufficient calibration contrast')
    singles=values[...,:n];pairs=values[...,n:]
    corrected_s=(singles-c)/d
    corrected_p=(pairs-c[...,:-1]*singles[...,1:]-c[...,1:]*singles[...,:-1]+c[...,:-1]*c[...,1:])/(d[...,:-1]*d[...,1:])
    return np.concatenate([corrected_s,corrected_p],axis=-1)


def observables(values,prefactor,n):
    pop=(1-values[...,0,:n])/2
    current=prefactor*np.where(np.arange(n-1)%2==0,1.,-1.)*(values[...,2,n:]-values[...,1,n:])
    return np.concatenate([pop,current],axis=-1)


def batch_features(cfg,records,observations,indices):
    n=cfg['physics']['N'];D=2*n-1;F=3
    groups={};controls={a:{c:[] for c in ('all_zero','all_one','initial_mixture')} for a in ('raw','twirled')}
    if len(records)!=len(observations):raise ValueError('incomplete batch results')
    factors={}
    for row,obs in zip(records,observations):
        if sum(obs['outcome_counts'].values())!=row['shots']:raise ValueError('unexpected shot count')
        counts=original.undo_measurement_twirl(obs['outcome_counts'],n,row['mask'])
        if row['kind']=='control':controls[row['arm']][row['control']].append(counts);continue
        arm=0 if row['arm']=='raw' else 1+row['factor_index']
        key=(row['trajectory'],arm,model.BASES.index(row['basis']))
        groups.setdefault(key,[]).append(counts)
        if row['arm']=='twirled':
            old=factors.setdefault((row['trajectory'],row['factor_index']),row['actual_factor'])
            if abs(old-row['actual_factor'])>1e-12:raise ValueError('noise factors differ across basis/variants')
    values=np.empty((len(indices),4,3,D));variance=np.empty_like(values);result=np.empty((len(indices),3,3,D))
    for i,index in enumerate(indices):
        for arm in range(4):
            for basis in range(3):
                samples=groups[(index,arm,basis)]
                expected=1 if arm==0 else cfg['mitigation']['randomizations']
                if len(samples)!=expected:raise ValueError('missing or duplicate matched variants')
                moments=[features(c,n) for c in samples]
                values[i,arm,basis]=np.mean([m[0] for m in moments],axis=0)
                variance[i,arm,basis]=sum(m[1] for m in moments)/len(samples)**2
        w=original.zne_weights([factors[index,k] for k in range(F)])
        result[i,0]=values[i,0];result[i,1]=values[i,1];result[i,2]=np.einsum('k,kbj->bj',w,values[i,1:])
    for arm,ctrl in controls.items():
        for name,counts in ctrl.items():
            if len(counts)!=(1 if arm=='raw' else cfg['mitigation']['randomizations']):raise ValueError('missing preparation controls')
    return result,values,controls


def bootstrap_weights(size,count,rng):
    return rng.multinomial(size,np.full(size,1/size),size=count)/size


def analyze(folder):
    folder=Path(folder);source,manifest,data=model.load_run(folder);cfg=source['configuration'];n=source['case']['N'];D=2*n-1
    if any(b['status']!='collected' for b in manifest['batches']):raise ValueError('all batches must be collected before final analysis')
    B=cfg['analysis']['bootstrap'];rng=np.random.default_rng(cfg['analysis']['seed']);R=cfg['sampling']['trajectories']
    points={m:np.zeros(D) for m in METHODS};samples={m:np.zeros((B,D)) for m in METHODS}
    points.update({m+'_spam_corrected':np.zeros(D) for m in METHODS})
    samples.update({m+'_spam_corrected':np.zeros((B,D)) for m in METHODS})
    conditional_paths={m:[] for m in METHODS};ideal_boot=np.zeros((B,D));diagnostics=[];receipts=[]
    corrected_valid=True;minimum=cfg['analysis']['minimum_calibration_contrast'];scales=np.zeros((3,D))
    for batch in manifest['batches']:
        path=folder/batch['result_file'];record=model.store.read_sealed(path)
        if record['source_sha256']!=source['sha256'] or record['sha256']!=batch['result_sha256']:raise ValueError('batch result changed')
        compiled=model.store.read_sealed(folder/batch['records_file'])
        if compiled['sha256']!=record['compiled_sha256']:raise ValueError('compiled/result identity mismatch')
        idx=batch['indices'];weight=len(idx)/R
        f,raw_f,controls=batch_features(cfg,compiled['circuits'],record['observations'],idx)
        W=bootstrap_weights(len(idx),B,rng)
        boot=(W@f.reshape(len(idx),-1)).reshape(B,3,3,D)
        mean=f.mean(axis=0);ideal_boot+=weight*(W@data['ideal_paths'][idx])
        diag={'batch':batch['index'],'job_id':batch['job_id'],'calibration':{},'heldout_mixture':{}}
        cal={};draws={}
        for arm in ('raw','twirled'):
            cal[arm]={};draws[arm]={}
            for name in ('all_zero','all_one','initial_mixture'):
                cal[arm][name]=original.group_moments(controls[arm][name],n)[0]
                draws[arm][name]=original.draw_control(controls[arm][name],n,B,rng)
            diag['calibration'][arm]={'zero':cal[arm]['all_zero'].tolist(),'one':cal[arm]['all_one'].tolist()}
            mixed=np.mean([features(c,n)[0] for c in controls[arm]['initial_mixture']],axis=0)
            try:
                corrected=correct_features(mixed,cal[arm]['all_zero'],cal[arm]['all_one'],minimum)
                diag['heldout_mixture'][arm]={'populations':((1-corrected[:n])/2).tolist(),
                    'Z_pair_parities':corrected[n:].tolist(),'expected_population':.5,'expected_pair_parity':0.}
            except ValueError as exc:diag['heldout_mixture'][arm]={'unavailable':str(exc)}
        for k,m in enumerate(METHODS):
            v=observables(f[:,k],cfg['physics']['prefactor'],n)
            conditional_paths[m].append(v);points[m]+=weight*v.mean(axis=0)
            samples[m]+=weight*observables(boot[:,k],cfg['physics']['prefactor'],n)
            arm='raw' if m=='raw' else 'twirled'
            try:
                corrected=correct_features(mean[k],cal[arm]['all_zero'],cal[arm]['all_one'],minimum)
                corrected_boot=correct_features(boot[:,k],draws[arm]['all_zero'][:,None,:],draws[arm]['all_one'][:,None,:],minimum)
                points[m+'_spam_corrected']+=weight*observables(corrected,cfg['physics']['prefactor'],n)
                samples[m+'_spam_corrected']+=weight*observables(corrected_boot,cfg['physics']['prefactor'],n)
            except ValueError as exc:
                corrected_valid=False;diag.setdefault('correction_unavailable',[]).append(str(exc))
        for scale in range(3):scales[scale]+=weight*observables(raw_f[:,scale+1].mean(axis=0),cfg['physics']['prefactor'],n)
        diagnostics.append(diag);receipts.append({'file':batch['result_file'],'sha256':record['sha256']})
    if not corrected_valid:
        for m in METHODS:del points[m+'_spam_corrected'];del samples[m+'_spam_corrected']
    ideal=data['ideal_paths'].mean(axis=0);target=data['target'];digital=data['digital'];profiles={}
    for name,value in points.items():
        profiles[name]={'populations':value[:n].tolist(),'currents_over_J':value[n:].tolist(),
            'bootstrap_ci95':np.quantile(samples[name],[.025,.975],axis=0).tolist(),
            'same_path_ideal_MAE':float(abs(value-ideal).mean()),'same_path_ideal_maximum':float(abs(value-ideal).max()),
            'digital_mean_maximum':float(abs(value-digital).max()),'stationary_maximum':float(abs(value-target).max()),
            'population_stationary_maximum':float(abs(value[:n]-target[:n]).max()),
            'current_stationary_maximum':float(abs(value[n:]-target[n:]).max())}
    contrasts=[]
    for method,reference in (('twirled','raw'),('zne','twirled'),('zne','raw'),('zne_spam_corrected','zne')):
        if method not in samples:continue
        delta=np.mean(abs(samples[method]-ideal_boot),axis=1)-np.mean(abs(samples[reference]-ideal_boot),axis=1)
        contrasts.append(dict(method=method,reference=reference,metric='same_path_ideal_MAE',
                             difference=profiles[method]['same_path_ideal_MAE']-profiles[reference]['same_path_ideal_MAE'],
                             bootstrap_ci95=np.quantile(delta,[.025,.975]).tolist()))
    report=model.store.sealed(dict(protocol=model.PROTOCOL,source_sha256=source['sha256'],configuration=cfg,
        workload=manifest['workload'],profiles=profiles,paired_contrasts=contrasts,job_diagnostics=diagnostics,
        exact_digital_mean=digital.tolist(),same_path_ideal=ideal.tolist(),stationary_target=target.tolist(),
        scale_profiles=scales.tolist(),default_zne_weights=original.zne_weights([1,1.25,1.5]).tolist(),
        batch_receipts=receipts,bootstrap={'replicates':B,'seed':cfg['analysis']['seed'],
        'scope':'Paired path bootstrap stratified by observed job, with joint control-count resampling shared across noise factors. No second resampling of evolution shots. Pointwise descriptive intervals, not simultaneous; drift and calibration-model bias excluded.'},
        current_readout_correction='Same marginal zero/one affine SPAM model extended to pairs assuming local product response. Not a correlated-readout inverse; basis-rotation and evolution errors are not removed. No clipping or target fitting.'))
    output=folder/'analysis';output.mkdir(exist_ok=True);model.store.atomic_json(output/'analysis.json',report)
    write_report(output,report)
    return report


def write_report(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=report['configuration'];n=cfg['physics']['N'];ideal=np.array(report['same_path_ideal']);target=np.array(report['stationary_target'])
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    methods=[m for m in ('raw','twirled','zne','zne_spam_corrected') if m in report['profiles']]
    for col,(offset,count,label) in enumerate(((0,n,'Population'),(n,n-1,'Current / J'))):
        ax=axes[col];x=np.arange(1,count+1)
        ax.plot(x,ideal[offset:offset+count],':',color='0.25',label='Same-path ideal')
        ax.plot(x,target[offset:offset+count],'-',color='black',lw=1,label='Stationary target')
        for name in methods:
            profile=report['profiles'][name];v=np.array(profile['populations']+profile['currents_over_J'])[offset:offset+count]
            ci=np.array(profile['bootstrap_ci95'])[:,offset:offset+count]
            # Quantile bands need not bracket a biased plug-in point.
            ax.plot(x,v,'o-',ms=3,label=name)
            ax.fill_between(x,ci[0],ci[1],alpha=.08)
        ax.set(xlabel='Site' if col==0 else 'Bond',ylabel=label)
        if col==0:ax.set_ylim(0,1)
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'observables.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    lines=['# Model 3 hardware populations and currents','',f"R={cfg['sampling']['trajectories']}; S={cfg['sampling']['shots']} per basis and arm/factor.",'',
           '| Method | Same-path ideal MAE | Maximum population target error | Maximum current target error |',
           '|---|---:|---:|---:|']
    for name,p in report['profiles'].items():lines.append(f"| {name} | {p['same_path_ideal_MAE']:.6g} | {p['population_stationary_maximum']:.6g} | {p['current_stationary_maximum']:.6g} |")
    lines+=['','![Populations and currents](observables.png)','',report['bootstrap']['scope'],'',report['current_readout_correction']]
    (output/'Results.md').write_text('\n'.join(lines)+'\n')
