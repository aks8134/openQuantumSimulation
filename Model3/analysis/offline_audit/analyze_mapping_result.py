"""Analyze the pulled paired mapping result without any IBM connection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import t

from Model3.Experiment5.run_schedule_comparison import analyze
from Model3.Experiment5.analyze_hardware import serializable, source_integrity

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'Model3/Experiment5/hardware_runs/ibm_phoenix_mapping_comparison_20261026T050356_310922Z'
OUT=ROOT/'Model3/Experiment5/analysis/hardware'/RUN.name/'mapping_comparison'


def csv_table(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()}
    report=analyze(RUN);manifest=json.loads((RUN/'manifest.json').read_bytes());cfg=report['configuration']
    assert cfg['comparison']=='mapping' and cfg['trajectory_count']==1024 and cfg['shots']==32
    assert (RUN/'baseline.json').read_bytes()==(RUN/'candidate.json').read_bytes()
    cached=json.loads((RUN/'hardware_results.json').read_bytes())
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(RUN/'results_bundle.zip') as archive:
        assert archive.testzip() is None
        for name in ('manifest.json','baseline.json','candidate.json','input_manifest.json','backend_snapshot.json','hardware_results.json'):
            assert json.loads(archive.read(name))==json.loads((RUN/name).read_bytes())
    paths={label:np.asarray(report['schedules'][label]['per_path_populations']) for label in ('baseline','candidate')}
    ideals={label:np.asarray(report['schedules'][label]['per_path_ideal_populations']) for label in paths}
    np.testing.assert_array_equal(ideals['baseline'],ideals['candidate'])
    total=0;groups=[]
    for batch in manifest['batches']:
        records=manifest['circuits'][batch['start']:batch['stop']];indices=[]
        for record,obs in zip(records,batch['observations'],strict=True):
            counts=obs['outcome_counts'];shots=sum(counts.values());total+=shots
            assert shots==record['shots']
            if record['kind']=='trajectory':
                vector=np.array([sum(count*((int(key,2)>>j)&1) for key,count in counts.items())/shots for j in range(10)])
                np.testing.assert_array_equal(vector,paths[record['schedule']][record['trajectory']])
                if record['schedule']=='baseline':indices.append(record['trajectory'])
            path=cfg['physical_paths'][record['schedule']]
            assert record['final_logical_to_physical']==path[1:-1]+[path[0],path[-1]]
        groups.append(np.asarray(indices))
    assert total==77824
    for label in paths:
        for key in ('hardware_populations','ideal_same_paths','across_path_se','fixed_paths_shot_se'):
            np.testing.assert_allclose(report['schedules'][label][key],cached['schedules'][label][key],atol=1e-15,rtol=0)
        assert report['schedules'][label]['errors_vs_own_ideal']==cached['schedules'][label]['errors_vs_own_ideal']
    assert report['control_assessment']==cached['control_assessment']
    previous=cfg['previous_qpy_verification']
    integrity=source_integrity((RUN.parent/previous['run']/'manifest.json').read_bytes(),previous['manifest_sha256'])
    h={label:values.mean(axis=0) for label,values in paths.items()};ideal=ideals['baseline'].mean(axis=0)
    target=np.asarray(report['schedules']['baseline']['physical_stationary'])
    n=1024;difference=paths['candidate']-paths['baseline'];delta=difference.mean(axis=0)
    se=difference.std(axis=0,ddof=1)/np.sqrt(n);raw=2*t.sf(abs(delta/se),n-1)
    order=np.argsort(raw);adjusted=np.empty(10)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(raw[order]*(10-np.arange(10))))
    site_rows=[]
    for j in range(10):
        site_rows.append(dict(site=j+1,current_physical_qubit=cfg['physical_paths']['baseline'][j+1],
            alternative_physical_qubit=cfg['physical_paths']['candidate'][j+1],physical_target=target[j],
            ideal=ideal[j],current_hardware=h['baseline'][j],alternative_hardware=h['candidate'][j],
            current_residual=h['baseline'][j]-ideal[j],alternative_residual=h['candidate'][j]-ideal[j],
            alternative_minus_current=delta[j],paired_se=se[j],holm_p_ten_sites=adjusted[j]))
    physical_rows=[]
    current_qubits=cfg['physical_paths']['baseline'][1:-1];alternative_qubits=cfg['physical_paths']['candidate'][1:-1]
    for q in current_qubits:
        if q not in alternative_qubits:continue
        a,b=current_qubits.index(q),alternative_qubits.index(q)
        physical_rows.append(dict(physical_qubit=q,current_site=a+1,alternative_site=b+1,
            current_hardware=h['baseline'][a],alternative_hardware=h['candidate'][b],
            current_ideal=ideal[a],alternative_ideal=ideal[b],
            current_residual=h['baseline'][a]-ideal[a],alternative_residual=h['candidate'][b]-ideal[b]))
    def metrics(a,b,reference):
        ea=abs(a-reference);eb=abs(b-reference)
        return np.stack([eb.mean(axis=-1)-ea.mean(axis=-1),
            eb.max(axis=-1)-ea.max(axis=-1),
            np.sqrt((eb**2).mean(axis=-1))-np.sqrt((ea**2).mean(axis=-1))],axis=-1)
    observed=np.r_[metrics(h['baseline'],h['candidate'],ideal),metrics(h['baseline'],h['candidate'],target)]
    rng=np.random.default_rng(20261026);replicates=10000;boot=[]
    joined=np.concatenate([paths['baseline'],paths['candidate'],ideals['baseline']],axis=1)
    for _ in range(replicates//250):
        means=np.zeros((250,30))
        for indices in groups:
            size=len(indices);weights=rng.multinomial(size,np.full(size,1/size),size=250)
            means+=weights@joined[indices]/n
        a,b,ref=means[:,:10],means[:,10:20],means[:,20:]
        boot.append(np.column_stack([metrics(a,b,ref),metrics(a,b,target)]))
    intervals=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    names=['ideal_mae','ideal_maximum','ideal_rmse','physical_mae','physical_maximum','physical_rmse']
    aggregates=[dict(metric=name,alternative_minus_current=observed[j],bootstrap_95_low=intervals[0,j],
        bootstrap_95_high=intervals[1,j]) for j,name in enumerate(names)]
    details=dict(run=RUN.name,configuration=cfg,source_files_identical=True,total_circuits=len(manifest['circuits']),total_shots=total,
        raw_counts_independently_reconstructed=True,bundle_crc_and_json_verified=True,
        original_files_sha256=originals,previous_manifest_integrity=integrity,
        prior_qpy_verification_record=previous,current_cached_qpy_verification=cached['verification'],
        current_local_verification=report['verification'],controls=report['control_assessment'],
        schedules={label:{k:v for k,v in values.items() if not k.startswith('per_path_')} for label,values in report['schedules'].items()},
        paired_site_tests=site_rows,shared_physical_qubit_comparisons=physical_rows,aggregate_comparisons=aggregates,
        bootstrap=dict(seed=20261026,replicates=replicates,unit='paired trajectory, stratified by observed batch',group_sizes=[len(g) for g in groups],
            scope='Descriptive percentile intervals, not simultaneous guarantees; condition on these jobs, omit drift/context/shared calibration uncertainty. Nonlinear max/absolute statistics can be nonregular.'),
        scope='Populations only; no readout mitigation, hardware queries, simulator evolution or new job.')
    csv_table('paired_mapping_sites.csv',site_rows);csv_table('shared_physical_qubits.csv',physical_rows);csv_table('aggregate_comparisons.csv',aggregates)
    (OUT/'findings.json').write_text(json.dumps(serializable(details),indent=2,allow_nan=False)+'\n')
    plots(h,ideal,target,report,physical_rows)
    assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()}
    print(json.dumps(serializable(dict(aggregate_comparisons=aggregates,holm_site_pvalues=adjusted,previous_manifest_integrity=integrity)),indent=2),flush=True)


def plots(h,ideal,target,report,physical_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained',sharey=True)
    x=np.arange(1,11)
    for ax,label,title in zip(axes,('baseline','candidate'),('Current chain','Alternative chain')):
        ax.errorbar(x,h[label],yerr=2*np.array(report['schedules'][label]['across_path_se']),fmt='o-',capsize=3,
            color='#b71c1c' if label=='baseline' else '#1565c0',label='Hardware ±2 across-path SE')
        ax.plot(x,ideal,'s--',color='#00897b',label='Common ideal: same 1024 paths')
        ax.plot(x,target,'k:',label='Physical stationary target')
        ax.set(xlabel='Logical site',ylabel='Population',xticks=x,title=title+' — 48 steps, 32 shots/path')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'paired_mapping_populations.{ext}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');x=np.arange(len(physical_rows))
    ax.plot(x,[r['current_hardware'] for r in physical_rows],'o-',color='#b71c1c',label='Current chain hardware')
    ax.plot(x,[r['alternative_hardware'] for r in physical_rows],'o-',color='#1565c0',label='Alternative chain hardware')
    ax.plot(x,[r['current_ideal'] for r in physical_rows],'s--',color='#b71c1c',alpha=.7,label='Current chain ideal')
    ax.plot(x,[r['alternative_ideal'] for r in physical_rows],'s--',color='#1565c0',alpha=.7,label='Alternative chain ideal')
    ax.set(xticks=x,xticklabels=[f"q{r['physical_qubit']}\nsite {r['current_site']} → {r['alternative_site']}" for r in physical_rows],
        ylabel='Population',xlabel='Shared physical qubit and logical-site change',
        title='Shared physical qubits: populations must be compared with each role’s own ideal')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncols=2)
    for ext in ('png','pdf'):fig.savefig(OUT/f'shared_physical_populations.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
