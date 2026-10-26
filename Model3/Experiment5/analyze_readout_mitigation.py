"""Exploratory marginal SPAM correction of a saved mapping comparison, offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from Model3.Experiment5 import mitigation as mit
from Model3.Experiment5.run_schedule_comparison import analyze
from Model3.Experiment5.analyze_hardware import moments,serializable

DEFAULT_RUN=DIRECTORY/'hardware_runs/ibm_phoenix_mapping_comparison_20261026T050356_310922Z'


def analyze_saved(folder,*,replicates=2000,seed=29):
    if replicates<100:raise ValueError('at least 100 bootstrap replicates required')
    report=analyze(folder);cfg=report['configuration'];manifest=json.loads((folder/'manifest.json').read_bytes())
    if cfg.get('comparison')!='mapping':raise ValueError('this entry point expects a saved paired mapping run')
    result=dict(run=folder.name,configuration=cfg,mappings={},
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()},
        scope='Exploratory marginal SPAM inversion of saved counts, not a certified readout-only correction. No target fitting, clipping, simulator or hardware execution. Stable local response assumed; preparation errors, correlated/context-dependent readout and drift are not separated.')
    for label,summary in report['schedules'].items():
        n=len(summary['hardware_populations']);groups={name:[] for name in ('all_zero','all_one','initial_mixture')};covariances={}
        for batch in manifest['batches']:
            for record,obs in zip(manifest['circuits'][batch['start']:batch['stop']],batch['observations']):
                if record['schedule']!=label:continue
                if record['kind']=='control':groups[record['control']].append(obs['outcome_counts'])
                else:covariances[record['trajectory']]=moments(obs['outcome_counts'],n)[1]
        cal={name:mit.group_moments(counts,n) for name,counts in groups.items()}
        paths=np.asarray(summary['per_path_populations']);ideal_paths=np.asarray(summary['per_path_ideal_populations'])
        mean=paths.mean(axis=0);ideal=ideal_paths.mean(axis=0);target=np.asarray(summary['physical_stationary']);R=len(paths)
        a,ca=cal['all_zero'];b,cb=cal['all_one'];corrected=mit.corrected_population(mean,a,b)
        conditional_cov=sum(covariances.values())/R**2
        corrected_cov=mit.response_covariance(mean,conditional_cov,a,ca,b,cb)
        rng=np.random.default_rng(seed);draws={name:mit.draw_control(g,n,replicates,rng) for name,g in groups.items()}
        weights=rng.multinomial(R,np.full(R,1/R),size=replicates)/R
        sampled=weights@paths;sampled_ideal=weights@ideal_paths
        sampled_corrected=mit.corrected_population(sampled,draws['all_zero'],draws['all_one'])
        comparisons=[]
        for name,func in [('mae',np.mean),('maximum',np.max)]:
            delta=func(abs(sampled_corrected-sampled_ideal),axis=1)-func(abs(sampled-sampled_ideal),axis=1)
            comparisons.append(dict(metric='ideal_'+name,
                difference=float(func(abs(corrected-ideal))-func(abs(mean-ideal))),ci95=np.quantile(delta,[.025,.975])))
        result['mappings'][label]=dict(raw=mean,corrected=corrected,ideal=ideal,physical_target=target,
            physical_sites=cfg['physical_paths'][label][1:-1],
            corrected_ci95=np.quantile(sampled_corrected,[.025,.975],axis=0),
            corrected_fixed_path_delta_method_se=np.sqrt(np.maximum(0,np.diag(corrected_cov))),
            calibration=cal,raw_controls_screen=report['control_assessment']['by_mapping'][label],
            heldout_mixture=dict(raw=cal['initial_mixture'][0],corrected=mit.corrected_population(cal['initial_mixture'][0],a,b),
                ci95=np.quantile(mit.corrected_population(draws['initial_mixture'],draws['all_zero'],draws['all_one']),[.025,.975],axis=0),expected=.5),
            raw_ideal_mae=float(np.mean(abs(mean-ideal))),corrected_ideal_mae=float(np.mean(abs(corrected-ideal))),
            raw_ideal_maximum=float(max(abs(mean-ideal))),corrected_ideal_maximum=float(max(abs(corrected-ideal))),
            raw_physical_maximum=float(max(abs(mean-target))),corrected_physical_maximum=float(max(abs(corrected-target))),
            comparisons=comparisons,outside_probability_range=bool(np.any((corrected<0)|(corrected>1))))
    result['bootstrap']=dict(replicates=replicates,seed=seed,
        scope='Paired empirical path resampling plus independent joint control-count resampling. No second resampling of evolution shots. Descriptive intervals, not simultaneous; excludes temporal correlations and SPAM-model bias.')
    assert result['input_sha256']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    return result


def save(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(serializable(report),indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,len(report['mappings']),figsize=(13,4.8),layout='constrained',squeeze=False)
    rows=[]
    for ax,(label,data) in zip(axes.flat,report['mappings'].items()):
        x=np.arange(1,len(data['raw'])+1)
        ax.plot(x,data['raw'],'o-',label='Raw hardware')
        ax.plot(x,data['corrected'],'s-',label='Exploratory SPAM correction')
        ax.plot(x,data['ideal'],'--',color='#00897b',label='Ideal: same branch paths')
        ax.plot(x,data['physical_target'],'k:',label='Physical stationary target')
        ax.set(title=label,xlabel='Logical site',ylabel='Population',xticks=x);ax.grid(alpha=.2);ax.legend(fontsize=7)
        for j in range(len(x)):rows.append(dict(mapping=label,site=j+1,physical_qubit=data['physical_sites'][j],
            raw=data['raw'][j],corrected=data['corrected'][j],ideal=data['ideal'][j],physical_target=data['physical_target'][j],
            corrected_ci95_low=data['corrected_ci95'][0,j],corrected_ci95_high=data['corrected_ci95'][1,j]))
    for ext in ('png','pdf'):fig.savefig(output/f'populations.{ext}',dpi=170)
    plt.close(fig)
    with (output/'populations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Existing-data SPAM correction (exploratory)','',report['scope'],'',
        '![Population profiles](populations.png)','',
        '| Mapping | Raw ideal MAE | Corrected ideal MAE | Raw ideal maximum | Corrected ideal maximum |',
        '|---|---:|---:|---:|---:|']
    for label,d in report['mappings'].items():lines.append(f"| {label} | {d['raw_ideal_mae']:.6f} | {d['corrected_ideal_mae']:.6f} | {d['raw_ideal_maximum']:.6f} | {d['corrected_ideal_maximum']:.6f} |")
    lines+=['','Correction is (observed-zero)/(one-zero), with shared calibration uncertainty propagated. '
        'The mixture is held out of calibration. A passing mixture check cannot validate the response for all evolved states. '
        'These all-zero/all-one controls contain preparation errors; this is not a demonstrated unbiased readout-only correction. '
        'No estimates are clipped to [0,1]. Raw values and both references are retained.','',report['bootstrap']['scope'],'']
    (output/'Analysis.md').write_text('\n'.join(lines))


def main(arguments=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=29);p.add_argument('--output-directory',type=Path)
    a=p.parse_args(arguments);output=a.output_directory or DIRECTORY/'analysis/hardware'/a.run_directory.name/'readout_spam_exploratory'
    if output.resolve().is_relative_to(a.run_directory.resolve()):raise ValueError('analysis must be outside original run')
    report=analyze_saved(a.run_directory,replicates=a.bootstrap,seed=a.seed);save(report,output)
    print(output/'Analysis.md');return report


if __name__=='__main__':main()
