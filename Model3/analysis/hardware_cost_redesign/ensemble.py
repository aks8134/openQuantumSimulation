"""Fixed finite-ensemble population/current checks; no hardware execution."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from library import reservoir_scaling as physics


def main():
    folder=Path(__file__).parent;study=json.loads((folder/'search_results.json').read_bytes())
    candidate=min([x for x in study['trials'] if x['steps']==48],key=lambda x:x['forward_maximum'])
    model=physics.Model(10,1,.9,.1,.5);target=physics.exact_stationary(model);rows=[]
    for seed in [29,30,31]:
        streams=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(4096)]
        choices=physics.draw_choice_block(streams,np.array([.45,.05,.05,.45]))
        for label,selected in [('baseline',study['baseline']),('candidate48',candidate)]:
            C=np.repeat((np.eye(10)/2)[None,None],4096,axis=1)
            for k,dt in enumerate(selected['step_dts']):
                C=physics.sampled_step(C,choices[:,k],physics.channel(model,dt,'rate-weighted','hamiltonian-first','even-odd'))
            for R in [128,512,1024,4096]:
                values=C[0,:R];mean=values.mean(axis=0);pop=np.diag(mean);current=2*np.diag(mean,1)
                se_pop=np.std(np.diagonal(values,axis1=-2,axis2=-1),axis=0,ddof=1)/np.sqrt(R)
                se_current=np.std(2*np.diagonal(values,offset=1,axis1=-2,axis2=-1),axis=0,ddof=1)/np.sqrt(R)
                rows.append(dict(method=label,seed=seed,trajectories=R,populations=pop.tolist(),currents=current.tolist(),
                    population_error=float(np.max(np.abs(pop-np.diag(target)))),
                    current_error=float(np.max(np.abs(current-2*np.diag(target,1)))),
                    max_population_two_se=float(2*se_pop.max()),max_current_two_se=float(2*se_current.max())))
    payload=dict(scope='Existing chosen schedules, no schedule/seed selection on these ensembles. Covariance-path means exclude preparation/measurement shots and hardware noise.',rows=rows)
    (folder/'ensemble_size_analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload


if __name__=='__main__':main()
