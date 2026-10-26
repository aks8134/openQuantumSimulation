"""One bounded local polishing pass on the near-feasible power schedules."""
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import numpy as np
from scipy.optimize import minimize
from library import reservoir_scaling as physics
from Model3.Experiment7.schedules import Predictor,Settings
from Model3.analysis.hardware_cost_redesign.search import decode,evaluate


def main():
    path=Path(__file__).with_name('search_results.json');result=json.loads(path.read_bytes())
    physical=result['physics'];model=physics.Model(physical['n_qubits'],physical['gamma']/physical['J'],physical['p_left'],physical['p_right'],physical['exchange_prefactor'])
    original=[x for x in result['trials'] if x['family']=='power']
    # Fixed budget: two starts at each of two lengths, <=350 proxy calls each.
    result['polish_configuration']=dict(steps=[48,56],starts='best 48-step parameters and best 64-step parameters',
        method='bounded Nelder-Mead',maximum_evaluations_per_start=350,total_maximum_evaluations=1400)
    settings=Settings(.001,4.,.016,10.,64,200.)
    for count in [48,56]:
        for seed_steps in [48,64]:
            start=time.perf_counter();predictor=Predictor(model,1.,settings,order='hamiltonian-first',formula='even-odd')
            initial=next(x['parameters'] for x in original if x['steps']==seed_steps)
            def objective(parameters):
                dts=decode('power',count,parameters)
                return predictor.window_proxy(predictor.forecast(np.eye(model.n)/2,dts,'power'),float(dts[-1]))
            optimum=minimize(objective,initial,method='Nelder-Mead',bounds=[(np.log(.001),np.log(4.))]*2+[(.3,3.)],
                options=dict(maxfev=350,xatol=1e-6,fatol=1e-8))
            row=evaluate(model,1.,decode('power',count,optimum.x),10.,.016)
            row.update(family='power',parameters=optimum.x.tolist(),proxy_error=float(optimum.fun),evaluations=int(optimum.nfev),
                optimizer_converged=bool(optimum.success),wall_seconds=time.perf_counter()-start,polished=True,seed_steps=seed_steps,
                estimated_cz=20*count,estimated_resets=count,cz_reduction=1-count/result['baseline']['steps'])
            result['trials'].append(row)
            best=min([x for x in result['trials'] if x['passed']],key=lambda x:(x['estimated_cz'],x['forward_maximum']))
            result['selection']=dict(family=best['family'],steps=best['steps'],forward_maximum=best['forward_maximum'],
                trial_index=result['trials'].index(best),estimated_cz=best['estimated_cz'],cz_reduction=best['cz_reduction'],sampled_replay_validated=False)
            physics.atomic_json(path,result)
            print(f"polish {count}, start {seed_steps}: {row['forward_maximum']:.9f}, passed={row['passed']}, {optimum.nfev} evaluations",flush=True)
    return result


if __name__=='__main__':main()
