"""Export self-contained 1024-path paired inputs from validated compact sources."""
from pathlib import Path
import sys
import copy
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from library import reservoir_scaling as physics
from Model3.Experiment5.run_hardware import require_validated_source


def main():
    folder=Path(__file__).parent
    templates=dict(baseline=folder/'baseline_seed29_paths128.json',candidate=folder/'reallocated_budget/candidate_seed29_paths128.json')
    model=physics.Model(10,1,.9,.1,.5);R=1024;seed=29
    streams=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(R)]
    choices=physics.draw_choice_block(streams,np.array([.45,.05,.05,.45]))
    payloads={}
    metadata=dict(schema_version=1,experiment='Model3/paired_schedule_inputs',backend='ibm_phoenix',
        physical_path=[39,49,48,58,57,56,66,76,86,85,95,105],trajectory_count=R,seed_trajectories=seed,sources={})
    for label,path in templates.items():
        original=json.loads(path.read_bytes());source=copy.deepcopy(original)
        dts=source['schedule']['step_dts'];state=np.repeat((np.eye(10)/2)[None,None],R,axis=1)
        for k,dt in enumerate(dts):state=physics.sampled_step(state,choices[:,k],physics.channel(model,dt,'rate-weighted','hamiltonian-first','even-odd'))
        assert np.allclose(state[0,:128],np.asarray(original['final_covariances_real_gauge'])[0],atol=2e-14,rtol=0.)
        assert choices[:128,:len(dts)].tolist()==original['schedule']['jump_choices']
        source['configuration']['trajectories']=R
        source['schedule']['jump_choices']=choices[:,:len(dts)].tolist()
        source['final_covariances_real_gauge']=state.tolist()
        source['provenance']['saved_trajectory_indices']=list(range(R))
        source['provenance']['prefix_extension']=dict(seed=seed,template_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            note='Same per-path SeedSequence streams and channel as validated full replay; original 128 paths independently reproduced. No new schedule selection.')
        populations=np.diagonal(state[0],axis1=-2,axis2=-1).mean(axis=0)
        population_error=float(np.max(np.abs(populations-np.diag(physics.exact_stationary(model)))))
        source['assessment']['subset_ideal_population_error']=population_error
        source['assessment']['interpretation']='Full-replay budget validation belongs to original large ensemble. This 1024-path subset is not certified to that sampling allocation; measurement/hardware noise excluded.'
        require_validated_source(source)
        raw=json.dumps(source,separators=(',',':'),allow_nan=False).encode()+b'\n'
        payloads[label+'.json']=raw
        metadata['sources'][label]=dict(file=label+'.json',sha256=hashlib.sha256(raw).hexdigest(),steps=len(dts),
            ideal_populations=populations.tolist(),ideal_population_error=population_error,
            full_replay_trajectories=source['provenance']['full_replay_trajectories'])
    output=ROOT/'Model3/Experiment5/inputs/paired_cosine72_vs48_seed29_R1024.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('input_manifest.json',json.dumps(metadata,indent=2)+'\n')
        for name,raw in payloads.items():archive.writestr(name,raw)
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    print(output,output.stat().st_size,'bytes',flush=True)
    return output


if __name__=='__main__':main()
