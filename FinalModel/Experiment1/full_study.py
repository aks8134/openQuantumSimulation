"""Run the complete offline size sweep with audited acceleration and streaming compilation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from FinalModel.Experiment1 import study,stream_compile
from FinalModel.Experiment1.accelerated import Evaluator,kernel
from library import reservoir_scaling as physics


def independent_validation(n,spec,config,dts):
    model=study.model_for(n,spec)
    state=study.initial(n,spec["preparation"])
    target=study.stationary(model)
    last_dt,branch=None,None
    def advance(C,dt):
        nonlocal last_dt,branch
        if dt!=last_dt:
            branch=physics.channel(model,spec["J"]*dt,"rate-weighted","hamiltonian-first","even-odd")
            last_dt=dt
        return physics.mean_step(C,branch)
    for dt in dts:state=advance(state,float(dt))
    output=state.copy()
    errors=[study.error(state,target,model)]
    tail=int(np.ceil(config["forward_window"]/dts[-1]))
    if tail>config["validation_step_cap"]:raise ValueError("validation cap")
    for _ in range(tail):
        state=advance(state,float(dts[-1]))
        errors.append(study.error(state,target,model))
    eigenvalues=np.linalg.eigvalsh(output)
    return dict(passed=bool(max(errors)<=config["mean_tolerance"] and eigenvalues[0]>=-1e-9 and eigenvalues[-1]<=1+1e-9),
        engine="existing independent dense affine channel",output_error=errors[0],maximum_forward_error=max(errors),
        forward_steps=tail,forward_duration=float(tail*dts[-1]),forward_errors=errors,
        minimum_covariance_eigenvalue=float(eigenvalues[0]),maximum_covariance_eigenvalue=float(eigenvalues[-1]),
        output_covariance=output.tolist(),output_total_excitation=float(np.trace(output)))


def schedules_case(n,spec,config,progress=None):
    started=time.perf_counter()
    selection_config=dict(config,mean_tolerance=.995*config["mean_tolerance"])
    evaluator=Evaluator(n,spec,selection_config)
    fixed_count=evaluator.constant(config["fixed_dt"],config["fixed_step_budget"])
    fixed=None if fixed_count<1 else np.full(fixed_count,config["fixed_dt"])
    if progress:progress(dict(event="fixed",steps=fixed_count))
    low,high=config["minimum_dt"],config["maximum_dt"]
    grid=np.unique(np.r_[np.geomspace(low,high,12),np.linspace(max(low,.05),high,20),
        [v for v in (config["fixed_dt"],np.pi/(4*spec["prefactor"]*spec["J"]),np.pi**2/(8*spec["gamma"])) if low<=v<=high]])
    incumbent=None if fixed is None or not low<=config["fixed_dt"]<=high else fixed.copy()
    grid_trials=[]
    for dt in grid:
        # No grid probe needs more steps than the incumbent to improve it.
        cap=min(config["step_budget"],len(incumbent) if incumbent is not None else config["step_budget"])
        count=evaluator.constant(float(dt),cap)
        grid_trials.append(dict(dt=float(dt),steps=None if count<1 else count))
        if count>0 and (incumbent is None or count<len(incumbent)):
            incumbent=np.full(count,dt)
    results={}
    for family in ("constant","warped_cosine"):
        upper=min(config["step_budget"],len(incumbent) if incumbent is not None else config["step_budget"])
        bounds=[(1,upper),(np.log(low),np.log(high))]
        if family=="warped_cosine":
            bounds += [bounds[1],(np.log(config["minimum_power"]),np.log(config["maximum_power"]))]
        trials=[]
        best_parameters=None
        if incumbent is not None and len(incumbent)<=upper:
            best_parameters=[len(incumbent),np.log(incumbent[-1])]
            if family=="warped_cosine":best_parameters=[len(incumbent),np.log(incumbent[-1]),np.log(incumbent[0]),0.]
        for seed in config["optimizer_seeds"]:
            calls=0
            rng=np.random.default_rng(seed)
            population=np.column_stack([rng.uniform(a,b,max(5,config["population_multiplier"]*len(bounds))) for a,b in bounds])
            if best_parameters is not None:population[0]=best_parameters
            def objective(parameters):
                nonlocal incumbent,calls,best_parameters
                calls+=1
                dts=study.schedule(family,int(round(parameters[0])),parameters[1:])
                if dts.sum()>config["time_cap"]:return 2.+float(dts.sum()/config["time_cap"])
                check=evaluator.evaluate(dts)
                if check["passed"]:
                    if incumbent is None or len(dts)<len(incumbent):
                        incumbent=dts.copy();best_parameters=list(map(float,parameters))
                    return len(dts)/(upper+1)
                return 1.+check["maximum_forward_error"]/selection_config["mean_tolerance"]
            optimum=differential_evolution(objective,bounds,integrality=[True]+[False]*(len(bounds)-1),
                seed=seed,init=population,maxiter=config["iterations"],polish=False,tol=1e-6)
            trial=dict(seed=seed,calls=calls,optimizer_converged=bool(optimum.success),objective=float(optimum.fun),
                       selected_steps=None if incumbent is None else len(incumbent))
            trials.append(trial)
            if progress:progress(dict(event=family,**trial))
        record=dict(status="no_passing_schedule_found",selected_steps=None,trials=trials)
        if incumbent is not None:
            accelerated=evaluator.evaluate(incumbent,save=True)
            validation=independent_validation(n,spec,config,incumbent)
            disagreement=float(np.max(abs(np.asarray(accelerated["output_covariance"])-validation["output_covariance"])))
            if not validation["passed"] or disagreement>1e-9:raise RuntimeError("independent schedule verification failed")
            record.update(status="mean_forward_window_passed",selected_steps=len(incumbent),physical_time=float(incumbent.sum()),
                step_dts=incumbent.tolist(),validation=validation,kernel_maximum_disagreement=disagreement)
        if family=="constant":record["grid_trials"]=grid_trials
        results[family]=record
    record=dict(status="no_passing_schedule_found",selected_steps=None)
    if fixed is not None:
        validation=independent_validation(n,spec,config,fixed)
        if not validation["passed"]:raise RuntimeError("independent fixed baseline verification failed")
        record.update(status="mean_forward_window_passed",selected_steps=len(fixed),physical_time=float(fixed.sum()),
                      step_dts=fixed.tolist(),validation=validation)
    results["fixed"]=record
    return dict(status="search_complete",families=results,wall_seconds=time.perf_counter()-started,
        selection_tolerance=selection_config["mean_tolerance"],
        minimality="best passing schedules found in declared bounded searches; no minimum proof",
        sampling="deterministic mean only",settling="full-node finite forward window")


def case_worker(stage,name,n,spec,config,directory):
    directory=Path(directory)
    path=directory/f"{stage}_{name}_N{n}.json"
    if path.exists():return json.loads(path.read_text())
    def progress(data):
        data.update(model=name,N=n,utc=datetime.now(timezone.utc).isoformat())
        physics.atomic_json(directory/f"progress_{name}_N{n}.json",data)
        print(json.dumps(data),flush=True)
    started=time.perf_counter()
    if stage=="exact":result=study.exact_case(n,spec,config)
    elif stage=="schedules":result=schedules_case(n,spec,config,progress)
    else:
        source=json.loads((directory/f"schedules_{name}_N{n}.json").read_text())["result"]
        result=stream_compile.resources_case(n,spec,config,source)
    item=dict(model=name,N=n,parameters=spec,result=result,wall_seconds=time.perf_counter()-started)
    physics.atomic_json(path,item)
    print(f"COMPLETE {stage} {name} N={n} ({item['wall_seconds']:.2f}s)",flush=True)
    return item


def provenance(config):
    files=[study.DIRECTORY/name for name in ("study.py","full_study.py","accelerated.py","channel.cpp","stream_compile.py")]
    files.append(ROOT/"library/reservoir_scaling.py")
    return dict(configuration=config,versions={n:importlib.metadata.version(n) for n in ("numpy","scipy","qiskit")},
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=study.DIRECTORY/"config.json")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--stage",choices=("all","exact","schedules","resources"),default="all")
    args=parser.parse_args(arguments)
    config=json.loads(args.config.read_text());study.validate(config)
    source=provenance(config)
    signature=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:12]
    directory=study.DIRECTORY/"runs"/f"full_{signature}"
    directory.mkdir(parents=True,exist_ok=True)
    physics.atomic_json(directory/"configuration.json",source)
    print(f"RUN_DIRECTORY={directory}",flush=True)
    kernel()
    stages=("exact","schedules","resources") if args.stage=="all" else (args.stage,)
    for stage in stages:
        records=[]
        # The sandbox disallows the socket used by Python 3.14's forkserver.
        # Fork uses local pipes; BLAS is single-threaded in the run commands.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("fork")) as pool:
            jobs=[pool.submit(case_worker,stage,name,n,spec,config,str(directory))
                for name,spec in config["models"].items() for n in config["sizes"]]
            for future in as_completed(jobs):
                records.append(future.result())
        records.sort(key=lambda r:(r["model"],r["N"]))
        study.write_summary(directory,stage,records)
    print(f"FINISHED={directory}",flush=True)
    return directory


if __name__=="__main__":main()
