"""Noiseless Aer shot study for the frozen cosine44 hardware circuits.

The default traces freshly reset baths into exact one-site Kraus channels.
--full retains both baths, preparation records and all resets. Both sample
actual final outcomes; expectation snapshots are never substituted for shots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data,run_hardware
from Model3.Experiment5.check_aer_populations import bath_channel


def aer_body(body,n,full=False):
    circuit=QuantumCircuit(*body.qregs,*body.cregs) if full else QuantumCircuit(n)
    occupied={q:0 for q in range(n,body.num_qubits)}
    awaiting=None
    dephase=Kraus([np.diag([1,0]),np.diag([0,1])]).to_instruction()
    for instruction in body.data:
        op=instruction.operation
        q=[body.find_bit(bit).index for bit in instruction.qubits]
        if not full and awaiting is not None and (op.name!="reset" or q!=[awaiting]):
            raise ValueError("bath collision must be immediately reset")
        if not full and op.name=="measure":
            if q[0]>=n:
                raise ValueError("unexpected bath preparation measurement")
            circuit.append(dephase,q)
        elif not full and op.name=="x" and q[0]>=n:
            occupied[q[0]]^=1
        elif not full and op.name=="reset":
            if q!=[awaiting]:
                raise ValueError("unexpected reset")
            occupied[q[0]]=0;awaiting=None
        elif not full and op.name=="xx_plus_yy" and q[1]>=n:
            circuit.append(bath_channel(op,occupied[q[1]]),[q[0]])
            awaiting=q[1]
        elif op.name=="xx_plus_yy":
            if op.params[1]!=0:
                raise ValueError("only zero-phase exchange supported")
            circuit.rxx(float(op.params[0])/2,*q)
            circuit.ryy(float(op.params[0])/2,*q)
        else:
            circuit.append(op,q,instruction.clbits if full else [])
    if awaiting is not None:
        raise ValueError("unreset bath")
    circuit.metadata={"experiment":"Model2/Experiment5/Aer","full_baths":full,
                      "preparation":"nonselective original measurements"}
    return circuit


def packed_memory(memory,circuit,n):
    outcome=next(reg for reg in circuit.cregs if reg.name=="outcome")
    offsets=[circuit.find_bit(bit).index for bit in outcome]
    packed=[]
    for bitstring in memory:
        raw=int(bitstring.replace(" ",""),2)
        packed.append(sum(((raw>>offset)&1)<<j for j,offset in enumerate(offsets)))
    return np.asarray(packed,dtype=np.uint16)


def estimate(samples):
    # shape: paths, three bases, shots; final bit j is system site j.
    bits=((samples[...,None]>>np.arange(10))&1).astype(float)
    populations=bits[:,0].mean(axis=1)
    signs=1-2*bits
    pairs=(signs[...,:-1]*signs[...,1:]).mean(axis=2)
    parity=np.where(np.arange(9)%2==0,1.,-1.)
    currents=.25*parity*(pairs[:,2]-pairs[:,1])
    return np.concatenate([populations,currents],axis=1)


def planning_rows(source,path_counts,shot_counts,alpha=.05):
    C=np.asarray(source["final_covariances_real_gauge"])
    inf=np.r_[source["mean_channel_observables"]["populations"][0][-1],source["mean_channel_observables"]["excitation_flows"][0][-1]]
    target=np.r_[np.full(10,.25),np.zeros(9)]
    rows=[]
    for R in path_counts:
        populations=np.diagonal(C[:R],axis1=-2,axis2=-1)
        off=np.diagonal(C[:R],offset=1,axis1=-2,axis2=-1)
        ideal=np.r_[populations.mean(0),off.mean(0)]
        bias=ideal-inf
        # Populations are Bernoulli. U(1) symmetry gives <YX>=-<XY>=2*C_j,j+1.
        base_variance=np.r_[(populations*(1-populations)).mean(0),(.125*(1-4*off**2)).mean(0)]
        for S in shot_counts:
            variance=np.maximum(base_variance,0)/(R*S)+1e-24
            radius=np.r_[np.full(10,1/(R*S)),np.full(9,.5/(R*S))]
            lower=np.maximum(target-.02,inf-.004)
            upper=np.minimum(target+.02,inf+.004)
            def tails(distance):
                positive=np.maximum(distance,0)
                return np.where(distance>0,np.exp(-positive**2/(2*(variance+radius*positive/3))),1.)
            failure=min(1.,float(np.sum(tails(upper-ideal)+tails(ideal-lower))))
            logarithm=np.log(38/alpha)
            widths=radius*logarithm/3+np.sqrt((radius*logarithm/3)**2+2*variance*logarithm)
            rows.append({"trajectories":R,"shots":S,"ideal_target_error":float(max(abs(ideal-target))),
                "ideal_sampling_distance":float(max(abs(bias))),"simultaneous_failure_probability_bound":failure,
                "sufficient_at_confidence":1-alpha,"sufficient":bool(failure<=alpha),
                "uniform_95_target_upper":float(max(abs(ideal-target)+widths)),
                "uniform_95_sampling_upper":float(max(abs(bias)+widths)),
                "conditional_shot_standard_errors":np.sqrt(variance).tolist(),
                "evolution_shots":3*R*S,"jobs":int(np.ceil(R/169)),
                "hardware_shots_including_controls":3*R*S+3*2048*int(np.ceil(R/169))})
    return rows


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths",type=int,default=1024)
    parser.add_argument("--shots",type=int,default=8)
    parser.add_argument("--seed",type=int,default=29)
    parser.add_argument("--aer-seed",type=int,default=2026)
    parser.add_argument("--chunk",type=int,default=16)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output-directory",type=Path,default=Path(__file__).parent/"runs/aer_sampling")
    options=parser.parse_args(arguments)
    if not 2<=options.paths<=16384 or min(options.shots,options.chunk,options.threads)<1 or min(options.seed,options.aer_seed)<0:
        raise ValueError("invalid simulation settings")
    configuration={key:value for key,value in vars(options).items() if key not in ("resume","output_directory")}
    configuration["spec_sha256"]=input_data.SPEC_SHA256
    configuration["code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tag=input_data.digest(configuration)[:12]
    folder=options.output_directory/f"aer_{tag}"
    folder.mkdir(parents=True,exist_ok=True)
    if options.resume and (folder/"source.json").exists():
        source=json.loads((folder/"source.json").read_text())
    else:
        source=input_data.make_source(input_data.load_spec(),options.paths,options.seed)
        atomic_json(folder/"source.json",source)
    report={"configuration":configuration,"versions":{name:importlib.metadata.version(name) for name in ("qiskit","qiskit-aer","numpy")},
            "source_sha256":hashlib.sha256((folder/"source.json").read_bytes()).hexdigest(),"chunks":[],"complete":False}
    simulator=AerSimulator(method="statevector",max_parallel_threads=options.threads,max_parallel_experiments=1,
        max_parallel_shots=options.threads,fusion_threshold=1,fusion_max_qubit=2)
    arrays=[]
    for begin in range(0,options.paths,options.chunk):
        end=min(begin+options.chunk,options.paths)
        file=folder/f"samples_{begin:05d}_{end:05d}.npz"
        metadata_file=file.with_suffix(".json")
        if options.resume and file.exists() and metadata_file.exists():
            metadata=json.loads(metadata_file.read_text())
            if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata["sha256"]:
                raise ValueError("saved Aer samples changed")
            values=np.load(file)["samples"]
        else:
            publications=[]
            for path in range(begin,end):
                body=aer_body(run_hardware.build_body(source,path),10,options.full)
                for basis in ("z","xy","yx"):
                    publications.append(run_hardware.circuits.append_readout(body,list(range(10)),basis))
            started=time.monotonic()
            result=simulator.run(publications,shots=options.shots,memory=True,
                                 seed_simulator=options.aer_seed+begin*271).result()
            if not result.success:
                raise RuntimeError(str(result))
            values=np.asarray([packed_memory(result.get_memory(index),circuit,10)
                               for index,circuit in enumerate(publications)]).reshape(end-begin,3,options.shots)
            np.savez_compressed(file,samples=values)
            metadata={"begin":begin,"end":end,"seconds":time.monotonic()-started,
                "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                "aer_metadata":result.results[0].metadata,"shots":options.shots}
            atomic_json(metadata_file,metadata)
        arrays.append(values);report["chunks"].append(metadata)
        atomic_json(folder/"study.json",report)
        print(f"Aer paths {end}/{options.paths}: chunk {metadata['seconds']:.2f}s",flush=True)
    values=np.concatenate(arrays)
    measured=estimate(values)
    C=np.asarray(source["final_covariances_real_gauge"])
    ideal=np.c_[np.diagonal(C,axis1=-2,axis2=-1),np.diagonal(C,offset=1,axis1=-2,axis2=-1)]
    infinity=np.r_[source["mean_channel_observables"]["populations"][0][-1],source["mean_channel_observables"]["excitation_flows"][0][-1]]
    target=np.r_[np.full(10,.25),np.zeros(9)]
    mean=measured.mean(0);se=measured.std(axis=0,ddof=1)/np.sqrt(options.paths)
    report.update(complete=True,observables=mean.tolist(),same_path_ideal=ideal.mean(0).tolist(),
        mean_channel=infinity.tolist(),physical_target=target.tolist(),
        maximum_target_error=float(max(abs(mean-target))),maximum_mean_channel_error=float(max(abs(mean-infinity))),
        maximum_same_path_error=float(max(abs(mean-ideal.mean(0)))),maximum_two_SE=float(2*max(se)),
        across_path_standard_errors=se.tolist(),total_simulation_seconds=sum(row["seconds"] for row in report["chunks"]),
        empirical_checks_passed=bool(max(abs(mean-target))<=.02 and max(abs(mean-infinity))<=.004 and 2*max(se)<=.004),
        scope="Actual Aer final-outcome samples; exact fresh-bath partial trace unless --full, including preparation/reset randomness; no hardware noise or calibration controls. Conditional concentration planning is separate from Aer samples.")
    atomic_json(folder/"study.json",report)
    print(json.dumps({key:report[key] for key in ("maximum_target_error","maximum_mean_channel_error","maximum_same_path_error","maximum_two_SE","empirical_checks_passed","total_simulation_seconds")}),flush=True)
    print(f"Saved {folder/'study.json'}",flush=True)
    return report


if __name__=="__main__":
    main()
