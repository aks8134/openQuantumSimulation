"""Offline noisy FakeKingston comparison of Model2 raw, twirled and ZNE populations.

The frozen cosine44 schedule and path seed are unchanged. No IBM service is
constructed. Snapshot errors are copied onto the twelve active qubits before
simulation, preserving physical calibration while avoiding 156-qubit allocation.
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
from qiskit import qpy,transpile
from qiskit.providers import BackendV2,Options
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKingston

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data,run_hardware
from Model3.Experiment5 import mitigation as mit
from Model3.Experiment5 import run_mitigation as protocol

DIRECTORY=Path(__file__).resolve().parent
DEFAULT_PATH=[95,94,93,79,73,74,75,59,55,54,53,52]


class SnapshotSubset(BackendV2):
    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def run(self,*args,**kwargs):
        raise RuntimeError("snapshot description only; run circuits through local Aer")


def compact_backend(original,path):
    active=path[1:-1]+[path[0],path[-1]]
    if len(set(active))!=len(active):
        raise ValueError("physical path must be simple")
    lookup={q:i for i,q in enumerate(active)}
    backend=SnapshotSubset(name="fake_kingston_active_subset")
    backend._target=Target(num_qubits=len(active),dt=original.dt,
        qubit_properties=[original.target.qubit_properties[q] for q in active])
    for name in ("id","rz","sx","x","cz","measure","reset","delay"):
        if name not in original.target.operation_names:
            continue
        properties={tuple(lookup[q] for q in args):value for args,value in original.target[name].items()
                    if args is not None and set(args)<=lookup.keys()}
        if properties:
            backend._target.add_instruction(original.target.operation_from_name(name),properties)
    for left,right in zip(path,path[1:]):
        if not backend.target.instruction_supported(operation_name="cz",qargs=(lookup[left],lookup[right])):
            raise ValueError("physical path lacks a supported CZ edge")
    return backend,active


def canonical_noise(model):
    data=model.to_dict(serializable=True)
    for error in data["errors"]:
        # QuantumError UUIDs are regenerated; they are not physical parameters.
        error.pop("id",None)
    return data


def final_counts(counts,circuit,n):
    outcome=next(register for register in circuit.cregs if register.name=="outcome")
    offsets=[circuit.find_bit(bit).index for bit in outcome]
    output={}
    for key,count in counts.items():
        raw=int(key.replace(" ",""),2)
        label=format(sum(((raw>>offset)&1)<<j for j,offset in enumerate(offsets)),f"0{n}b")
        output[label]=output.get(label,0)+int(count)
    return output


def compile_unit(records,source,backend,cfg):
    first=records[0]
    control=first["kind"]=="control"
    body=(run_hardware.build_body(source,0,steps=0,populations=first["expected_populations"])
          if control else run_hardware.build_body(source,first["trajectory"]))
    native=transpile(body,backend=backend,initial_layout=list(range(12)),
        optimization_level=3,seed_transpiler=cfg["seed_transpiler"])
    if native.layout.final_index_layout(filter_ancillas=True)!=list(range(12)):
        raise ValueError("compact fixed mapping changed")
    native=run_hardware.circuits.append_readout(native,list(range(10)),"z")
    if not control and (native.count_ops().get("cz")!=880 or native.count_ops().get("reset")!=44):
        raise ValueError("base circuit differs from cosine44 resources")
    pubs=[]
    for row in records:
        circuit=protocol.make_variant(native,cfg,row)
        protocol.assert_supported(circuit,backend)
        row["metrics"]=run_hardware.circuits.metrics(circuit)
        if circuit.count_ops().get("reset",0)!=(0 if control else 44):
            raise ValueError("mitigation changed reservoir resets")
        pubs.append(circuit)
    return pubs


def simulate_unit(pubs,records,simulator,seed):
    observations=[None]*len(records)
    metadata=[]
    for group,shots in enumerate(sorted(set(row["shots"] for row in records))):
        indices=[i for i,row in enumerate(records) if row["shots"]==shots]
        start=time.monotonic()
        result=simulator.run([pubs[i] for i in indices],shots=shots,seed_simulator=seed+group).result()
        if not result.success:
            raise RuntimeError(str(result))
        for local,index in enumerate(indices):
            counts=final_counts(result.get_counts(local),pubs[index],10)
            if sum(counts.values())!=shots:
                raise ValueError("Aer shot count mismatch")
            observations[index]={"outcome_counts":counts}
        metadata.append({"shots_per_variant":shots,"circuits":len(indices),"seconds":time.monotonic()-start,
                         "seed_simulator":seed+group,"aer_metadata":result.results[0].metadata})
    return observations,metadata


def analyze(units,source,cfg):
    R=cfg["trajectory_count"]
    controls={arm:{name:[] for name in ("all_zero","all_one","initial_mixture")} for arm in ("raw","twirled")}
    paths={}
    for unit in units:
        for row,observation in zip(unit["records"],unit["observations"]):
            counts=mit.undo_measurement_twirl(observation["outcome_counts"],10,row["measurement_mask"])
            if row["kind"]=="control":
                controls[row["arm"]][row["control"]].append(counts)
            else:
                key=(row["trajectory"],0 if row["arm"]=="raw" else row["factor_index"]+1)
                paths.setdefault(key,[]).append(counts)
    values=np.zeros((R,4,10));covariances=np.zeros((R,4,10,10))
    for key,groups in paths.items():
        values[key],covariances[key]=mit.group_moments(groups,10)
    if len(paths)!=4*R:
        raise ValueError("missing matched trajectory/arm results")
    ideal=np.asarray(source["final_covariances_real_gauge"]).diagonal(axis1=-2,axis2=-1)
    factors=[1+2*mit.fold_count(f,880)/880 for f in cfg["fold_factors"]]
    stats=mit.estimates(values,covariances,controls,factors,ideal,np.full(10,.25),
        replicates=cfg["bootstrap"],seed=cfg["seed"],return_samples=True)
    draws=stats.pop("_bootstrap_samples");ideal_draws=stats.pop("_bootstrap_ideal")
    for metric,reducer in [("ideal_mae",np.mean),("ideal_maximum",np.max)]:
        delta=reducer(abs(draws["zne"]-ideal_draws),axis=1)-reducer(abs(draws["raw"]-ideal_draws),axis=1)
        stats["comparisons"].append({"method":"zne","reference":"raw","metric":metric,
            "difference":stats["profiles"]["zne"][metric]-stats["profiles"]["raw"][metric],
            "bootstrap_ci95":np.quantile(delta,[.025,.975])})
    return {"experiment":"Model2/Experiment6/FakeKingston_ZNE","configuration":cfg,"statistics":stats,
        "ideal_same_paths":ideal.mean(0),"physical_target":np.full(10,.25),
        "per_path_populations":values,"per_path_ideal_populations":ideal,
        "scope":"Population-only noisy snapshot pilot. Native CZ folding and manual gate/terminal twirling, raw and exploratory marginal SPAM inversions. No live hardware, no currents, no 0.02 certification."}


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths",type=int,default=32)
    parser.add_argument("--shots",type=int,default=128)
    parser.add_argument("--control-shots",type=int,default=2048)
    parser.add_argument("--randomizations",type=int,default=4)
    parser.add_argument("--seed",type=int,default=29)
    parser.add_argument("--sim-seed",type=int,default=2026)
    parser.add_argument("--seed-transpiler",type=int,default=11)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--physical-path",type=int,nargs="+",default=DEFAULT_PATH)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output-directory",type=Path,default=DIRECTORY/"runs")
    options=parser.parse_args(arguments)
    if (options.paths<2 or options.randomizations<2 or options.randomizations%2 or options.threads<1
            or options.bootstrap<100 or min(options.seed,options.sim_seed,options.seed_transpiler)<0
            or min(options.shots,options.control_shots)<2*options.randomizations
            or options.shots%options.randomizations or options.control_shots%options.randomizations
            or len(options.physical_path)!=12 or len(set(options.physical_path))!=12):
        raise ValueError("invalid paths, shots, randomizations, seeds or physical path")
    cfg={"trajectory_count":options.paths,"shots":options.shots,"control_shots":options.control_shots,
        "randomizations":options.randomizations,"seed":options.seed,"sim_seed":options.sim_seed,
        "seed_transpiler":options.seed_transpiler,"threads":options.threads,"bootstrap":options.bootstrap,
        "physical_path":options.physical_path,"steps":44,"fold_factors":[1.,1.25,1.5],
        "controls_per_batch":True,"batch_size":10**9,"spec_sha256":input_data.SPEC_SHA256,
        "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tag=input_data.digest(cfg)[:12]
    folder=options.output_directory/f"fake_zne_{tag}"
    folder.mkdir(parents=True,exist_ok=True)
    original=FakeKingston()
    backend,active=compact_backend(original,options.physical_path)
    model=NoiseModel.from_backend(backend)
    noise=canonical_noise(model)
    atomic_json(folder/"noise_model.json",noise)
    cfg["noise_model_sha256"]=input_data.digest(noise)
    cfg["active_compact_to_physical"]=active
    cfg["snapshot_date"]=original.properties().last_update_date.isoformat()
    cfg["versions"]={name:importlib.metadata.version(name) for name in ("qiskit","qiskit-aer","qiskit-ibm-runtime")}
    cfg["implementation_sha256"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(module.__file__) for module in (input_data,run_hardware,run_hardware.circuits,mit,protocol)]}
    if options.resume and (folder/"source.json").exists():
        source=json.loads((folder/"source.json").read_text())
    else:
        source=input_data.make_source(input_data.load_spec(),options.paths,options.seed)
        atomic_json(folder/"source.json",source)
    records,_=protocol.plan(cfg)
    cfg["total_circuits"]=len(records);cfg["total_shots"]=sum(row["shots"] for row in records)
    cfg["source_sha256"]=input_data.digest(source)
    if options.resume and (folder/"configuration.json").exists():
        if json.loads((folder/"configuration.json").read_text())!=cfg:
            raise ValueError("saved configuration, code, snapshot or input changed")
    atomic_json(folder/"configuration.json",cfg)
    simulator=AerSimulator(method="statevector",noise_model=model,max_parallel_threads=options.threads,
        max_parallel_experiments=1,max_parallel_shots=options.threads,fusion_threshold=1,fusion_max_qubit=2)
    print(f"FakeKingston: {options.paths} matched paths; {cfg['total_circuits']} circuits, {cfg['total_shots']} actual shots.",flush=True)
    units=[]
    keys=[("control",name) for name in ("all_zero","all_one","initial_mixture")]+[("trajectory",i) for i in range(options.paths)]
    for index,(kind,key) in enumerate(keys):
        filename=f"{kind}_{key}"
        record_file=folder/f"{filename}.json"
        selected=[row for row in records if row["kind"]==kind and row.get("trajectory",row.get("control"))==key]
        if options.resume and record_file.exists():
            unit=json.loads(record_file.read_text())
            if unit["noise_model_sha256"]!=cfg["noise_model_sha256"]:
                raise ValueError("noise snapshot changed")
            if unit["source_sha256"]!=cfg["source_sha256"]:
                raise ValueError("source trajectories changed")
            if len(unit["records"])!=len(selected) or any(
                    any(saved.get(k)!=v for k,v in planned.items())
                    for saved,planned in zip(unit["records"],selected)):
                raise ValueError("cached protocol records changed")
            if hashlib.sha256((folder/unit["qpy_file"]).read_bytes()).hexdigest()!=unit["qpy_sha256"]:
                raise ValueError("saved variants changed")
        else:
            pubs=compile_unit(selected,source,backend,cfg)
            qpy_file=folder/f"{filename}.qpy"
            with qpy_file.open("wb") as handle:qpy.dump(pubs,handle)
            observations,metadata=simulate_unit(pubs,selected,simulator,options.sim_seed+100000*(index+1))
            unit={"records":selected,"observations":observations,"simulation":metadata,
                  "noise_model_sha256":cfg["noise_model_sha256"],"qpy_file":qpy_file.name,
                  "source_sha256":cfg["source_sha256"],
                  "qpy_sha256":hashlib.sha256(qpy_file.read_bytes()).hexdigest()}
            atomic_json(record_file,unit)
        units.append(unit)
        print(f"Completed {kind} {key}; {sum(row['seconds'] for row in unit['simulation']):.2f}s",flush=True)
    report=plain(analyze(units,source,cfg))
    report["simulation_seconds"]=sum(row["seconds"] for unit in units for row in unit["simulation"])
    atomic_json(folder/"analysis.json",report)
    print(json.dumps({name:{k:row[k] for k in ("ideal_mae","ideal_maximum","physical_maximum")}
                      for name,row in report["statistics"]["profiles"].items()},indent=2),flush=True)
    print(f"Saved {folder/'analysis.json'}",flush=True)
    return report


if __name__=="__main__":
    main()
