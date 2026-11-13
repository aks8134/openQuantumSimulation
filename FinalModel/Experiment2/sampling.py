"""Ideal bath-choice trajectory dynamics and simultaneous sampling bounds."""
from __future__ import annotations
import ctypes
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from FinalModel.Experiment1 import phoenix_resources as portable
from FinalModel.Experiment1 import study as dynamics
from library import reservoir_scaling as physics

DIRECTORY=Path(__file__).resolve().parent
CONFIG=DIRECTORY/"config.json"
INPUT=DIRECTORY/"inputs/schedules.json"
FLAGS=["-O3","-std=c++17","-march=skylake-avx512","-mprefer-vector-width=512","-fopenmp","-shared","-fPIC"]


def settings():
    value=json.loads(CONFIG.read_text())
    if (value["mean_tolerance"]!=.016 or value["sampling_tolerance"]!=.004
            or value["total_tolerance"]!=.02):
        raise ValueError("The agreed ideal study uses 0.016 mean + 0.004 sampling = 0.02")
    if not 0<value["confidence"]<1 or value["confidence_case_count"]!=117:
        raise ValueError("Invalid family-wise confidence allocation")
    if value["checkpoint_multiplier"]!=2 or value["first_checkpoint"]<2:
        raise ValueError("The checkpoint grid must double from an initial sample >=2")
    return value


def export_inputs():
    source=portable.read_inputs(portable.DEFAULT_INPUT)
    value=portable.sealed(dict(artifact_type="ideal_trajectory_schedules_v1",
        source_schedule_input_sha256=source["sha256"], channel=source["channel"],cases=source["cases"],
        scope="Saved ideal schedules only; current tolerances are in Experiment2/config.json"))
    if INPUT.exists() and portable.read_sealed(INPUT)!=value:
        raise ValueError("Existing ideal schedule input differs")
    INPUT.parent.mkdir(parents=True,exist_ok=True)
    portable.atomic_json(INPUT,value)
    return value


def inputs():
    value=portable.read_sealed(INPUT)
    if value["artifact_type"]!="ideal_trajectory_schedules_v1" or len(value["cases"])!=117:
        raise ValueError("Expected all 117 saved ideal schedules")
    if value["channel"]!=dict(order="hamiltonian-first",system_formula="even-odd",sampling_policy="rate-weighted"):
        raise ValueError("Channel convention changed")
    for case in value["cases"]:
        portable.validate_case(case)
        spec=case["physics"]
        if (spec["J"],spec["gamma"],spec["prefactor"])!=(1.,1.,.5):
            raise ValueError("All models must retain J=gamma=1 and prefactor 1/2")
    return value


def physical_identity(case):
    return portable.digest(dict(N=case["N"],physics=case["physics"],schedule_sha256=case["schedule_sha256"]))


def canonical_groups(payload):
    result={}
    for case in payload["cases"]:
        key=physical_identity(case)
        result.setdefault(key,dict(physical_sha256=key,case=case,aliases=[]))["aliases"].append(case["id"])
    return list(result.values())


@lru_cache(maxsize=1)
def build_kernel():
    source=DIRECTORY/"trajectory.cpp"
    included=DIRECTORY.parent/"Experiment1/channel.cpp"
    compiler=subprocess.run(["/usr/bin/g++","--version"],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    identity=portable.digest(dict(source=portable.file_digest(source),included=portable.file_digest(included),
                                  flags=FLAGS,compiler=compiler))
    directory=ROOT/".local-artifacts/trajectory-kernel"/identity[:16]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"trajectory.so"
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory,suffix=".so",delete=False) as handle:
            temporary=Path(handle.name)
        subprocess.run(["/usr/bin/g++",*FLAGS,str(source),"-o",str(temporary)],check=True)
        temporary.replace(path)
    return path,dict(source_sha256=identity,binary_sha256=portable.file_digest(path),compiler=compiler,flags=FLAGS)


@lru_cache(maxsize=1)
def kernel():
    path,_=build_kernel()
    lib=ctypes.CDLL(str(path))
    i,d,u=ctypes.c_int,ctypes.c_double,ctypes.c_uint64
    a=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    ai=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    lib.trajectory_reference.argtypes=[i]+[d]*5+[a,a,i,i,a]
    lib.trajectory_reference.restype=None
    lib.trajectory_given_choices.argtypes=[i]+[d]*3+[a,a,i,i,ai,i,a,a]
    lib.trajectory_given_choices.restype=i
    lib.trajectory_batch.argtypes=[i]+[d]*5+[a,a,i,i,a,u,u,i,i,i,a,a,a]
    lib.trajectory_batch.restype=i
    return lib


def provenance(config,payload):
    _,binary=build_kernel()
    sources=[Path(__file__),DIRECTORY/"trajectory.cpp",DIRECTORY/"run_study.py",CONFIG,
        DIRECTORY.parent/"Experiment1/channel.cpp",Path(dynamics.__file__),Path(physics.__file__),Path(portable.__file__)]
    return dict(configuration=config,input_sha256=payload["sha256"],kernel=binary,
        sources={str(path.relative_to(ROOT)):portable.file_digest(path) for path in sources},
        versions=dict(python=list(sys.version_info[:3]),**{n:importlib.metadata.version(n) for n in ("numpy","scipy")}))


class Evaluator:
    def __init__(self,case,config):
        self.case,self.config=case,config
        self.n=case["N"]
        self.spec=case["physics"]
        self.dts=np.ascontiguousarray(portable.steps(case),dtype=np.float64)
        self.tail=int(np.ceil(config["forward_window"]/self.dts[-1]))
        self.initial=np.ascontiguousarray(dynamics.initial(self.n,self.spec["preparation"]))
        self.rank_one=self.spec["preparation"]=="center_excitation" and self.spec["p_left"]==self.spec["p_right"]==0
        self.common=[self.n,self.spec["prefactor"],self.spec["J"],self.spec["gamma"],
                     self.spec["p_left"],self.spec["p_right"],self.initial,self.dts,len(self.dts),self.tail]
        self.reference=np.empty((self.tail+1,2*self.n-1))
        kernel().trajectory_reference(*self.common,self.reference)
        target=dynamics.stationary(dynamics.model_for(self.n,self.spec))
        self.target=np.r_[np.diag(target),4*self.spec["prefactor"]*np.diag(target,1)]
        self.bias=np.abs(self.reference-self.target)
        self.maximum_mean_error=float(self.bias.max())
        if (self.maximum_mean_error>config["mean_tolerance"]+config["numerical_guard"]
                or abs(self.maximum_mean_error-case["mean_validation"]["maximum_forward_error"])>1e-9):
            raise ValueError("Saved schedule no longer matches its independently validated mean")
        self.ranges=np.r_[np.ones(self.n),np.full(self.n-1,4*self.spec["prefactor"])]
        self.seed=int(physical_identity(case)[:16],16)^config["seed"]

    def batch(self,first,count,*,threads=None):
        if first<0 or count<1:
            raise ValueError("Invalid trajectory range")
        sums=np.empty_like(self.reference)
        squares=np.empty_like(self.reference)
        bounds=np.empty(3)
        code=kernel().trajectory_batch(*self.common,self.reference,self.seed,first,count,
            threads or self.config["threads"],int(self.rank_one),sums,squares,bounds)
        if code or not np.all(np.isfinite(sums)) or not np.all(np.isfinite(squares)):
            raise ValueError("A simulated observable violates its physical support")
        return sums,squares,bounds

    def given_choices(self,choices,*,rank_one=False):
        choices=np.ascontiguousarray(choices,dtype=np.int32)
        if choices.shape!=(len(self.dts)+self.tail,) or np.any((choices<0)|(choices>3)):
            raise ValueError("One valid bath choice is required for every step")
        probabilities=np.array([self.spec["p_left"],1-self.spec["p_left"],self.spec["p_right"],1-self.spec["p_right"]])/2
        if np.any(probabilities[choices]==0):
            raise ValueError("Zero-probability bath branches cannot be sampled")
        if rank_one and (not self.rank_one or np.any(choices%2==0)):
            raise ValueError("Rank-one evolution requires loss-only branches")
        output=np.empty((self.n,self.n))
        observations=np.empty_like(self.reference)
        kernel().trajectory_given_choices(self.n,self.spec["prefactor"],self.spec["J"],self.spec["gamma"],
            self.initial,self.dts,len(self.dts),self.tail,choices,int(rank_one),output,observations)
        return output,observations


def look_index(count,config):
    ratio=count/config["first_checkpoint"]
    look=int(round(math.log2(ratio))) if ratio>0 else -1
    if look<0 or count!=config["first_checkpoint"]*(2**look):
        raise ValueError("Confidence is evaluated only at the prespecified doubling checkpoints")
    return look


def radius(variance,count,ranges,delta):
    """Two-sided Maurer--Pontil Theorem 4, scaled to each bounded range."""
    if count<2 or not 0<delta<1:
        raise ValueError("Invalid confidence inputs")
    logarithm=math.log(4/delta)
    return np.sqrt(2*np.maximum(variance,0)*logarithm/count)+7*ranges*logarithm/(3*(count-1))


def assess(evaluator,sums,squares,count):
    config=evaluator.config
    look=look_index(count,config)
    mean=sums/count
    centered=squares-sums*mean
    if np.any(centered < -1e-10*np.maximum(1,squares)):
        raise ValueError("Invalid accumulated variance")
    variance=np.maximum(centered,0)/(count-1)
    d=sums.shape[1]
    # Sum_l 1/((l+1)(l+2))=1. Split alpha across 117 cases, both
    # output/window claims, every component/time and every checkpoint.
    delta=(1-config["confidence"])/(config["confidence_case_count"]*2*(look+1)*(look+2))
    output_radius=radius(variance[0],count,evaluator.ranges,delta/d)
    window_radius=radius(variance,count,evaluator.ranges,delta/(d*sums.shape[0]))
    actual=np.abs(mean)
    target_error=np.abs(evaluator.reference+mean-evaluator.target)
    guard=config["numerical_guard"]
    output_width=float(output_radius.max())+guard
    window_width=float(window_radius.max())+guard
    return dict(R=count,look=look,allocated_failure_probability=delta,
        output_confidence_radius=output_width,window_confidence_radius=window_width,
        output_sampling_error=float(actual[0].max()),window_sampling_error=float(actual.max()),
        output_population_error=float(actual[0,:evaluator.n].max()),
        output_current_error=float(actual[0,evaluator.n:].max()),
        window_population_error=float(actual[:,:evaluator.n].max()),
        window_current_error=float(actual[:,evaluator.n:].max()),
        output_total_error=float(target_error[0].max()),window_total_error=float(target_error.max()),
        output_certified=bool(output_width<=config["sampling_tolerance"]
            and actual[0].max()<=config["sampling_tolerance"] and target_error[0].max()<=config["total_tolerance"]),
        window_certified=bool(window_width<=config["sampling_tolerance"]
            and actual.max()<=config["sampling_tolerance"] and target_error.max()<=config["total_tolerance"]),
        maximum_variance=float(variance.max()))
