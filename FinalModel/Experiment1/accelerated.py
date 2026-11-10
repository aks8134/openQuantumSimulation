"""Locally compiled numerical mean-channel kernel, independently checked in tests."""
from functools import lru_cache
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from FinalModel.Experiment1 import study


@lru_cache(maxsize=1)
def kernel():
    source=Path(__file__).with_name("channel.cpp")
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory=Path(tempfile.gettempdir())/f"finalmodel_channel_{digest}"
    directory.mkdir(exist_ok=True)
    library=directory/"channel.so"
    if not library.exists():
        # Atomic publication allows multiple case workers to start together.
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".so", delete=False) as f:
            temporary=Path(f.name)
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(temporary)], check=True)
        temporary.replace(library)
    lib=ctypes.CDLL(str(library))
    integer, scalar=ctypes.c_int, ctypes.c_double
    array=np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.evolve.argtypes=[integer]+[scalar]*5+[array]*3+[integer]+[scalar]*2+[integer]+[array]*2
    lib.evolve.restype=integer
    lib.constant_crossing.argtypes=[integer]+[scalar]*5+[array]*2+[scalar,integer,scalar,scalar,integer]
    lib.constant_crossing.restype=integer
    return lib


class Evaluator:
    def __init__(self,n,spec,config):
        self.n,self.spec,self.config=n,spec,config
        self.model=study.model_for(n,spec)
        self.initial=np.ascontiguousarray(study.initial(n,spec["preparation"]))
        self.target=np.ascontiguousarray(study.stationary(self.model))
        self.common=[n,spec["prefactor"],spec["J"],spec["gamma"],spec["p_left"],spec["p_right"],self.initial,self.target]

    def evaluate(self,dts,*,save=False):
        dts=np.ascontiguousarray(dts,dtype=float)
        if dts.ndim!=1 or len(dts)==0 or np.any(dts<=0) or not np.all(np.isfinite(dts)):
            raise ValueError("nonempty, finite, positive step list required")
        output=np.empty((self.n,self.n))
        stats=np.zeros(4)
        passed=kernel().evolve(*self.common,dts,len(dts),self.config["mean_tolerance"],
            self.config["forward_window"],self.config["validation_step_cap"],output,stats)
        return {"passed":bool(passed),"output_error":float(stats[0]),"maximum_forward_error":float(stats[1]),
                "forward_steps":int(stats[2]),"forward_steps_evaluated":int(stats[3]),
                "forward_duration":float(stats[2]*dts[-1]),"output_covariance":output.tolist() if save else None}

    def constant(self,dt,cap):
        cap=min(cap,int(self.config["time_cap"]/dt))
        return kernel().constant_crossing(*self.common,dt,cap,self.config["mean_tolerance"],
            self.config["forward_window"],self.config["validation_step_cap"])
