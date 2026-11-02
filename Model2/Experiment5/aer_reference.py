"""Exact ten-system-qubit Aer density reference for selected frozen paths."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data,run_hardware,aer_sampling


def main(arguments=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths",type=int,nargs="+",default=[0,1])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"runs/aer_sampling/exact_reference.json")
    args=parser.parse_args(arguments)
    if min(args.paths)<0 or args.threads<1:
        raise ValueError("invalid paths or thread count")
    source=input_data.make_source(input_data.load_spec(),max(2,max(args.paths)+1),29)
    simulator=AerSimulator(method="density_matrix",max_parallel_threads=args.threads,max_parallel_experiments=1,
                           max_memory_mb=1500,fusion_threshold=1,fusion_max_qubit=2)
    report={"spec_sha256":input_data.SPEC_SHA256,"paths":[],
            "scope":"Exact nonselective preparation and exact immediate bath trace; no sampled shots or noise model."}
    for path in args.paths:
        circuit=aer_sampling.aer_body(run_hardware.build_body(source,path),10)
        for site in range(10):
            circuit.save_expectation_value(Pauli("Z"),[site],label=f"z{site}")
        for bond in range(9):
            circuit.save_expectation_value(Pauli("XY"),[bond,bond+1],label=f"yx{bond}")
            circuit.save_expectation_value(Pauli("YX"),[bond,bond+1],label=f"xy{bond}")
        circuit.save_probabilities(list(range(10)),label="probabilities")
        start=time.monotonic()
        result=simulator.run(circuit,shots=1).result()
        if not result.success:
            raise RuntimeError(str(result))
        data=result.data(0)
        C=np.asarray(source["final_covariances_real_gauge"][path])
        populations=np.asarray([(1-data[f"z{j}"])/2 for j in range(10)])
        yx=np.asarray([data[f"yx{j}"] for j in range(9)])
        xy=np.asarray([data[f"xy{j}"] for j in range(9)])
        currents=.25*(yx-xy)
        expected=np.diag(C,1)
        discrepancies=[np.max(abs(populations-np.diag(C))),np.max(abs(currents-expected)),
                       np.max(abs(yx-2*expected)),np.max(abs(xy+2*expected))]
        row={"path":path,"maximum_discrepancy":float(max(discrepancies)),
             "trace":float(np.sum(data["probabilities"])),"seconds":time.monotonic()-start,
             "populations":populations.tolist(),"currents_over_J":currents.tolist(),
             "metadata":result.results[0].metadata}
        row["passed"]=bool(row["maximum_discrepancy"]<1e-10 and abs(row["trace"]-1)<1e-10)
        report["paths"].append(row)
        atomic_json(args.output,report)
        print(f"Path {path}: density agreement={row['maximum_discrepancy']:.3g}, trace={row['trace']:.15g}, {row['seconds']:.2f}s",flush=True)
        if not row["passed"]:
            raise ValueError("density reference disagrees with covariance prediction")
    report["passed"]=all(row["passed"] for row in report["paths"])
    atomic_json(args.output,report)
    return report


if __name__=="__main__":
    main()
