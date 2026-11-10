"""Exact resource accounting for a block-preserving offline Qiskit compiler.

Small gates are actually transpiled; counts and dependency depths of their
composition are streamed without retaining the full many-million-gate circuit.
No inter-block optimization is claimed. Test against materialized compositions.
"""
from collections import Counter
from functools import lru_cache

import numpy as np

from FinalModel.Experiment1 import study

BASES={"native": ["rz","sx","x","cx"], "axial": ["rz","h","s","sdg","x","z","cx"]}


@lru_cache(maxsize=8)
def parametric_template(kind,basis):
    from qiskit import QuantumCircuit,transpile
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import XXPlusYYGate
    size=2 if kind=="exchange" else 1
    circuit=QuantumCircuit(size)
    theta=Parameter("theta") if kind=="exchange" else None
    if kind=="exchange":
        circuit.append(XXPlusYYGate(theta),[0,1])
    elif kind=="h":
        circuit.h(0)
    elif kind=="x":
        circuit.x(0)
    else:
        raise ValueError(kind)
    return transpile(circuit,basis_gates=BASES[basis],optimization_level=1,seed_transpiler=11),theta


@lru_cache(maxsize=8192)
def template(kind,angle,basis):
    circuit,theta=parametric_template(kind,basis)
    return circuit if theta is None else circuit.assign_parameters({theta:angle})


def describe(circuit):
    size=circuit.num_qubits
    dependency=np.full((size,size),-np.inf)
    np.fill_diagonal(dependency,0.)
    two=np.zeros(size,dtype=int)
    generic,exact_t=0,0
    for instruction in circuit.data:
        qubits=[circuit.find_bit(q).index for q in instruction.qubits]
        dependency[qubits]=np.max(dependency[qubits],axis=0)+1
        if len(qubits)==2:
            two[qubits]=np.max(two[qubits])+1
        if instruction.operation.name=="rz":
            units=float(instruction.operation.params[0])/(np.pi/4)
            if abs(units-round(units))<1e-10:
                exact_t+=int(round(units))%2
            else:
                generic+=1
    return dict(counts=dict(circuit.count_ops()),dependency=dependency,two_depth=int(two.max()),
                generic=generic,exact_t=exact_t)


@lru_cache(maxsize=8192)
def compiled(kind,angle,basis):
    return describe(template(kind,angle,basis))


def exchange_layer(depth,two,left,block):
    """Disjoint two-qubit blocks, evaluated by their exact max-plus maps."""
    if len(left)==0:
        return
    right=left+1
    a,b=depth[left].copy(),depth[right].copy()
    mapping=block["dependency"]
    depth[left]=np.maximum(a+mapping[0,0],b+mapping[0,1])
    depth[right]=np.maximum(a+mapping[1,0],b+mapping[1,1])
    if block["two_depth"]:
        combined=np.maximum(two[left],two[right])+block["two_depth"]
        two[left]=combined; two[right]=combined


def compile_path(n,spec,dts,seed,basis,*,materialize=False):
    from qiskit import QuantumCircuit
    counts=Counter()
    depth=np.zeros(n+2)
    two=np.zeros(n+2,dtype=int)
    generic,exact_t=0,0
    circuit=QuantumCircuit(n+2,2*n) if materialize else None

    def add_one(kind,q):
        nonlocal generic,exact_t
        block=compiled(kind,0.,basis)
        counts.update(block["counts"])
        depth[q]+=block["dependency"][0,0]
        generic+=block["generic"];exact_t+=block["exact_t"]
        if circuit is not None:
            circuit.compose(template(kind,0.,basis),qubits=[q],inplace=True)

    if spec["preparation"]=="center_excitation":
        add_one("x",n//2+1)
    elif spec["preparation"]=="filled":
        for q in range(1,n+1):add_one("x",q)
    elif spec["preparation"]=="half_mixed":
        for q in range(1,n+1):
            add_one("h",q);depth[q]+=1;counts["measure"]+=1
            if circuit is not None:circuit.measure(q,q-1)
    probabilities=np.array([spec["p_left"],1-spec["p_left"],spec["p_right"],1-spec["p_right"]])/2
    choices=np.random.default_rng(seed).choice(4,size=len(dts),p=probabilities)
    for dt,choice in zip(dts,choices):
        angle=float(4*spec["prefactor"]*spec["J"]*dt)
        block=compiled("exchange",angle,basis)
        for parity in (0,1):
            left=np.arange(parity+1,n,2)
            exchange_layer(depth,two,left,block)
            counts.update({k:v*len(left) for k,v in block["counts"].items()})
            generic+=block["generic"]*len(left);exact_t+=block["exact_t"]*len(left)
            if circuit is not None:
                for q in left:circuit.compose(template("exchange",angle,basis),qubits=[int(q),int(q+1)],inplace=True)
        bath,site=(0,1) if choice<2 else (n+1,n)
        if choice%2==0:add_one("x",bath)
        bath_angle=float(2*np.sqrt(2*spec["gamma"]*dt))
        block=compiled("exchange",bath_angle,basis)
        # Keep the template's system,bath order even at the left boundary.
        pair=np.array([site,bath])
        before=depth[pair].copy()
        depth[pair]=np.max(block["dependency"]+before[None,:],axis=1)
        if block["two_depth"]:two[pair]=np.max(two[pair])+block["two_depth"]
        counts.update(block["counts"])
        generic+=block["generic"];exact_t+=block["exact_t"]
        depth[bath]+=1;counts["reset"]+=1
        if circuit is not None:
            circuit.compose(template("exchange",bath_angle,basis),qubits=[site,bath],inplace=True)
            circuit.reset(bath)
    counts["measure"]+=n
    depth[1:n+1]+=1
    if circuit is not None:circuit.measure(list(range(1,n+1)),list(range(n,2*n)))
    return dict(operations=dict(counts),depth=int(depth.max()),two_qubit_depth=int(two.max()),
                generic_rotations=int(generic),exact_t_rotations=int(exact_t),circuit=circuit)


def resources_case(n,spec,config,schedules):
    rows=[]
    for family,selection in schedules["families"].items():
        if selection["selected_steps"] is None:
            rows.append(dict(family=family,status="no_validated_schedule"));continue
        dts=selection["step_dts"]
        for path in range(config["resource_paths"]):
            seed=config["resource_seed"]+path
            native=compile_path(n,spec,dts,seed,"native")
            axial=compile_path(n,spec,dts,seed,"axial")
            R=axial["generic_rotations"]
            eta=config["synthesis_allowance"]/(2*max(1,4*spec["prefactor"])*R) if R else None
            T=axial["exact_t_rotations"]+(R*int(np.ceil(3*np.log2(1/eta)))) if R else axial["exact_t_rotations"]
            base_clifford=sum(v for k,v in axial["operations"].items() if k not in ("rz","measure","reset"))
            rows.append(dict(family=family,path=path,topology="line_and_all_to_all",status="compiled",
                compiler="Qiskit-local-blocks; exact streamed composition; no inter-block optimization",
                system_qubits=n,logical_qubits=n+2,steps=len(dts),logical_exchanges=n*len(dts),
                native_operations=native["operations"],native_depth=native["depth"],two_qubit_depth=native["two_qubit_depth"],
                clifford_plus_rz_operations=axial["operations"],
                fault_tolerant=dict(generic_rotations=R,exact_T_or_Tdg_rotations=axial["exact_t_rotations"],
                    Clifford_gates_excluding_RZ=base_clifford,per_rotation_operator_error_budget=eta,
                    T_count_leading_order_estimate=int(T),T_depth=None,physical_qubits=None,
                    method="leading 3 log2(1/eta), not actual synthesis or a certified bound")))
    return dict(status="resource_study_complete",records=rows,
                scope="exact resources of block-preserving compiled Z-readout paths; topologies coincide because every gate is on the line")
