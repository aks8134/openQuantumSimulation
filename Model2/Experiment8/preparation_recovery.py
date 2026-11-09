"""Verify legacy unregistered QPY against its source, compiler and DD schedule."""
from collections import defaultdict

import numpy as np
from qiskit.converters import circuit_to_dag

from Model2.Experiment8.preparation import native_template


def wire_signatures(circuit):
    wires = defaultdict(list)
    for item in circuit.data:
        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
        signature = (item.operation.name, item.operation.label, qargs,
            tuple(circuit.find_bit(b).index for b in item.clbits), tuple(float(p) for p in item.operation.params))
        for q in qargs:
            wires[q].append(signature)
    return dict(wires)


def recover_group(source, configuration, rows, programs, backend, *, shard_name, target_only=False):
    from Model2.Experiment8 import run_hardware as run
    if len(programs) != len(rows) or [(p.metadata or {}).get("pub_index") for p in programs] != [r["pub_index"] for r in rows]:
        raise ValueError("legacy QPY group coverage or order changed")
    native = native_template(source, configuration, rows[0], backend, target_only=target_only)
    path = configuration["physical_path"]
    layout = path[1:-1]+[path[0], path[-1]]
    policy = dict(configuration["dd_policy"], physical_qubits=path[1:-1])
    windows = []
    vector = np.zeros(backend.num_qubits, dtype=np.int64)
    for index, (row, program) in enumerate(zip(rows, programs)):
        variant = run.protocol.make_variant(native, configuration, row)
        # Legacy files lack a receipt. Rebuild and independently check the full
        # paired schedules once, then certify the untouched saved QPY. Future
        # resumes use its sealed receipt and do not repeat this computation.
        baseline, original = run.timing.schedule(variant, backend.target,
            block_ordering_callable=run.flat_block_order)
        expected, events = run.timing.schedule(variant, backend.target, policy=policy,
            block_ordering_callable=run.flat_block_order)
        if circuit_to_dag(expected) != circuit_to_dag(program) or wire_signatures(expected) != wire_signatures(program):
            raise ValueError(f"legacy QPY differs from the saved trajectory/twirl/DD schedule: {row['case_id']}")
        check = run.timing.verify_pair(baseline, expected, original, events, policy,
                                       alignment_dt=int(backend.target.pulse_alignment or 1))
        row.update(isa_file=shard_name, isa_index=index, metrics=run.raw.circuits.metrics(program),
            final_logical_to_physical=layout, dd_check={k:v for k,v in check.items() if k != "windows"})
        run.base.validate_circuits({"configuration":configuration}, [program], [row], backend,
                                   expected_experiment=run.EXPERIMENT)
        run.validate_isa_circuits([program], backend.target)
        if (program.metadata or {}).get("dd_policy") != configuration["dd_policy"] or program.metadata.get("dd_condition") != "on":
            raise ValueError("legacy DD policy metadata changed")
        if configuration["capture_timing"]:
            windows.append(dict(pub_index=row["pub_index"], case_id=row["case_id"], **check))
        vector += run.instruction_audit.footprint(program)
    return windows, vector
