# This code is part of Qiskit-derived scheduling utilities.
#
# (C) Copyright IBM 2022-2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of the original Qiskit IBM Runtime source tree or at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Modified for Model2 Experiment8: compute flat-DAG reachability once using
# bitsets, preserving block_order_op_nodes ordering. Control flow falls back
# to the installed SDK implementation. This avoids repeated descendants walks.
"""The SDK's flat reset/measurement block order with cached reachability."""
from qiskit.circuit import ControlFlowOp, Measure, Reset
from qiskit.dagcircuit import DAGOpNode
from qiskit_ibm_runtime.transpiler.passes.scheduling.utils import block_order_op_nodes


def flat_block_order(dag):
    nodes = list(dag.topological_op_nodes())
    if any(isinstance(node.op, ControlFlowOp) for node in nodes):
        yield from block_order_op_nodes(dag)
        return
    positions = {node:i for i,node in enumerate(nodes)}
    descendants = [0]*len(nodes)
    successors = [0]*len(nodes)
    grouped = [isinstance(node.op, (Measure, Reset)) for node in nodes]
    measurement_bits = sum(1<<i for i,is_grouped in enumerate(grouped) if is_grouped)
    for i in range(len(nodes)-1, -1, -1):
        reachable = 0
        for successor in dag.successors(nodes[i]):
            if isinstance(successor, DAGOpNode):
                j = positions[successor]
                successors[i] |= 1<<j
                reachable |= (1<<j) | descendants[j]
        descendants[i] = reachable
    current = list(range(len(nodes)))
    while current:
        blocked, emit_extra, later, pending, measures = 0, 0, [], [], []
        for i in current:
            if blocked & (1<<i):
                later.append(i)
            elif grouped[i]:
                blocked |= descendants[i]
                emit_extra |= successors[i]
                measures.append(i)
            else:
                pending.append(i)
        tail = []
        for i in pending:
            # is_predecessor(node, other) tests an immediate incoming edge.
            # SDK _emit accepts grouped descendants or immediate successors
            # of the current measure/reset group, rather than all descendants.
            if descendants[i] & (measurement_bits | emit_extra):
                yield nodes[i]
            else:
                tail.append(i)
        for i in measures:
            yield nodes[i]
        if not measures:
            for i in tail:
                yield nodes[i]
            break
        current = tail+later
