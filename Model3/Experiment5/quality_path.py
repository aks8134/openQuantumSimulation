"""Select or validate a CZ path using an already loaded backend target."""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np


def calibration(backend, avoid, min_t1, min_t2, max_cz_error, max_readout_error=.02):
    if min_t1 <= 0 or min_t2 <= 0 or not 0 < max_cz_error < 1 or not 0 < max_readout_error < 1:
        raise ValueError("invalid layout calibration thresholds")
    target = backend.target
    if "cz" not in target.operation_names or "measure" not in target.operation_names:
        raise ValueError("CZ and measurement calibration are required")
    qubit_properties = target.qubit_properties
    if qubit_properties is None:
        raise ValueError("qubit coherence calibration unavailable")
    nodes = {}
    for q, prop in enumerate(qubit_properties):
        readout = target["measure"].get((q,))
        values = [getattr(prop, name, None) for name in ("t1", "t2")]+[getattr(readout, "error", None)]
        if q in avoid or any(v is None or not math.isfinite(v) for v in values):
            continue
        t1, t2, error = values
        if t1 >= min_t1 and t2 >= min_t2 and 0 <= error <= max_readout_error:
            nodes[q] = {"t1_seconds": t1, "t2_seconds": t2, "readout_error": error}
    # CZ is symmetric; use the worst reported orientation of an edge.
    edges = {}
    for pair, prop in target["cz"].items():
        if pair is None or len(pair) != 2 or not set(pair) <= nodes.keys():
            continue
        error = getattr(prop, "error", None)
        if error is None or not math.isfinite(error) or not 0 <= error < 1:
            continue
        pair = tuple(sorted(pair))
        edges[pair] = max(edges.get(pair, 0.), error)
    edges = {pair: error for pair, error in edges.items() if error <= max_cz_error}
    return nodes, edges


def gate_weights(source, indices):
    steps = len(source["schedule"]["step_dts"])
    n = source["configuration"]["physics"]["n_qubits"]
    choices = np.asarray(source["schedule"]["jump_choices"])[indices]
    left = 2*float(np.mean(np.sum(choices < 2, axis=1)))
    right = 2*steps-left
    return [left]+[2.*steps]*(n-1)+[right]


def score(path, nodes, edges, weights):
    errors = [edges[tuple(sorted(pair))] for pair in zip(path, path[1:])]
    readout = [nodes[q]["readout_error"] for q in path[1:-1]]
    t1 = min(nodes[q]["t1_seconds"] for q in path)
    t2 = min(nodes[q]["t2_seconds"] for q in path)
    weighted = sum(weight*error for weight, error in zip(weights, errors))
    ranking = (max(errors), weighted, max(readout), sum(readout), -t2, -t1, tuple(path))
    return ranking, {"worst_cz_error": max(errors), "gate_use_weighted_cz_error_sum": weighted,
                     "worst_system_readout_error": max(readout), "system_readout_error_sum": sum(readout),
                     "minimum_t1_seconds": t1, "minimum_t2_seconds": t2}


def select(backend, source, indices, *, avoid=(7,), min_t1=80e-6, min_t2=60e-6,
           max_cz_error=.005, max_readout_error=.02, max_expansions=2_000_000):
    if max_expansions < 1:
        raise ValueError("max-expansions must be positive")
    nodes, edges = calibration(backend, set(avoid), min_t1, min_t2, max_cz_error, max_readout_error)
    size = source["configuration"]["physics"]["n_qubits"]+2
    adjacency = {q: set() for q in nodes}
    for a, b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    weights = gate_weights(source, indices)
    best, candidates, expansions, limited = None, 0, 0, False

    def visit(path):
        nonlocal best, candidates, expansions, limited
        if expansions >= max_expansions:
            limited = True
            return
        expansions += 1
        if len(path) == size:
            candidates += 1
            ranking, metrics = score(path, nodes, edges, weights)
            if best is None or ranking < best[0]:
                best = (ranking, list(path), metrics)
            return
        for q in sorted(adjacency[path[-1]]-set(path)):
            visit(path+[q])
            if limited:
                return
    for first in sorted(nodes):
        visit([first])
        if limited:
            break
    if best is None:
        raise ValueError(f"no qualifying {size}-qubit path found ({len(nodes)} eligible qubits, {len(edges)} edges, "
                         f"{expansions} search expansions); thresholds were not relaxed")
    settings = {"avoid_qubits": list(avoid), "min_t1_seconds": min_t1, "min_t2_seconds": min_t2,
                "max_cz_error": max_cz_error, "max_readout_error": max_readout_error, "max_expansions": max_expansions}
    return {"mode": "calibration_search", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend.name, "physical_path": best[1], "metrics": best[2], "settings": settings,
            "edge_gate_weights": weights, "candidate_paths_examined": candidates, "search_expansions": expansions,
            "search_complete": not limited, "selected_qubits": {str(q): nodes[q] for q in best[1]},
            "ranking": "worst CZ, gate-use-weighted CZ sum, worst/sum system readout, minimum T2/T1, path order",
            "interpretation": "Heuristic ranking after hard coherence/CZ filters; not an observable error estimate or accuracy certificate."}


def validate(backend, path, source, indices, settings):
    nodes, edges = calibration(backend, set(settings["avoid_qubits"]), settings["min_t1_seconds"],
                               settings["min_t2_seconds"], settings["max_cz_error"], settings["max_readout_error"])
    size = source["configuration"]["physics"]["n_qubits"]+2
    if len(path) != size or len(set(path)) != size or not set(path) <= nodes.keys():
        raise ValueError("saved control path no longer meets coherence/avoidance criteria; run fresh controls")
    if any(tuple(sorted(pair)) not in edges for pair in zip(path, path[1:])):
        raise ValueError("saved control path no longer meets CZ criteria; run fresh controls")
    _, metrics = score(path, nodes, edges, gate_weights(source, indices))
    return {"mode": "reuse_control_path", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "physical_path": list(path), "settings": settings, "metrics": metrics,
            "selected_qubits": {str(q): nodes[q] for q in path}}
