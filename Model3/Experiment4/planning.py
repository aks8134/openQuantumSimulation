"""Deterministic budget-constrained preparation/schedule search for Model3 XX."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from library import reservoir_scaling as physics

PREPARATIONS = ("vacuum", "filled", "left_excitation", "neel", "half_mixed", "profile_mixed")


def preparations(model, target, names):
    """Cheap diagonal product states, never a free correlated target state."""
    values = {
        "vacuum": np.zeros(model.n), "filled": np.ones(model.n),
        "left_excitation": np.r_[1., np.zeros(model.n-1)],
        "neel": (np.arange(model.n) % 2 == 0).astype(float),
        "half_mixed": np.full(model.n, .5),
        "profile_mixed": np.clip(np.diag(target), 0, 1),
    }
    return [{"name": name, "populations": values[name].tolist(),
             "local_preparation_depth": int(np.any(values[name] > 0)),
             "expected_X_gates_per_shot": float(values[name].sum()),
             "classical_bit_randomization": bool(np.any((values[name] > 0) & (values[name] < 1))),
             "reference_assisted_populations": name == "profile_mixed",
             "construction": "independent local bits with supplied occupation probabilities; all initial coherences/current zero"}
            for name in names]


def distance(left, right, model):
    return max(physics.observable_error(left, right, model.prefactor).values())


@dataclass
class Node:
    covariance: np.ndarray
    indices: tuple[int, ...]
    times: tuple[float, ...]
    errors: tuple[float, ...]
    stages: int
    stage_start: int = 0


def advance(node, index, count, dt, branch, target, model):
    state, times, errors = node.covariance, list(node.times), list(node.errors)
    for _ in range(count):
        state = physics.mean_step(state, branch)
        now = times[-1]+dt
        if not np.isfinite(now) or now <= times[-1]:
            raise ValueError("step grid cannot represent increasing finite physical times")
        times.append(float(now))
        errors.append(distance(state, target, model))
    changed = not node.indices or node.indices[-1] != index
    return Node(state, node.indices+(index,)*count, tuple(times), tuple(errors),
                node.stages+int(changed), len(node.indices) if changed else node.stage_start)


def holding_steps(dt, window):
    """Whole constant-dt steps covering the requested validation interval."""
    ratio = window/dt
    return max(1, int(np.ceil(ratio-32*np.finfo(float).eps*max(1., ratio))))


def load_baselines(options):
    """Load compatible prior studies; do not silently drop a prior winner."""
    rows, manifest = [], []
    physics_keys = ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor",
                    "sampling_policy", "order", "system_formula")
    for filename in getattr(options, "baseline_study", []):
        path = Path(filename)
        raw = path.read_bytes()
        study = json.loads(raw)
        if study.get("experiment") != "Model3/Experiment4":
            raise ValueError("baseline-study must be an Experiment4 study.json")
        config = study["configuration"]
        if any(config[key] != getattr(options, key) for key in physics_keys):
            raise ValueError(f"baseline-study physics/channel mismatch: {path}")
        retained = 0
        for row in study.get("best_per_preparation", [study["selected"]]):
            if row["preparation"] not in options.preparations:
                continue
            stages = row["stages"]
            if any(stage["dt"] not in options.step_grid for stage in stages):
                raise ValueError(f"baseline-study timesteps must be included in step-grid: {path}")
            if (any(not isinstance(s["steps"], int) or s["steps"] < 1 for s in stages)
                    or any(a["dt"] < b["dt"] for a, b in zip(stages, stages[1:]))
                    or len(stages) > options.max_stages):
                raise ValueError(f"invalid or unsupported baseline stages: {path}")
            rows.append({"preparation": row["preparation"], "stages": stages, "source": str(path.resolve())})
            retained += 1
        if not retained:
            raise ValueError(f"baseline-study has no requested preparations: {path}")
        manifest.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    return rows, manifest


def evaluate(node, preparation, options):
    if getattr(options, "validation_mode", "trailing") == "forward":
        end = len(node.indices)
        hold = holding_steps(options.step_grid[node.indices[-1]], options.steady_window) if end else 0
        first = end-hold
        # Only a constant final-step continuation can validate an output state.
        # Future refinement must never be treated as free validation evolution.
        covered = bool(end and (first > node.stage_start or (first == 0 and node.stage_start == 0)))
        output = first if covered else end
        score = max(node.errors[first:]) if covered else None
        depth = (preparation["local_preparation_depth"]+options.estimated_step_depth*output
                 if options.estimated_step_depth is not None else None)
        return {"steps": output, "final_time": node.times[output],
            "validation_mode": "forward", "tail_window_covered": covered,
            "window_start": node.times[output], "window_end": node.times[-1],
            "checked_from_step": output, "checked_from_time": node.times[output],
            "tail_mean_error": score, "final_mean_error": node.errors[output],
            "mean_target_met": bool(covered and score <= options.mean_tolerance),
            "estimated_total_depth": depth, "classical_validation_steps": hold if covered else 0,
            "total_planning_path_steps": end}
    end = node.times[-1]
    epsilon = 32*np.finfo(float).eps*max(1., end, options.steady_window)
    covered = bool(end+epsilon >= options.steady_window)
    start = max(0., end-options.steady_window)
    # Include the preceding node conservatively if the boundary is between nodes.
    first = max(0, bisect_right(node.times, start)-1)
    score = max(node.errors[first:]) if covered else None
    steps = len(node.indices)
    depth = (preparation["local_preparation_depth"]+options.estimated_step_depth*steps
             if options.estimated_step_depth is not None else None)
    return {"steps": steps, "final_time": end, "tail_window_covered": covered,
        "window_start": start, "checked_from_step": first, "checked_from_time": node.times[first],
        "tail_mean_error": score, "final_mean_error": node.errors[-1],
        "mean_target_met": bool(covered and score <= options.mean_tolerance),
        "estimated_total_depth": depth}


def rank(node, preparation, options):
    result = evaluate(node, preparation, options)
    return rank_result(result, preparation, options)


def rank_result(result, preparation, options):
    cost = result["estimated_total_depth"] if options.estimated_step_depth is not None else result["steps"]
    if result["mean_target_met"]:
        return (0, cost, result["tail_mean_error"], preparation["local_preparation_depth"])
    if result["tail_window_covered"]:
        return (1, result["tail_mean_error"], cost, result["final_mean_error"])
    return (2, result["final_mean_error"], cost, preparation["local_preparation_depth"])


def describe(node, preparation, options):
    result = {"preparation": preparation["name"], **evaluate(node, preparation, options)}
    stages = []
    for index in node.indices[:result["steps"]]:
        dt = options.step_grid[index]
        if stages and stages[-1]["dt"] == dt:
            stages[-1]["steps"] += 1
        else:
            stages.append({"dt": dt, "steps": 1})
    result["stages"] = stages
    return result


def search(options, progress=None, baselines=None):
    """Constants plus a bounded beam search; no randomness or exact transients.

    Keep a separate beam for each final step size and preparation, so coarse
    burn-in paths are retained even when their current endpoint error is high.
    Final ranking always uses the sustained-window objective, not this heuristic.
    """
    target = physics.exact_stationary(options.model)
    if not np.all(np.isfinite(target)):
        raise ValueError("physical stationary covariance is not finite")
    preps = preparations(options.model, target, options.preparations)
    with np.errstate(over="ignore", invalid="ignore"):
        branches = [physics.channel(options.model, options.J*dt, options.sampling_policy,
                                    options.order, options.system_formula) for dt in options.step_grid]
    if any(not all(np.all(np.isfinite(value)) for value in branch[:-1]) for branch in branches):
        raise ValueError("step-grid produces nonfinite channel coefficients")
    limits = [options.step_budget if options.depth_budget is None else
              min(options.step_budget, max(0, (options.depth_budget-prep["local_preparation_depth"])//options.estimated_step_depth))
              for prep in preps]
    roots = [Node(np.diag(prep["populations"]), (), (0.,),
                  (distance(np.diag(prep["populations"]), target, options.model),), 0) for prep in preps]
    best = roots.copy()
    best_results = [evaluate(node, prep, options) for node, prep in zip(best, preps)]
    forward = getattr(options, "validation_mode", "trailing") == "forward"
    hold_counts = [holding_steps(dt, options.steady_window) if forward else 0 for dt in options.step_grid]

    def output_limit(i):
        result = best_results[i]
        return min(limits[i], result["steps"]) if result["mean_target_met"] else limits[i]

    def consider(i, node):
        result = evaluate(node, preps[i], options)
        if result["steps"] <= limits[i] and rank_result(result, preps[i], options) < rank_result(best_results[i], preps[i], options):
            best[i] = node
            best_results[i] = result

    seeds = load_baselines(options)[0] if baselines is None else baselines
    incumbent_rows = []
    incumbent_mean_steps = 0
    for seed in seeds:
        i = options.preparations.index(seed["preparation"])
        node = roots[i]
        for stage in seed["stages"]:
            index = options.step_grid.index(stage["dt"])
            for _ in range(stage["steps"]):
                node = advance(node, index, 1, stage["dt"], branches[index], target, options.model)
                consider(i, node)
                incumbent_mean_steps += 1
        # Evaluate output prefixes near the end of a saved preparation too.
        if forward and node.indices:
            index = node.indices[-1]
            for _ in range(hold_counts[index]):
                if len(node.indices) >= limits[i]+hold_counts[index]:
                    break
                node = advance(node, index, 1, options.step_grid[index], branches[index], target, options.model)
                consider(i, node)
                incumbent_mean_steps += 1
        incumbent_rows.append({"source": seed["source"], "input_stages": seed["stages"],
                               "best_after_inclusion": describe(best[i], preps[i], options)})

    constants = []
    baseline_mean_steps = 0
    for index, (dt, branch) in enumerate(zip(options.step_grid, branches)):
        for i, prep in enumerate(preps):
            node, constant_best = roots[i], roots[i]
            for _ in range(limits[i]+hold_counts[index]):
                node = advance(node, index, 1, dt, branch, target, options.model)
                if evaluate(node, prep, options)["steps"] <= limits[i] and rank(node, prep, options) < rank(constant_best, prep, options):
                    constant_best = node
                consider(i, node)
                baseline_mean_steps += 1
            constants.append({"dt": dt, **describe(constant_best, prep, options)})
        if progress is not None:
            progress({"phase": "baselines", "grid_index": index+1, "grid_size": len(branches)})

    beams = [[root] for root in roots]
    expansions, layers, truncated = 0, 0, False
    while options.search_node_cap:
        actions = []
        for i, beam in enumerate(beams):
            for node in beam:
                for index in range(node.indices[-1] if node.indices else 0, len(branches)):
                    stages = node.stages+int(not node.indices or node.indices[-1] != index)
                    same = bool(node.indices and node.indices[-1] == index)
                    if (stages > options.max_stages or len(node.indices) >= output_limit(i)+hold_counts[index]
                            or (forward and not same and len(node.indices) > output_limit(i))):
                        continue
                    actions.append((i, node, index))
        if not actions:
            break
        # End before an incomplete layer, giving all preparations equal search depth.
        if expansions+len(actions) > options.search_node_cap:
            truncated = True
            break
        buckets = [{} for _ in preps]
        for i, node, index in actions:
            count = min(options.block_steps, output_limit(i)+hold_counts[index]-len(node.indices))
            if count <= 0:
                continue
            child = advance(node, index, count,
                            options.step_grid[index], branches[index], target, options.model)
            consider(i, child)
            buckets[i].setdefault(index, []).append(child)
        beams = []
        for prep_buckets in buckets:
            beam = []
            for nodes in prep_buckets.values():
                nodes.sort(key=lambda node: (node.errors[-1], node.stages, node.indices))
                beam.extend(nodes[:options.beam_width])
            beams.append(beam)
        expansions += len(actions)
        layers += 1
        if progress is not None:
            progress({"phase": "beam", "layer": layers, "expansions": expansions})

    winner = min(range(len(preps)), key=lambda i: rank(best[i], preps[i], options))
    selected = describe(best[winner], preps[winner], options)
    node = best[winner]
    output = selected["steps"]
    selected["step_dts"] = [options.step_grid[index] for index in node.indices[:output]]
    selected["time_nodes"] = list(node.times[:output+1])
    if forward:
        state = roots[winner].covariance
        for index in node.indices[:output]:
            state = physics.mean_step(state, branches[index])
        selected["final_mean_covariance_real_gauge"] = state.tolist()
        selected["forward_validation"] = {
            "step_dts": [options.step_grid[index] for index in node.indices[output:]],
            "times": list(node.times[output:]), "mean_errors": list(node.errors[output:]),
            "interpretation": "classical constant-final-step continuation; sampling checked only at output"}
    else:
        selected["final_mean_covariance_real_gauge"] = node.covariance.tolist()
    return {"method": "constant baselines plus monotone-stage beam search",
        "scope": "best tested candidate only; not a global optimum or compiled circuit-depth result",
        "objective": "fewest steps (or modeled depth) among mean-window passing candidates; otherwise lowest covered-window error within budget",
        "mean_tolerance": options.mean_tolerance, "steady_window": options.steady_window,
        "physical_stationary_real_gauge": target.tolist(), "preparations": preps,
        "preparation_step_limits": dict(zip(options.preparations, limits)),
        "best_per_preparation": [describe(node, prep, options) for node, prep in zip(best, preps)],
        "constant_baselines": constants, "incumbent_baselines": incumbent_rows, "selected": selected,
        "selected_preparation": preps[winner], "beam_expansions": expansions,
        "beam_layers": layers, "search_limit_reached": truncated,
        "baseline_mean_steps": baseline_mean_steps, "incumbent_mean_steps": incumbent_mean_steps,
        "validation_mode": getattr(options, "validation_mode", "trailing"),
        "status": "mean_target_met" if selected["mean_target_met"] else
                  ("target_not_met_within_budget" if selected["tail_window_covered"] else "tail_window_not_covered")}
