"""Reference-assisted schedule planning; no trajectory sampling occurs here."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from library import reservoir_scaling as physics


@dataclass(frozen=True)
class Settings:
    initial_dt: float
    minimum_dt: float
    check_interval: float
    steady_window: float
    max_time: float
    bias_tolerance: float
    relaxation_tolerance: float
    sampling_tolerance: float
    refine_factor: float = 1.0

    def payload(self):
        return asdict(self)


def settings_from_options(options):
    """Defaults scale physical times by the supplied J and reservoir rate."""
    supplied = lambda name, default: default if getattr(options, name) is None else getattr(options, name)
    scale = max(options.J, options.gamma)
    interval = supplied("check_interval", 5/options.J)
    settings = Settings(
        initial_dt=supplied("initial_dt", .5/scale),
        minimum_dt=supplied("minimum_dt", 1e-5/scale),
        check_interval=interval,
        steady_window=supplied("steady_window", 2*interval),
        max_time=supplied("max_time", 200/options.J),
        bias_tolerance=supplied("bias_tolerance", .3*options.steady_tolerance),
        relaxation_tolerance=supplied("relaxation_tolerance", .3*options.steady_tolerance),
        sampling_tolerance=supplied("sampling_tolerance", .4*options.steady_tolerance),
        refine_factor=supplied("refine_factor", 1.0))
    for name, value in settings.payload().items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive and finite")
    if settings.minimum_dt > settings.initial_dt:
        raise ValueError("minimum-dt must not exceed initial-dt")
    if settings.check_interval < settings.initial_dt:
        raise ValueError("check-interval must be at least initial-dt")
    if settings.steady_window < settings.check_interval:
        raise ValueError("steady-window must be at least check-interval")
    total = settings.bias_tolerance + settings.relaxation_tolerance + settings.sampling_tolerance
    if total > options.steady_tolerance*(1+1e-12):
        raise ValueError("bias, relaxation, and sampling budgets must sum to at most steady-tolerance")
    scaled = options.J*np.array([settings.initial_dt, settings.minimum_dt])
    if np.any(~np.isfinite(scaled)) or np.any(scaled <= 0):
        raise ValueError("J times the automatic timestep limits must be positive and finite")
    return settings


def distance(left, right, model):
    """Max population/current-over-J discrepancy over every preparation/site/bond."""
    return max(physics.observable_error(left, right, model.prefactor).values())


def _interval_nodes(start, end, dt, remaining):
    """Reach the physical checkpoint, or truncate at the remaining step limit."""
    quotient = (end-start)/dt
    full = remaining if not np.isfinite(quotient) else min(int(np.floor(quotient)), remaining)
    nodes = start + np.arange(1, full+1)*dt
    epsilon = 8*np.finfo(float).eps*max(end, dt)
    if full and abs(nodes[-1]-end) <= epsilon:
        nodes[-1] = end
    elif full < remaining:
        nodes = np.r_[nodes, end]
    if not len(nodes) or np.any(np.diff(np.r_[start, nodes]) <= 0):
        raise ValueError("automatic schedule cannot represent distinct positive time steps")
    return nodes


def plan(model, initial, J, settings, *, policy, order, formula, step_cap, progress=None):
    """Select a descending grid, then evolve the mean channel to plan a replay.

    A passing window checks EVERY integration node, not just checkpoints.
    Checkpoint remainders are evolved with their actual shorter timestep.
    The finite-window result is not a permanent-settling certificate.
    """
    target = physics.exact_stationary(model)
    candidate_rows, candidates, history = [], [], []
    dt = settings.initial_dt
    while True:
        row = {"dt": float(dt)}
        try:
            branch = physics.channel(model, J*dt, policy, order, formula)
            # Screen dilation resonances with effectively no endpoint loss.
            if min(1-branch.weights[0, 0], 1-branch.weights[-1, -1]) <= 1e-12:
                raise ValueError("negligible endpoint population damping at this timestep")
            stationary, checks = physics.mean_stationary(branch, model.n)
            bias = distance(stationary, target, model)
            if not np.isfinite(bias):
                raise ValueError("stationary bias is not finite")
            row.update(status="valid", bias=float(bias), checks=checks,
                       within_bias_budget=bool(bias <= settings.bias_tolerance))
            candidates.append((dt, branch, stationary, bias))
        except (ValueError, np.linalg.LinAlgError) as error:
            row.update(status="rejected", reason=str(error), within_bias_budget=False)
        candidate_rows.append(row)
        if progress is not None:
            progress({"event": "candidate", **row})
        if row["within_bias_budget"]:
            break
        if dt <= settings.minimum_dt:
            break
        dt = max(settings.minimum_dt, dt/2)

    result = {"method": "stationary_bias_halving_then_mean_channel_relaxation",
        "settings": settings.payload(), "candidates": candidate_rows,
        "physical_stationary_real_gauge": target.tolist(),
        "selected_final_dt": None, "checkpoints": history, "stages": [],
        "status": "minimum_dt_reached", "stop_reason": "no tested timestep passed the stationary bias budget",
        "window_start": None, "window_end": None,
        "interpretation": "reference-assisted local-observable tests at integration nodes; no infinite-time or simultaneous statistical guarantee"}
    state, nodes = np.array(initial, dtype=float, copy=True), [0.0]
    result["final_mean_covariances_real_gauge"] = state.tolist()
    if not candidate_rows[-1]["within_bias_budget"]:
        return np.asarray(nodes), result

    result["selected_final_dt"] = float(candidates[-1][0])
    active = 0
    dt, branch, stationary, bias = candidates[active]
    cache = {float(dt): branch}
    result["stages"].append({"start_time": 0.0, "dt": float(dt), "bias": float(bias)})
    held_since = None
    worst_relaxation = worst_physical = 0.0

    def measure():
        return distance(state, stationary, model), distance(state, target, model)

    def update_window(now, relaxation, physical):
        nonlocal held_since, worst_relaxation, worst_physical
        valid = (bias <= settings.bias_tolerance
                 and relaxation <= settings.relaxation_tolerance
                 and physical <= settings.bias_tolerance+settings.relaxation_tolerance+1e-12)
        if not valid:
            held_since = None
            worst_relaxation = worst_physical = 0.0
        else:
            if held_since is None:
                held_since = now
            worst_relaxation = max(worst_relaxation, relaxation)
            worst_physical = max(worst_physical, physical)

    update_window(0.0, *measure())
    result.update(status="max_time_reached", stop_reason="maximum physical evolution time reached before a passing window")
    checkpoint = 1
    while nodes[-1] < settings.max_time and len(nodes)-1 < step_cap:
        end = min(checkpoint*settings.check_interval, settings.max_time)
        generated = _interval_nodes(nodes[-1], end, dt, step_cap-(len(nodes)-1))
        for now in generated:
            actual_dt = float(now-nodes[-1])
            if actual_dt not in cache:
                cache[actual_dt] = physics.channel(model, J*actual_dt, policy, order, formula)
            state = physics.mean_step(state, cache[actual_dt])
            nodes.append(float(now))
            relaxation, physical = measure()
            update_window(float(now), relaxation, physical)
        now = nodes[-1]
        relaxation, physical = measure()
        entry = {"time": now, "step_count": len(nodes)-1, "dt": float(dt),
                 "bias": float(bias), "relaxation_distance": relaxation,
                 "physical_stationary_distance": physical, "held_since": held_since}
        duration_epsilon = 16*np.finfo(float).eps*max(1.0, now, settings.steady_window)
        if held_since is not None and now-held_since+duration_epsilon >= settings.steady_window:
            result.update(status="converged", stop_reason="bias and relaxation budgets held at every node for the required window",
                          window_start=float(max(held_since, now-settings.steady_window)), window_end=now,
                          passing_span_max_relaxation=worst_relaxation,
                          passing_span_max_physical_distance=worst_physical)
            entry["action"] = "stop_mean_converged"
        elif len(nodes)-1 >= step_cap:
            result.update(status="step_cap_reached", stop_reason="maximum number of integration steps reached before a passing window")
            entry["action"] = "stop_step_cap"
        elif now >= settings.max_time:
            entry["action"] = "stop_max_time"
        elif bias > settings.bias_tolerance and relaxation <= max(settings.relaxation_tolerance, settings.refine_factor*bias):
            active += 1
            dt, branch, stationary, bias = candidates[active]
            result["stages"].append({"start_time": now, "dt": float(dt), "bias": float(bias)})
            cache[float(dt)] = branch
            held_since = None
            worst_relaxation = worst_physical = 0.0
            update_window(now, *measure())
            entry.update(action="refine", next_dt=float(dt), next_bias=float(bias))
        else:
            entry["action"] = "observe_steady_window" if held_since is not None else "continue_relaxation"
        history.append(entry)
        if progress is not None:
            progress({"event": "checkpoint", **entry})
        if entry["action"].startswith("stop_"):
            break
        checkpoint += 1
    result.update(final_mean_covariances_real_gauge=state.tolist(), final_nominal_dt=float(dt),
                  final_channel_stationary_real_gauge=stationary.tolist(), final_channel_bias=float(bias),
                  final_relaxation_distance=distance(state, stationary, model),
                  final_physical_stationary_distance=distance(state, target, model))
    return np.asarray(nodes), result


def assess_sampling(payload, controller):
    """Assess realized error and descriptive 2-SE bands over the entire replay tail."""
    if controller["status"] != "converged":
        return {"status": "not_assessed", "reason": "mean controller did not reach a passing window"}
    times = np.asarray(payload["times"])
    start, end = controller["window_start"], controller["window_end"]
    mask = (times >= start-1e-10) & (times <= end+1e-10)
    settings = controller["settings"]
    J = payload["configuration"]["physics"]["J"]
    discrepancy = uncertainty = physical = 0.0
    for family, divisor in (("populations", 1.0), ("excitation_flows", J)):
        sampled = np.asarray(payload["observables"][family])[:, mask]/divisor
        mean = np.asarray(payload["mean_channel_observables"][family])[:, mask]/divisor
        se = np.asarray(payload["trajectory_standard_errors"][family])[:, mask]/divisor
        target = np.asarray(payload["stationary_observables"][family])/divisor
        discrepancy = max(discrepancy, float(np.abs(sampled-mean).max()))
        uncertainty = max(uncertainty, float((2*se).max()))
        physical = max(physical, float(np.abs(sampled-target).max()))
    estimable = payload["configuration"]["trajectories"] > 1
    budget = settings["sampling_tolerance"]
    reasons = []
    if not estimable:
        reasons.append("trajectory uncertainty is unestimable with R=1")
    if discrepancy > budget:
        reasons.append("realized sampling discrepancy exceeds its budget")
    if estimable and uncertainty > budget:
        reasons.append("descriptive two-standard-error bands exceed the sampling budget")
    if physical > payload["configuration"]["steady_tolerance"]:
        reasons.append("sampled stationary distance exceeds total tolerance")
    return {"status": "passed" if not reasons else "sampling_budget_not_met", "reasons": reasons,
        "window_start": start, "window_end": end, "recorded_nodes_checked": int(mask.sum()),
        "max_realized_sampling_distance": discrepancy, "max_two_standard_errors": uncertainty,
        "max_sampled_stationary_distance": physical, "uncertainty_estimable": estimable,
        "interpretation": "all saved replay nodes in the controller window; descriptive 2-SE check, not simultaneous confidence or permanent settling"}
