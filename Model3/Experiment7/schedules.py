"""Automatic two-stage feedback and cosine scheduling for the XX mean channel.

Search evaluations are classical predictions. Only committed steps enter the
preparation circuit. Neither planner reads a previous experiment's schedules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import differential_evolution

from library import reservoir_scaling as physics
from Model3.Experiment4 import planning


@dataclass(frozen=True)
class Settings:
    minimum_dt: float
    maximum_dt: float
    mean_tolerance: float
    steady_window: float
    step_budget: int
    maximum_time: float
    decision_steps: int = 4
    optimizer_iterations: int = 80
    feedback_iterations: int = 16
    population_multiplier: int = 10
    optimizer_seed: int = 1729
    forecast_nodes: int = 64
    validation_step_cap: int = 100000
    forecast_margin: float = .99

    def payload(self):
        return asdict(self)


def validate_settings(settings):
    for name in ("minimum_dt", "maximum_dt", "mean_tolerance", "steady_window", "maximum_time"):
        value = getattr(settings, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if settings.minimum_dt > settings.maximum_dt:
        raise ValueError("minimum-dt must not exceed maximum-dt")
    for name in ("step_budget", "decision_steps", "optimizer_iterations", "feedback_iterations",
                 "population_multiplier", "forecast_nodes", "validation_step_cap"):
        if not isinstance(getattr(settings, name), int) or getattr(settings, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(settings.optimizer_seed, int) or settings.optimizer_seed < 0:
        raise ValueError("optimizer-seed must be a nonnegative integer")
    if not np.isfinite(settings.forecast_margin) or not 0 < settings.forecast_margin <= 1:
        raise ValueError("forecast-margin must be in (0,1]")
    if planning.holding_steps(settings.minimum_dt, settings.steady_window) > settings.validation_step_cap:
        raise ValueError("minimum-dt needs more continuation steps than validation-step-cap")


def decode(family, parameters):
    """A monotone two-stage plan or a single cosine decay with a positive floor."""
    steps = max(1, int(round(parameters[0])))
    fine, coarse = sorted(np.exp(parameters[1:3]))
    if family == "cosine":
        if steps == 1:
            return np.array([fine]), {"steps": steps, "initial_dt": float(fine), "final_dt": float(fine)}
        dts = fine + .5*(coarse-fine)*(1+np.cos(np.pi*np.arange(steps)/(steps-1)))
        # Set endpoints exactly, including the continuation step.
        dts[0], dts[-1] = coarse, fine
        return dts, {"steps": steps, "initial_dt": float(coarse), "final_dt": float(fine)}
    if family != "two_stage":
        raise ValueError(f"unknown schedule family: {family}")
    split = min(steps-1, max(0, int(round(parameters[3]*(steps-1)))))
    dts = np.r_[np.full(split, coarse), np.full(steps-split, fine)]
    return dts, {"steps": steps, "coarse_steps": split, "fine_steps": steps-split,
                 "coarse_dt": float(coarse), "fine_dt": float(fine)}


class Predictor:
    def __init__(self, model, J, settings, policy="rate-weighted", order="hamiltonian-first", formula="even-odd"):
        self.model, self.J, self.settings = model, J, settings
        self.policy, self.order, self.formula = policy, order, formula
        self.target = physics.exact_stationary(model)
        self.indices = np.triu_indices(model.n)
        self.dimension = len(self.indices[0])
        self.basis = np.zeros((self.dimension, model.n, model.n))
        for i, (row, column) in enumerate(zip(*self.indices)):
            self.basis[i, row, column] = self.basis[i, column, row] = 1.
        self.observable_indices = np.flatnonzero(self.indices[1]-self.indices[0] <= 1)
        self.observable_factors = np.where(self.indices[0][self.observable_indices] == self.indices[1][self.observable_indices],
                                           1., 4*model.prefactor)
        self.target_packed = self.pack(self.target)
        self.prediction_mean_steps = 0
        self.validation_mean_steps = 0
        self.objective_evaluations = 0

    def pack(self, state):
        return np.asarray(state)[..., self.indices[0], self.indices[1]]

    def unpack(self, vector):
        state = np.zeros((self.model.n, self.model.n))
        state[self.indices] = vector[:self.dimension]
        state[(self.indices[1], self.indices[0])] = vector[:self.dimension]
        return state

    def packed_error(self, vector):
        delta = vector[:self.dimension]-self.target_packed
        return float(np.max(np.abs(delta[self.observable_indices])*self.observable_factors))

    @lru_cache(maxsize=2048)
    def channel(self, dt):
        return physics.channel(self.model, self.J*dt, self.policy, self.order, self.formula)

    def affine_matrix(self, branch):
        """Exact mean map on symmetric covariances, plus a homogeneous coordinate."""
        result = np.eye(self.dimension+1)
        result[:-1, :-1] = self.pack(physics.mean_step(self.basis, branch, affine=False)).T
        result[:-1, -1] = self.pack(physics.mean_step(np.zeros_like(self.target), branch))
        return result

    def constant(self, state, dt, count):
        branch = self.channel(float(dt))
        self.prediction_mean_steps += count
        if count <= 32:
            for _ in range(count):
                state = physics.mean_step(state, branch)
            return state
        vector = np.r_[self.pack(state), 1.]
        return self.unpack(np.linalg.matrix_power(self.affine_matrix(branch), count) @ vector)

    def forecast(self, state, dts, family):
        if family == "two_stage":
            split = int(np.count_nonzero(dts != dts[-1]))
            if split:
                state = self.constant(state, dts[0], split)
            return self.constant(state, dts[-1], len(dts)-split)
        for dt in dts:
            state = physics.mean_step(state, self.channel(float(dt)))
        self.prediction_mean_steps += len(dts)
        return state

    def window_proxy(self, state, dt):
        """Exact block advancement, with sparse future observations for search only.

        Every candidate retained for execution must separately pass all nodes.
        """
        count = planning.holding_steps(dt, self.settings.steady_window)
        stride = max(1, int(np.ceil(count/self.settings.forecast_nodes)))
        branch = self.channel(float(dt))
        maximum = planning.distance(state, self.target, self.model)
        if stride == 1:
            for _ in range(count):
                state = physics.mean_step(state, branch)
                maximum = max(maximum, planning.distance(state, self.target, self.model))
        else:
            matrix = self.affine_matrix(branch)
            block = np.linalg.matrix_power(matrix, stride)
            vector = np.r_[self.pack(state), 1.]
            for _ in range(count//stride):
                vector = block @ vector
                maximum = max(maximum, self.packed_error(vector))
            if count % stride:
                vector = np.linalg.matrix_power(matrix, count % stride) @ vector
                maximum = max(maximum, self.packed_error(vector))
        self.prediction_mean_steps += count
        return maximum

    def validate(self, state, dt, time=0.):
        """Independent direct stepping at EVERY node of the same-final-dt window."""
        count = planning.holding_steps(dt, self.settings.steady_window)
        errors = [planning.distance(state, self.target, self.model)]
        times = [float(time)]
        passed = count <= self.settings.validation_step_cap and errors[0] <= self.settings.mean_tolerance
        branch = self.channel(float(dt))
        for step in range(count if passed else 0):
            state = physics.mean_step(state, branch)
            self.validation_mean_steps += 1
            errors.append(planning.distance(state, self.target, self.model))
            times.append(float(time+(step+1)*dt))
            if errors[-1] > self.settings.mean_tolerance:
                passed = False
                break
        return bool(passed), {"times": times, "mean_errors": errors,
            "step_dts": [float(dt)]*(len(times)-1),
            "interpretation": "classical constant-final-step continuation; sampling checked only at output"}

    def optimize(self, state, family, remaining_steps, remaining_time, *, decision=0, incumbent=None):
        """Joint automatic optimization of length, timestep values and switch point.

        A passing incumbent is retained, so replanning cannot discard a known
        feasible continuation. The initial call has no incumbent or prior data.
        """
        settings = self.settings
        cap = min(remaining_steps, len(incumbent) if incumbent is not None else remaining_steps)
        best = None
        improvements = []
        if incumbent is not None:
            output = self.forecast(state.copy(), incumbent, "two_stage")
            passed, validation = self.validate(output, float(incumbent[-1]))
            if not passed:
                raise ValueError("committed incumbent continuation lost its independent validation")
            best = {"dts": np.asarray(incumbent).copy(), "parameters": None,
                    "proxy_error": max(validation["mean_errors"]), "validation": validation,
                    "origin": "retained_own_forecast"}

        def objective(parameters):
            nonlocal best
            self.objective_evaluations += 1
            dts, description = decode(family, parameters)
            duration = float(dts.sum())
            if duration > remaining_time:
                return 10*settings.step_budget+duration/remaining_time
            output = self.forecast(state.copy(), dts, family)
            error = self.window_proxy(output, float(dts[-1]))
            # Feasibility first, then preparation steps. A 1% search margin is
            # a heuristic safeguard; actual acceptance uses the original budget.
            proxy_tolerance = settings.forecast_margin*settings.mean_tolerance
            if error <= settings.mean_tolerance and (best is None or len(dts) < len(best["dts"]) or
                    (len(dts) == len(best["dts"]) and error < best["proxy_error"])):
                passed, validation = self.validate(output, float(dts[-1]))
                if passed:
                    best = {"dts": dts.copy(), "parameters": description, "proxy_error": error,
                            "validation": validation, "origin": "optimized_forecast"}
                    improvements.append({**description, "proxy_error": error,
                                         "all_node_error": max(validation["mean_errors"])})
            if error <= proxy_tolerance:
                return len(dts)+.1*error/proxy_tolerance
            return settings.step_budget+1+settings.step_budget*max(0., error/proxy_tolerance-1)

        bounds = [(1, cap), (np.log(settings.minimum_dt), np.log(settings.maximum_dt)),
                  (np.log(settings.minimum_dt), np.log(settings.maximum_dt))]
        if family == "two_stage":
            bounds.append((0., 1.))
        seed = settings.optimizer_seed+decision
        rng = np.random.default_rng(seed)
        population = rng.uniform(size=(max(5, settings.population_multiplier*len(bounds)), len(bounds)))
        for index, (low, high) in enumerate(bounds):
            population[:, index] = low+(high-low)*population[:, index]
        if incumbent is not None:
            split = int(np.count_nonzero(incumbent != incumbent[-1]))
            population[0] = [len(incumbent), np.log(incumbent[0]), np.log(incumbent[-1]),
                             split/max(1, len(incumbent)-1)]
        before = self.objective_evaluations
        optimizer = differential_evolution(objective, bounds, init=population,
            integrality=[True]+[False]*(len(bounds)-1), rng=rng, polish=False,
            maxiter=settings.optimizer_iterations if incumbent is None else settings.feedback_iterations,
            tol=.001, atol=.001, updating="immediate")
        audit = {"family": family, "optimizer_seed": seed, "remaining_step_cap": cap,
            "evaluations": self.objective_evaluations-before, "iterations": int(optimizer.nit),
            "optimizer_converged": bool(optimizer.success), "best_objective": float(optimizer.fun),
            "passing_improvements": improvements, "selected_parameters": best["parameters"] if best else None,
            "selected_origin": best["origin"] if best else None,
            "predicted_preparation_steps": len(best["dts"]) if best else None}
        return best, audit


def plan(model, J, preparation, settings, *, method="two_stage", policy="rate-weighted",
         order="hamiltonian-first", formula="even-odd", progress=None):
    validate_settings(settings)
    if method not in ("two_stage", "cosine"):
        raise ValueError("method must be two_stage or cosine")
    predictor = Predictor(model, J, settings, policy, order, formula)
    state = np.diag(preparation["populations"])
    times, dts, decisions = [0.], [], []
    errors = [planning.distance(state, predictor.target, model)]
    passed, status, incumbent = False, "no_feasible_forecast", None
    validation = {"times": [0.], "mean_errors": errors.copy(), "step_dts": [],
                  "interpretation": "no passing forward window"}
    if errors[0] <= settings.mean_tolerance:
        passed, validation = predictor.validate(state.copy(), settings.minimum_dt)
        if passed:
            status = "mean_target_met"
    while not passed and len(dts) < settings.step_budget:
        if settings.maximum_time-times[-1] < settings.minimum_dt:
            status = "maximum_time_reached"
            break
        best, audit = predictor.optimize(state, method, settings.step_budget-len(dts),
            settings.maximum_time-times[-1], decision=len(decisions), incumbent=incumbent)
        audit.update(step=len(dts), time=times[-1], current_error=errors[-1])
        decisions.append(audit)
        if best is None:
            break
        proposed = best["dts"]
        count = len(proposed) if method == "cosine" else min(settings.decision_steps, len(proposed))
        for dt in proposed[:count]:
            dt = float(dt)
            state = physics.mean_step(state, predictor.channel(dt))
            dts.append(dt)
            times.append(float(times[-1]+dt))
            errors.append(planning.distance(state, predictor.target, model))
            if errors[-1] <= settings.mean_tolerance:
                passed, validation = predictor.validate(state.copy(), dt, times[-1])
                if passed:
                    status = "mean_target_met"
                    break
        audit.update(committed_steps=len(dts)-audit["step"], resulting_error=errors[-1])
        if progress is not None:
            progress({"method": method, "step": len(dts), "time": times[-1],
                "dt": dts[-1], "error": errors[-1], "passing_forecast_steps": len(proposed), "passed": passed})
        if passed:
            break
        incumbent = proposed[count:]
        if not len(incumbent):
            raise ValueError("a fully committed passing forecast failed direct output validation")
    if not passed:
        validation = {"times": [times[-1]], "mean_errors": [errors[-1]], "step_dts": [],
                      "interpretation": "no passing forward window at output"}
    stages = []
    for dt in dts:
        if stages and stages[-1]["dt"] == dt:
            stages[-1]["steps"] += 1
        else:
            stages.append({"dt": dt, "steps": 1})
    selected = {"preparation": preparation["name"], "steps": len(dts), "final_time": times[-1],
        "step_dts": dts, "time_nodes": times, "stages": stages, "validation_mode": "forward",
        "tail_window_covered": passed, "mean_target_met": passed, "checked_from_step": len(dts),
        "checked_from_time": times[-1], "window_start": times[-1], "window_end": validation["times"][-1],
        "tail_mean_error": max(validation["mean_errors"]), "final_mean_error": errors[-1],
        "estimated_total_depth": None, "classical_validation_steps": len(validation["step_dts"]),
        "final_mean_covariance_real_gauge": state.tolist(), "forward_validation": validation}
    return {"method": method, "status": status, "settings": settings.payload(), "selected": selected,
        "selected_preparation": preparation, "physical_stationary_real_gauge": predictor.target.tolist(),
        "decisions": decisions, "preparation_mean_errors": errors,
        "objective_evaluations": predictor.objective_evaluations,
        "prediction_mean_steps": predictor.prediction_mean_steps,
        "validation_mean_steps": predictor.validation_mean_steps,
        "interpretation": "reference-assisted heuristic schedule optimization; all-node finite-window validation, not a global minimum or permanent settling proof"}
