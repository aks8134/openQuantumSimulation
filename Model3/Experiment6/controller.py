"""Budget-driven receding-horizon control of the deterministic mean channel."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

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
    prediction_steps: int = 12
    decision_steps: int = 4
    optimizer_iterations: int = 24
    optimizer_log_tolerance: float = .025
    validation_step_cap: int = 100000
    maximum_prediction_steps: int = 192

    def payload(self):
        return asdict(self)


def validate_settings(settings):
    for name in ("minimum_dt", "maximum_dt", "mean_tolerance", "steady_window", "maximum_time", "optimizer_log_tolerance"):
        value = getattr(settings, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if settings.minimum_dt > settings.maximum_dt:
        raise ValueError("minimum-dt must not exceed maximum-dt")
    for name in ("step_budget", "prediction_steps", "decision_steps", "optimizer_iterations", "validation_step_cap", "maximum_prediction_steps"):
        value = getattr(settings, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings.decision_steps > settings.prediction_steps:
        raise ValueError("decision-steps must not exceed prediction-steps")
    if settings.maximum_prediction_steps < settings.prediction_steps:
        raise ValueError("maximum-prediction-steps must be at least prediction-steps")
    ratio = settings.steady_window/settings.minimum_dt
    if not np.isfinite(ratio) or planning.holding_steps(settings.minimum_dt, settings.steady_window) > settings.validation_step_cap:
        raise ValueError("minimum-dt needs more continuation steps than validation-step-cap")


def forecast_score(errors, bias, tolerance, current_error=None):
    """A gate-count proxy, not a rigorous bound on time to convergence."""
    horizon = len(errors)
    predicted_floor = max(errors[-1], bias)
    if predicted_floor <= tolerance:
        crossing = next(i+1 for i, error in enumerate(errors) if error <= tolerance)
        return crossing + .1*predicted_floor/tolerance, crossing
    current = errors[0] if current_error is None else current_error
    progress = current-predicted_floor
    if progress > 1e-12*max(1., current):
        # Extrapolate gate cost from progress per step. A vanishing dt that
        # barely changes the state is expensive even if its bias is tiny.
        cost = horizon*max(current-tolerance, tolerance)/progress
        return cost+.1*predicted_floor/tolerance, None
    return 1e12+horizon+(predicted_floor-current)/tolerance, None


class Predictor:
    def __init__(self, model, J, settings, policy, order, formula):
        self.model, self.J, self.settings = model, J, settings
        self.policy, self.order, self.formula = policy, order, formula
        self.target = physics.exact_stationary(model)
        if not np.all(np.isfinite(self.target)):
            raise ValueError("physical stationary covariance is not finite")
        self.prediction_mean_steps = 0

    @lru_cache(maxsize=256)
    def channel(self, dt):
        return physics.channel(self.model, self.J*dt, self.policy, self.order, self.formula)

    @lru_cache(maxsize=256)
    def stationary(self, dt):
        branch = self.channel(dt)
        if min(1-branch.weights[0, 0], 1-branch.weights[-1, -1]) <= 1e-12:
            raise ValueError("negligible endpoint damping")
        covariance, checks = physics.mean_stationary(branch, self.model.n)
        return planning.distance(covariance, self.target, self.model), checks

    def choose(self, state, previous_dt, remaining_steps, remaining_time):
        settings = self.settings
        upper = min(settings.maximum_dt, remaining_time)
        lower = settings.minimum_dt
        if upper < lower:
            return None, {"reason": "remaining physical time is below minimum-dt", "candidates": []}
        horizon = min(settings.prediction_steps, remaining_steps)
        evaluated = {}
        current_error = planning.distance(state, self.target, self.model)

        def evaluate(log_dt):
            dt = float(np.clip(np.exp(log_dt), lower, upper))
            if dt in evaluated:
                return evaluated[dt]["score"]
            row = {"dt": dt, "valid": False, "score": 1e100}
            try:
                branch = self.channel(dt)
                bias, checks = self.stationary(dt)
                covariance = state.copy()
                errors = []
                for _ in range(horizon):
                    covariance = physics.mean_step(covariance, branch)
                    errors.append(planning.distance(covariance, self.target, self.model))
                self.prediction_mean_steps += horizon
                score, crossing = forecast_score(errors, bias, settings.mean_tolerance, current_error)
                if not np.isfinite(score):
                    raise ValueError("nonfinite prediction score")
                row.update(valid=True, score=float(score), predicted_error=errors[-1],
                    stationary_bias=float(bias), predicted_crossing_step=crossing,
                    fixed_point_checks=checks)
            except (ValueError, np.linalg.LinAlgError) as error:
                row["reason"] = str(error)
            evaluated[dt] = row
            return row["score"]

        horizon_trials = []
        while True:
            evaluated = {}
            # Bounds and previous step are safeguards, not a timestep grid.
            for dt in (lower, upper, np.clip(previous_dt, lower, upper)):
                evaluate(np.log(dt))
            optimizer = None
            if np.log(upper)-np.log(lower) > settings.optimizer_log_tolerance:
                optimizer = minimize_scalar(evaluate, bounds=(np.log(lower), np.log(upper)), method="bounded",
                    options={"xatol": settings.optimizer_log_tolerance, "maxiter": settings.optimizer_iterations})
            candidates = list(evaluated.values())
            valid = [row for row in candidates if row["valid"]]
            horizon_trials.append({"prediction_steps": horizon, "candidates": candidates})
            progressing = [row for row in valid if row["score"] < 1e12]
            feasible = [row for row in progressing if row["score"] <= remaining_steps+.1]
            expanded = min(2*horizon, settings.maximum_prediction_steps, remaining_steps)
            if feasible or not valid or expanded <= horizon:
                break
            horizon = expanded
        audit = {"candidates": candidates, "prediction_steps": horizon,
                 "horizon_trials": horizon_trials,
                 "optimizer_converged": bool(optimizer.success) if optimizer is not None else True,
                 "mean_allowance": settings.mean_tolerance}
        if not valid:
            audit["reason"] = "no evaluated timestep had a valid stationary solve and prediction"
            return None, audit
        if not progressing:
            audit["reason"] = "no predicted progress within the available horizon"
            return None, audit
        winner = min(progressing, key=lambda row: (row["score"], -row["dt"]))
        audit["selected"] = winner
        audit["predicted_gate_budget_feasible"] = bool(winner["score"] <= remaining_steps+.1)
        return winner["dt"], audit


def forward_check(state, dt, time, predictor):
    settings = predictor.settings
    count = planning.holding_steps(dt, settings.steady_window)
    errors = [planning.distance(state, predictor.target, predictor.model)]
    times = [time]
    if count > settings.validation_step_cap:
        return False, {"times": times, "mean_errors": errors, "step_dts": [], "reason": "validation_step_cap"}
    covariance = state.copy()
    passed = errors[0] <= settings.mean_tolerance
    for step in range(count if passed else 0):
        covariance = physics.mean_step(covariance, predictor.channel(dt))
        errors.append(planning.distance(covariance, predictor.target, predictor.model))
        times.append(float(time+(step+1)*dt))
        if errors[-1] > settings.mean_tolerance:
            passed = False
            break
    return passed, {"times": times, "mean_errors": errors, "step_dts": [dt]*(len(times)-1),
        "interpretation": "classical constant-final-step continuation; sampling checked only at output"}


def plan(model, J, preparation, settings, *, policy="rate-weighted", order="hamiltonian-first", formula="even-odd", progress=None):
    validate_settings(settings)
    predictor = Predictor(model, J, settings, policy, order, formula)
    state = np.diag(preparation["populations"])
    times, dts, history, validation_attempts = [0.], [], [], []
    errors = [planning.distance(state, predictor.target, model)]
    previous = settings.maximum_dt
    validation = {"times": [0.], "mean_errors": errors.copy(), "step_dts": [],
                  "interpretation": "no complete passing forward check"}
    status, passed = "step_budget_reached", False
    validation_work = 0
    while len(dts) < settings.step_budget:
        if settings.maximum_time-times[-1] < settings.minimum_dt:
            status = "maximum_time_reached"
            break
        dt, audit = predictor.choose(state, previous, settings.step_budget-len(dts), settings.maximum_time-times[-1])
        audit.update(step=len(dts), time=times[-1], current_error=errors[-1])
        history.append(audit)
        if dt is None:
            status = "no_predicted_progress" if audit["candidates"] and any(r["valid"] for r in audit["candidates"]) else "no_valid_timestep"
            break
        # An already stationary preparation can have zero circuit evolution.
        if not dts and errors[-1] <= settings.mean_tolerance:
            passed, validation = forward_check(state, dt, times[-1], predictor)
            validation_work += len(validation["step_dts"])
            validation_attempts.append({"step": 0, "dt": dt, "passed": passed,
                                        "max_error": max(validation["mean_errors"])})
            if passed:
                status = "mean_target_met"
                break
        committed = 0
        for _ in range(min(settings.decision_steps, settings.step_budget-len(dts))):
            actual_dt = min(dt, settings.maximum_time-times[-1])
            if actual_dt < settings.minimum_dt:
                break
            state = physics.mean_step(state, predictor.channel(actual_dt))
            dts.append(float(actual_dt))
            times.append(float(times[-1]+actual_dt))
            errors.append(planning.distance(state, predictor.target, model))
            committed += 1
            if errors[-1] <= settings.mean_tolerance:
                passed, validation = forward_check(state, actual_dt, times[-1], predictor)
                validation_work += len(validation["step_dts"])
                validation_attempts.append({"step": len(dts), "dt": actual_dt, "passed": passed,
                                            "max_error": max(validation["mean_errors"])})
                if passed:
                    status = "mean_target_met"
                    break
        audit.update(committed_steps=committed, resulting_error=errors[-1])
        previous = dts[-1] if dts else dt
        if progress is not None:
            progress({"step": len(dts), "time": times[-1], "dt": previous,
                      "error": errors[-1], "validation_passed": passed})
        if passed:
            break
        if not committed:
            status = "maximum_time_reached"
            break
    if not passed:
        # A failed old probe must not be attached to a later output state.
        validation = {"times": [times[-1]], "mean_errors": [errors[-1]], "step_dts": [],
                      "interpretation": "no passing forward window at the output"}
    stages = []
    for dt in dts:
        if stages and stages[-1]["dt"] == dt:
            stages[-1]["steps"] += 1
        else:
            stages.append({"dt": dt, "steps": 1})
    selected = {"preparation": preparation["name"], "steps": len(dts), "final_time": times[-1],
        "step_dts": dts, "time_nodes": times, "stages": stages, "validation_mode": "forward",
        "tail_window_covered": passed, "mean_target_met": passed,
        "checked_from_step": len(dts), "checked_from_time": times[-1],
        "window_start": times[-1], "window_end": validation["times"][-1],
        "tail_mean_error": max(validation["mean_errors"]), "final_mean_error": errors[-1],
        "estimated_total_depth": None, "classical_validation_steps": len(validation["step_dts"]),
        "final_mean_covariance_real_gauge": state.tolist(), "forward_validation": validation}
    return {"method": "budget-driven continuous receding-horizon timestep control", "status": status,
        "settings": settings.payload(), "selected": selected, "selected_preparation": preparation,
        "physical_stationary_real_gauge": predictor.target.tolist(), "decisions": history,
        "preparation_mean_errors": errors, "validation_attempts": validation_attempts,
        "prediction_mean_steps": predictor.prediction_mean_steps, "validation_mean_steps": validation_work,
        "interpretation": "reference-assisted model control; local optimization and forecast scores do not prove globally minimal depth"}
