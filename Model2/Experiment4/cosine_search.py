"""Offline minimum-step search for equal-reservoir Model2 steady preparation.

Schedules are selected using deterministic means, then independently replayed.
No backend connection, circuit execution, or hardware noise model is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import reservoir_scaling as physics

PREPARATIONS = ("half_mixed", "vacuum", "filled", "left_excitation")


def initial_state(n, preparation):
    if preparation == "half_mixed":
        return np.eye(n) / 2
    if preparation not in PREPARATIONS:
        raise ValueError("unsupported preparation")
    return physics.initial_covariances(n)[PREPARATIONS.index(preparation)-1].copy()


def schedule(family, steps, parameters):
    """Parameters are log endpoint steps and, for warped cosine, log power."""
    if family not in ("constant", "cosine", "warped_cosine") or steps < 1:
        raise ValueError("invalid schedule family or length")
    if family == "constant":
        return np.full(steps, np.exp(parameters[0]))
    low, high = sorted(np.exp(parameters[:2]))
    if steps == 1:
        return np.array([low])
    power = np.exp(parameters[2]) if family == "warped_cosine" else 1.
    x = np.linspace(0., 1., steps) ** power
    values = low + (high-low)/2 * (1 + np.cos(np.pi*x))
    values[0], values[-1] = high, low
    return values


def local_error(state, target, model):
    return max(physics.observable_error(state, target, model.prefactor).values())


def observables(state, model):
    return {"populations": np.diagonal(state, axis1=-2, axis2=-1).tolist(),
            "currents_divided_by_J": (4*model.prefactor*np.diagonal(
                state, offset=1, axis1=-2, axis2=-1)).tolist()}


def branch(model, dt, config):
    return physics.channel(model, config["J"]*float(dt), "rate-weighted",
                           config["order"], config["system_formula"])


def evolve(model, dts, config, *, history=False):
    state = initial_state(model.n, config["preparation"])
    states = [state.copy()] if history else None
    for dt in dts:
        state = physics.mean_step(state, branch(model, dt, config))
        if history:
            states.append(state.copy())
    return state, states


def continuation(model, state, dt, config):
    """Check every whole step covering the declared physical-time window."""
    target = model.p_left*np.eye(model.n)
    count = int(np.ceil(config["forward_window"]/dt))
    if count > config["validation_step_cap"]:
        return {"passed": False, "reason": "validation_step_cap", "errors": [], "steps": count}
    errors = [local_error(state, target, model)]
    channel = branch(model, dt, config)
    for _ in range(count):
        state = physics.mean_step(state, channel)
        errors.append(local_error(state, target, model))
    return {"passed": bool(max(errors) <= config["mean_tolerance"]),
            "maximum_error": max(errors), "errors": errors, "steps": count,
            "dt": float(dt), "duration": float(count*dt)}


def evaluate(model, dts, config):
    """Independent direct validation, including target invariance and physicality."""
    state, states = evolve(model, dts, config, history=True)
    target = model.p_left*np.eye(model.n)
    validation = continuation(model, state.copy(), float(dts[-1]), config)
    stationary_residual = max(float(np.max(np.abs(
        physics.mean_step(target, branch(model, dt, config))-target))) for dt in dts)
    eigenvalues = np.linalg.eigvalsh(np.asarray(states))
    physical = bool(eigenvalues.min() >= -1e-10 and eigenvalues.max() <= 1+1e-10)
    # Every homogeneous branch is a contraction. This spectral bound covers
    # populations and current/J at every later node, including variable steps.
    permanent_bound = max(1., 4*model.prefactor)*float(np.linalg.norm(state-target, 2))
    return {"steps": len(dts), "step_dts": np.asarray(dts).tolist(),
            "physical_time": float(np.sum(dts)), "output_error": local_error(state, target, model),
            "forward": validation, "passed": bool(validation["passed"] and physical and stationary_residual < 1e-10),
            "output_covariance": state.tolist(), "output_observables": observables(state, model),
            "time_nodes": np.r_[0., np.cumsum(dts)].tolist(),
            "mean_errors": [local_error(C, target, model) for C in states],
            "mean_observables": observables(np.asarray(states), model),
            "target_fixed_point_residual": stationary_residual,
            "minimum_covariance_eigenvalue": float(eigenvalues.min()),
            "maximum_covariance_eigenvalue": float(eigenvalues.max()),
            "permanent_local_error_bound": permanent_bound,
            "permanent_mean_bound_passed": bool(permanent_bound <= config["mean_tolerance"])}


def parameter_bounds(family, config):
    endpoint = (np.log(config["minimum_dt"]), np.log(config["maximum_dt"]))
    bounds = [endpoint] if family == "constant" else [endpoint, endpoint]
    if family == "warped_cosine":
        bounds.append((np.log(config["minimum_power"]), np.log(config["maximum_power"])))
    return bounds


def search_family(model, family, config, progress=None):
    """Joint integer search followed by explicit shorter-length challenges.

    Failure of a bounded optimizer is never interpreted as infeasibility.
    Only directly validated candidates can become incumbents.
    """
    target = model.p_left*np.eye(model.n)
    best = None
    trials = []
    bounds = parameter_bounds(family, config)
    start_time = time.perf_counter()

    def optimize(seed, fixed_steps=None):
        nonlocal best
        evaluations = 0
        trial_best = None

        def objective(vector):
            nonlocal best, evaluations, trial_best
            evaluations += 1
            count = int(round(vector[0])) if fixed_steps is None else fixed_steps
            parameters = vector[1:] if fixed_steps is None else vector
            dts = schedule(family, count, parameters)
            if dts.sum() > config["maximum_time"]:
                return config["step_budget"]*(2 + dts.sum()/config["maximum_time"])
            state, _ = evolve(model, dts, config)
            error = local_error(state, target, model)
            if error <= config["mean_tolerance"]:
                future = continuation(model, state, float(dts[-1]), config)
                error = future.get("maximum_error", np.inf)
            candidate = {"steps": count, "parameters": list(map(float, parameters)),
                         "error": float(error), "passed": bool(error <= config["mean_tolerance"])}
            if trial_best is None or (not candidate["passed"], count if candidate["passed"] else error, error) < (
                    not trial_best["passed"], trial_best["steps"] if trial_best["passed"] else trial_best["error"], trial_best["error"]):
                trial_best = candidate
            if candidate["passed"] and (best is None or (count, error) < (best["steps"], best["error"])):
                best = candidate.copy()
            if fixed_steps is not None:
                return error
            if candidate["passed"]:
                return count + .1*error/config["mean_tolerance"]
            return config["step_budget"]+1+config["step_budget"]*error/config["mean_tolerance"]

        search_bounds = ([(1, config["step_budget"])]+bounds) if fixed_steps is None else bounds
        result = differential_evolution(objective, search_bounds, seed=seed,
            integrality=([True]+[False]*len(bounds)) if fixed_steps is None else None,
            maxiter=config["iterations"], popsize=config["population_multiplier"],
            polish=False, tol=1e-5, atol=1e-8)
        # A shorter schedule starts with its own optimized point plus the
        # incumbent's endpoints; it does not inherit acceptance from that fit.
        if fixed_steps is not None:
            starts = [result.x]
            if best is not None:
                starts.append(np.asarray(best["parameters"]))
            for point in starts:
                minimize(objective, point, method="Nelder-Mead", bounds=bounds,
                         options={"maxiter": config["local_iterations"], "xatol": 1e-7, "fatol": 1e-9})
        record = {"seed": seed, "fixed_steps": fixed_steps, "evaluations": evaluations,
                  "optimizer_converged": bool(result.success), "best": trial_best}
        trials.append(record)
        if progress:
            progress(record)

    for seed in config["optimizer_seeds"]:
        optimize(seed)
    # Always test several shorter counts even after a failure; feasible
    # schedule lengths need not form a monotone set under this parametrization.
    if best is not None:
        anchor = best["steps"]
        for count in range(anchor-1, max(0, anchor-config["neighbor_counts"]-1), -1):
            for seed in config["optimizer_seeds"]:
                optimize(seed, count)
    selected = None if best is None else evaluate(model, schedule(family, best["steps"], best["parameters"]), config)
    if selected is not None:
        selected.update(family=family, parameters=best["parameters"])
        if not selected["passed"]:
            raise RuntimeError("selected schedule failed independent direct validation")
    return {"family": family, "status": "mean_target_met" if selected else "no_passing_schedule_found",
            "selected": selected, "trials": trials, "wall_seconds": time.perf_counter()-start_time,
            "minimality": "smallest passing schedule found in this bounded search; no global-minimum proof"}


def sampled_replay(model, selected, config, seed, trajectories):
    """Independent per-path RNGs preserve trajectories when R increases."""
    initial = initial_state(model.n, config["preparation"])
    states = np.broadcast_to(initial, (1, trajectories, model.n, model.n)).copy()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(trajectories)]
    choices = np.empty((trajectories, selected["steps"]), dtype=np.uint8)
    for index, dt in enumerate(selected["step_dts"]):
        channel = branch(model, dt, config)
        uniforms = np.array([stream.random() for stream in streams])
        draws = np.searchsorted(np.cumsum(channel.probabilities), uniforms, side="right")
        choices[:, index] = draws
        states = physics.sampled_step(states, draws, channel)
    paths = states[0]
    mean = paths.mean(axis=0)
    standard_error = paths.std(axis=0, ddof=1)/np.sqrt(trajectories)
    target = model.p_left*np.eye(model.n)
    distance = local_error(mean, np.asarray(selected["output_covariance"]), model)
    error = local_error(mean, target, model)
    two_se = local_error(2*standard_error, np.zeros_like(target), model)
    return {"seed": seed, "trajectories": trajectories, "sampled_error": error,
            "sampling_distance": distance, "maximum_two_standard_errors": two_se,
            "passed": bool(error <= config["total_tolerance"] and distance <= config["sampling_tolerance"]
                           and two_se <= config["sampling_tolerance"]),
            "output_observables": observables(mean, model),
            "two_standard_errors": observables(2*standard_error, model),
            "choices_sha256": hashlib.sha256(choices.tobytes()).hexdigest(),
            "sampling_note": "initial mixture averaged analytically; no shot/device noise; two SE descriptive, not simultaneous coverage"}


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--J", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=2.)
    parser.add_argument("--p", type=float, default=.25)
    parser.add_argument("--preparation", choices=PREPARATIONS, default="half_mixed")
    parser.add_argument("--exchange-prefactor", type=float, choices=[.25, .5], default=.25)
    parser.add_argument("--order", choices=["jump-first", "hamiltonian-first"], default="jump-first")
    parser.add_argument("--system-formula", choices=["even-odd", "exact"], default="even-odd")
    parser.add_argument("--families", nargs="+", choices=["constant", "cosine", "warped_cosine"],
                        default=["constant", "cosine", "warped_cosine"])
    parser.add_argument("--step-budget", type=int, default=256)
    parser.add_argument("--total-tolerance", type=float, default=.02)
    parser.add_argument("--sampling-tolerance", type=float, default=.004)
    parser.add_argument("--minimum-dt", type=float, default=.01)
    parser.add_argument("--maximum-dt", type=float, default=2.)
    parser.add_argument("--minimum-power", type=float, default=.1)
    parser.add_argument("--maximum-power", type=float, default=4.)
    parser.add_argument("--forward-window", type=float, default=10.)
    parser.add_argument("--maximum-time", type=float, default=500.)
    parser.add_argument("--validation-step-cap", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--population-multiplier", type=int, default=8)
    parser.add_argument("--local-iterations", type=int, default=120)
    parser.add_argument("--neighbor-counts", type=int, default=4)
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[1729, 1730])
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[29, 30, 31])
    parser.add_argument("--trajectories", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"runs")
    options = parser.parse_args(arguments)
    config = vars(options).copy()
    config.pop("output_directory")
    config.pop("no_plots")
    for name in ("J", "gamma", "total_tolerance", "minimum_dt", "maximum_dt", "minimum_power",
                 "maximum_power", "forward_window", "maximum_time"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if options.n_qubits < 2 or not np.isfinite(options.gamma/options.J) or not 0 < options.p < 1:
        raise ValueError("invalid model parameters")
    if not np.isfinite(options.sampling_tolerance) or not 0 < options.sampling_tolerance < options.total_tolerance:
        raise ValueError("sampling tolerance must be between zero and total tolerance")
    if options.minimum_dt > options.maximum_dt or options.minimum_power > options.maximum_power:
        raise ValueError("inverted search bounds")
    for name in ("step_budget", "iterations", "population_multiplier", "local_iterations", "validation_step_cap"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if options.neighbor_counts < 0 or np.ceil(options.forward_window/options.minimum_dt) > options.validation_step_cap:
        raise ValueError("invalid neighbor count or continuation cap")
    for name in ("optimizer_seeds", "sampling_seeds", "trajectories", "families"):
        if len(set(config[name])) != len(config[name]):
            raise ValueError(f"duplicate {name}")
    if min(options.optimizer_seeds+options.sampling_seeds) < 0 or min(options.trajectories) < 2:
        raise ValueError("seeds must be nonnegative and trajectories >= 2")
    config["trajectories"] = sorted(options.trajectories)
    config["mean_tolerance"] = options.total_tolerance-options.sampling_tolerance
    config["sampling_policy"] = "rate-weighted"
    config["schema_version"] = 1
    return options, config


def write_report(study, directory):
    c = study["configuration"]
    lines = ["# Model2 Experiment4: cosine preparation search", "",
        f"N={c['n_qubits']}, J={c['J']:g}, gamma={c['gamma']:g}, p={c['p']:g}, exchange prefactor={c['exchange_prefactor']:g}; preparation `{c['preparation']}`.",
        f"Rate-weighted single-jump dilation, `{c['order']}`, `{c['system_formula']}` Hamiltonian factor.",
        f"All populations and currents/J: total tolerance {c['total_tolerance']:g} = mean {c['mean_tolerance']:g} + sampling {c['sampling_tolerance']:g}.", "",
        "| Family | Steps | Output time | Output mean error | Forward maximum | Permanent mean bound | Sampled checks |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for name, result in study["families"].items():
        selected = result["selected"]
        if selected is None:
            lines.append(f"| {name} | — | — | — | — | — | No passing schedule found |")
            continue
        replays = result.get("replays", [])
        label = "not run" if not replays else ("passed" if result.get("sampling_passed") else "unresolved")
        lines.append(f"| {name} | {selected['steps']} | {selected['physical_time']:.8g} | {selected['output_error']:.8g} | {selected['forward']['maximum_error']:.8g} | {selected['permanent_local_error_bound']:.8g} | {label} |")
    lines += ["", "The smallest passing count found is a bounded numerical-search result, not a proven global minimum. Shorter failed searches do not establish impossibility. Constant steps use the same model, bounds, objective and accuracy checks.",
        "", f"The forward check directly evaluates every node for at least {c['forward_window']:g} physical-time units at the final timestep, including the output. These classical continuation steps add no preparation gates. The separate spectral-norm bound, when passing, covers all later ideal mean nodes under contractive rate-weighted steps preserving pI; it is a floating-point bound, not an interval certificate or a many-body state-distance claim.",
        "", "The target is pI with zero current. Equal-reservoir rate-weighted mean steps preserve it even at finite step, so annealing is not needed to remove a stationary bias. Half-mixture is a nonstationary input when p differs from 0.5. Large preparation steps need not reproduce the physical transient.", "",
        "| Family | Seed | R | Sampled error | Sampling distance | Maximum two SE | Pass |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for name, result in study["families"].items():
        for row in result.get("replays", []):
            lines.append(f"| {name} | {row['seed']} | {row['trajectories']} | {row['sampled_error']:.8g} | {row['sampling_distance']:.8g} | {row['maximum_two_standard_errors']:.8g} | {row['passed']} |")
    lines += ["", "Schedule selection uses deterministic means only. The fixed seed list is not reselected. Larger ensembles are tried only if a smaller R fails; every attempted replay is retained. Mixture preparation is analytically averaged. Two standard errors are descriptive, not simultaneous confidence guarantees. Measurement shots, device noise, routing and native compiled depth are outside this experiment.",
              "", "Reproducible configuration, optimizer trials, complete step lists, means, forward checks and replay summaries are in `study.json`. Figures show actual observables as well as errors."]
    (directory/"Study.md").write_text("\n".join(lines)+"\n")


def plot(study, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    c = study["configuration"]
    sites = sorted(set([0, c["n_qubits"]//2, c["n_qubits"]-1]))
    for name, result in study["families"].items():
        s = result["selected"]
        if s is None:
            continue
        axes[0, 0].plot(np.arange(1, s["steps"]+1), s["step_dts"], label=f"{name}: {s['steps']} steps")
        axes[0, 1].plot(s["time_nodes"], s["mean_errors"], label=name)
        populations = np.asarray(s["mean_observables"]["populations"])
        for site in sites:
            axes[1, 0].plot(s["time_nodes"], populations[:, site], label=f"{name}, site {site}")
        axes[1, 1].plot(s["output_observables"]["populations"], marker="o", label=name)
    axes[0, 0].set(xlabel="Preparation step", ylabel="dt")
    axes[0, 1].axhline(c["mean_tolerance"], color="black", linestyle=":", label="Mean allowance")
    axes[0, 1].set(xlabel="Physical time", ylabel="Maximum local mean error", yscale="log")
    axes[1, 0].axhline(c["p"], color="black", linestyle=":")
    axes[1, 0].set(xlabel="Physical time", ylabel="Population")
    axes[1, 1].axhline(c["p"], color="black", linestyle=":", label="Target")
    axes[1, 1].set(xlabel="Site (zero based)", ylabel="Output population")
    for axis in axes.flat:
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.savefig(directory/"comparison.png", dpi=160)
    fig.savefig(directory/"comparison.pdf")
    plt.close(fig)


def main(arguments=None):
    options, config = parse_arguments(arguments)
    model = physics.Model(options.n_qubits, options.gamma/options.J, options.p, options.p, options.exchange_prefactor)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    directory = options.output_directory/f"cosine_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    study = {"experiment": "Model2/Experiment4", "configuration": config, "families": {},
             "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), ROOT/"library/reservoir_scaling.py")}}
    for family in config["families"]:
        print(f"Searching {family}", flush=True)
        def progress(record):
            print(f"{family}: seed={record['seed']}, fixed_steps={record['fixed_steps']}, best={record['best']}", flush=True)
        result = search_family(model, family, config, progress)
        study["families"][family] = result
        result["replays"] = []
        physics.atomic_json(directory/"study.json", study)
        write_report(study, directory)
        if result["selected"] is not None and not options.plan_only:
            result["sampling_passed"] = False
            for R in config["trajectories"]:
                rows = [sampled_replay(model, result["selected"], config, seed, R)
                        for seed in config["sampling_seeds"]]
                result["replays"].extend(rows)
                print(f"{family}: R={R}, sampled checks={[row['passed'] for row in rows]}", flush=True)
                physics.atomic_json(directory/"study.json", study)
                if all(row["passed"] for row in rows):
                    result["sampling_passed"] = True
                    break
        physics.atomic_json(directory/"study.json", study)
        write_report(study, directory)
    if not options.no_plots:
        plot(study, directory)
    print(f"Saved {directory/'Study.md'}", flush=True)
    return study


if __name__ == "__main__":
    main()
