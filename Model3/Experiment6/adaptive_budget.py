"""Automatically choose Model3 timesteps from a steady-state error budget.

No manual timestep grid, retained schedule, or multi-configuration sweep.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import reservoir_scaling as physics
from Model3.Experiment4 import budgeted_steady_state as replay_engine
from Model3.Experiment4 import planning
from Model3.Experiment6 import controller


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--p-left", type=float, default=.9)
    parser.add_argument("--p-right", type=float, default=.1)
    parser.add_argument("--exchange-prefactor", type=float, choices=(.25, .5), default=.5)
    parser.add_argument("--preparation", choices=planning.PREPARATIONS, default="half_mixed")
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument("--steady-tolerance", type=float, default=.02)
    parser.add_argument("--sampling-tolerance", type=float, help="default 20 percent of the total tolerance")
    parser.add_argument("--steady-window", type=float, help="default 10/J, checked after output on mean only")
    parser.add_argument("--minimum-dt", type=float, help="lower bound only; default 0.0001/max(J,gamma)")
    parser.add_argument("--maximum-dt", type=float, help="upper bound only; default 1/max(J,gamma)")
    parser.add_argument("--maximum-time", type=float, help="preparation time cap; default 200/J")
    parser.add_argument("--prediction-steps", type=int, default=12)
    parser.add_argument("--maximum-prediction-steps", type=int, default=192)
    parser.add_argument("--decision-steps", type=int, default=4)
    parser.add_argument("--optimizer-iterations", type=int, default=24)
    parser.add_argument("--optimizer-log-tolerance", type=float, default=.025)
    parser.add_argument("--validation-step-cap", type=int, default=100000)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--trajectories", type=int)
    sampling.add_argument("--R-values", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[29])
    parser.add_argument("--sampling-policy", choices=("rate-weighted", "uniform"), default="rate-weighted")
    parser.add_argument("--order", choices=("hamiltonian-first", "jump-first"), default="hamiltonian-first")
    parser.add_argument("--system-formula", choices=("even-odd", "exact"), default="even-odd")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY/"runs")
    options = parser.parse_args(arguments)
    for name in ("J", "gamma", "steady_tolerance"):
        if not np.isfinite(getattr(options, name)) or getattr(options, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if options.n_qubits < 2 or not np.isfinite(options.gamma/options.J):
        raise ValueError("require n-qubits >= 2 and finite gamma/J")
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in (options.p_left, options.p_right)):
        raise ValueError("invalid reservoir probabilities")
    scale = max(options.J, options.gamma)
    for name, default in (("minimum_dt", .0001/scale), ("maximum_dt", 1/scale),
                          ("steady_window", 10/options.J), ("maximum_time", 200/options.J)):
        if getattr(options, name) is None:
            setattr(options, name, default)
    options.sampling_tolerance = (.2*options.steady_tolerance if options.sampling_tolerance is None else options.sampling_tolerance)
    if not np.isfinite(options.sampling_tolerance) or not 0 <= options.sampling_tolerance < options.steady_tolerance:
        raise ValueError("sampling allowance must be nonnegative and smaller than total tolerance")
    options.mean_tolerance = options.steady_tolerance-options.sampling_tolerance
    options.settings = controller.Settings(options.minimum_dt, options.maximum_dt, options.mean_tolerance,
        options.steady_window, options.step_budget, options.maximum_time, options.prediction_steps,
        options.decision_steps, options.optimizer_iterations, options.optimizer_log_tolerance, options.validation_step_cap,
        options.maximum_prediction_steps)
    controller.validate_settings(options.settings)
    scaled = options.J*np.array([options.minimum_dt, options.maximum_dt])
    if not np.all(np.isfinite(scaled)) or np.any(scaled <= 0):
        raise ValueError("scaled dt bounds must be positive and finite")
    options.R_values = sorted(options.R_values if options.R_values is not None else [8192 if options.trajectories is None else options.trajectories])
    if min(options.R_values) < 1 or len(set(options.R_values)) != len(options.R_values):
        raise ValueError("R values must be distinct positive integers")
    if min(options.seeds) < 0 or len(set(options.seeds)) != len(options.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    options.seeds = sorted(options.seeds)
    options.model = physics.Model(options.n_qubits, options.gamma/options.J, options.p_left, options.p_right, options.exchange_prefactor)
    options.validation_mode = "forward"
    options.estimated_step_depth = options.depth_budget = None
    return options


def configuration(options):
    names = ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor", "preparation",
             "steady_tolerance", "sampling_tolerance", "sampling_policy", "order", "system_formula",
             "R_values", "seeds", "plan_only")
    return {"schema_version": 1, "algorithm": "continuous_budget_controller_v2",
            **{name: getattr(options, name) for name in names}, "controller": options.settings.payload()}


def plot_plan(plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = plan["selected"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    if selected["step_dts"]:
        axes[0].stairs(selected["step_dts"], selected["time_nodes"])
    axes[0].set(ylabel="Chosen dt", title=f"Automatic budget controller: {selected['steps']} preparation steps")
    axes[1].semilogy(selected["time_nodes"], np.maximum(plan["preparation_mean_errors"], 1e-16), label="Actual preparation mean")
    future = selected["forward_validation"]
    axes[1].semilogy(future["times"], np.maximum(future["mean_errors"], 1e-16), "--", label="Classical forward check")
    axes[1].axhline(plan["settings"]["mean_tolerance"], color="black", linestyle=":", label="Mean allowance")
    axes[1].axvline(selected["final_time"], color="gray", alpha=.5, label="Circuit stops")
    axes[1].set(xlabel="Physical time", ylabel="Maximum local error")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(options):
    config = configuration(options)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = options.output_directory/f"adaptive_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    target = physics.exact_stationary(options.model)
    prep = planning.preparations(options.model, target, [options.preparation])[0]
    def progress(event):
        if not options.no_progress:
            print(f"Controller: {event}", flush=True)
    print(f"Experiment6: automatic mean-budget={options.mean_tolerance:g}; no grid or prior schedules", flush=True)
    started = time.perf_counter()
    plan = controller.plan(options.model, options.J, prep, options.settings, policy=options.sampling_policy,
        order=options.order, formula=options.system_formula, progress=progress)
    plan["planning_wall_seconds"] = time.perf_counter()-started
    physics.atomic_json(folder/"plan.json", plan)
    selected = plan["selected"]
    study = {"experiment": "Model3/Experiment6", "configuration": config, "controller_status": plan["status"],
        "selected": selected, "replays": {}, "figure_generation_status": "disabled" if options.no_plots else "pending"}
    physics.atomic_json(folder/"study.json", study)
    print(f"Controller {plan['status']}: {selected['steps']} steps, mean window error={selected['tail_mean_error']}", flush=True)
    if selected["mean_target_met"] and not options.plan_only:
        for seed in options.seeds:
            for R in options.R_values:
                replay_options = copy.copy(options)
                replay_options.seed_trajectories = seed
                started = time.perf_counter()
                result = replay_engine.replay(replay_options, plan, R)
                result.update(experiment="Model3/Experiment6", controller_status=plan["status"],
                    controller_file="plan.json", replay_wall_seconds=time.perf_counter()-started,
                    figure_generation_status="disabled" if options.no_plots else "pending")
                result["configuration"]["schedule_input"]["mode"] = "adaptive_controller"
                filename = f"replay_seed{seed}_R{R}.json"
                physics.atomic_json(folder/filename, result)
                study["replays"][f"seed{seed}_R{R}"] = {"file": filename, "status": result["run_status"], "assessment": result["assessment"]}
                physics.atomic_json(folder/"study.json", study)
                print(f"Seed {seed}, R={R}: {result['run_status']}", flush=True)
    report = ["# Experiment6: automatic budget-driven schedule", "", f"Controller status: **{plan['status']}**.",
        f"Preparation: {options.preparation}; steps: {selected['steps']}/{options.step_budget}; output time: {selected['final_time']:.8g}.", "",
        f"Total tolerance {options.steady_tolerance:g}; mean {options.mean_tolerance:g}; sampling {options.sampling_tolerance:g}.",
        f"Worst forward mean error: {selected['tail_mean_error']:.9g}. Classical continuation: {selected['classical_validation_steps']} steps.", "",
        "The controller uses continuous bounded timestep optimization and current mean-state feedback. No prior schedules or timestep grid were loaded. Forecast scores are heuristics; actual mean validation checks every node. Sampling is checked at output only.", "",
        "| Steps | dt |", "|---:|---:|"]
    report += [f"| {stage['steps']} | {stage['dt']:.9g} |" for stage in selected["stages"]]
    report += ["", "| Replay | Status | Sampled output error | Sampling distance | Two SE |",
               "|---|---|---:|---:|---:|"]
    for key, row in study["replays"].items():
        a = row["assessment"]
        report.append(f"| {key} | {row['status']} | {a['output_sampled_error']:.9g} | {a['max_sampling_distance']:.9g} | {a['max_two_standard_errors']:.9g} |")
    if not study["replays"]:
        report += ["", "No sampled replay was performed. A mean-only result is not a sampling-validated success."]
    report += ["", "Reference-assisted covariance model; no hardware feedback, shots, hardware noise, or compiled-depth calculation. The controller does not guarantee a globally shortest schedule or beat an offline optimized schedule. Validation covers a finite future mean window, not permanent settling.", "",
        "The numerical optimizer is local; see [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)."]
    (folder/"Study.md").write_text("\n".join(report)+"\n")
    if not options.no_plots:
        from Model3.Experiment3.replot_selected_observables import plot_saved, selected_indices
        figures = folder/"figures"
        figures.mkdir(exist_ok=True)
        plot_plan(plan, figures/"controller.png")
        for key, row in study["replays"].items():
            result = json.loads((folder/row["file"]).read_text())
            if selected["steps"]:
                plot_saved(result, selected_indices(options.n_qubits, None), selected_indices(options.n_qubits-1, None), figures/f"observables_{key}.png")
                replay_engine.plot_forward_validation(result, options, figures/f"validation_{key}.png")
            result["figure_generation_status"] = "complete" if selected["steps"] else "no_evolution"
            physics.atomic_json(folder/row["file"], result)
        study["figure_generation_status"] = "complete"
        physics.atomic_json(folder/"study.json", study)
    print(f"Saved: {folder/'Study.md'}", flush=True)
    return study


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except ValueError as error:
        raise SystemExit(str(error)) from error
