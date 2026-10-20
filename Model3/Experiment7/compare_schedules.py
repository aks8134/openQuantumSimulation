"""Compare automatic two-stage feedback with automatically fitted cosine annealing."""
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
from Model3.Experiment6.adaptive_budget import plot_plan
from Model3.Experiment7 import schedules


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=("two_stage", "cosine"), nargs="+", default=["two_stage", "cosine"])
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--p-left", type=float, default=.9)
    parser.add_argument("--p-right", type=float, default=.1)
    parser.add_argument("--exchange-prefactor", type=float, choices=(.25, .5), default=.5)
    parser.add_argument("--preparation", choices=planning.PREPARATIONS, default="half_mixed")
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument("--steady-tolerance", type=float, default=.02)
    parser.add_argument("--sampling-tolerance", type=float, help="default 20 percent of total tolerance")
    parser.add_argument("--steady-window", type=float, help="default 10/J")
    parser.add_argument("--minimum-dt", type=float, help="bound only; default .0001/max(J,gamma)")
    parser.add_argument("--maximum-dt", type=float, help="bound only; default 1/max(J,gamma)")
    parser.add_argument("--maximum-time", type=float, help="default 200/J")
    parser.add_argument("--decision-steps", type=int, default=4)
    parser.add_argument("--search-iterations", type=int, default=80)
    parser.add_argument("--feedback-iterations", type=int, default=16)
    parser.add_argument("--population-multiplier", type=int, default=10)
    parser.add_argument("--optimizer-seed", type=int, default=1729)
    parser.add_argument("--forecast-nodes", type=int, default=64)
    parser.add_argument("--forecast-margin", type=float, default=.99)
    parser.add_argument("--validation-step-cap", type=int, default=100000)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--trajectories", type=int)
    sampling.add_argument("--R-values", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[29, 30, 31])
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
    options.settings = schedules.Settings(options.minimum_dt, options.maximum_dt, options.mean_tolerance,
        options.steady_window, options.step_budget, options.maximum_time, options.decision_steps,
        options.search_iterations, options.feedback_iterations, options.population_multiplier,
        options.optimizer_seed, options.forecast_nodes, options.validation_step_cap, options.forecast_margin)
    schedules.validate_settings(options.settings)
    if not np.all(np.isfinite(options.J*np.array([options.minimum_dt, options.maximum_dt]))):
        raise ValueError("scaled dt bounds must be finite")
    options.R_values = sorted(options.R_values if options.R_values is not None else
                              [8192 if options.trajectories is None else options.trajectories])
    if min(options.R_values) < 1 or len(set(options.R_values)) != len(options.R_values):
        raise ValueError("R values must be distinct positive integers")
    if min(options.seeds) < 0 or len(set(options.seeds)) != len(options.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if len(set(options.methods)) != len(options.methods):
        raise ValueError("methods must be distinct")
    options.seeds = sorted(options.seeds)
    options.model = physics.Model(options.n_qubits, options.gamma/options.J,
                                 options.p_left, options.p_right, options.exchange_prefactor)
    options.validation_mode = "forward"
    options.estimated_step_depth = options.depth_budget = None
    return options


def configuration(options):
    names = ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor", "preparation",
             "steady_tolerance", "sampling_tolerance", "sampling_policy", "order", "system_formula",
             "R_values", "seeds", "plan_only", "methods")
    return {"schema_version": 1, "algorithm": "joint_budget_schedule_search_v1",
            **{name: getattr(options, name) for name in names}, "controller": options.settings.payload()}


def plot_comparison(plans, options, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for method, plan in plans.items():
        selected = plan["selected"]
        if selected["step_dts"]:
            axes[0].plot(np.arange(1, selected["steps"]+1), selected["step_dts"], label=method)
        axes[1].plot(selected["time_nodes"], plan["preparation_mean_errors"],
                     label=f"{method}: {selected['steps']} preparation steps")
        future = selected["forward_validation"]
        axes[1].plot(future["times"], future["mean_errors"], "--", color=axes[1].lines[-1].get_color())
    axes[0].set(xlabel="Preparation step", ylabel="dt", title="Automatically selected schedules")
    axes[1].axhline(options.mean_tolerance, color="black", linestyle=":", label="Mean allowance")
    axes[1].set(xlabel="Physical time (dashed: classical forward check)", ylabel="Maximum local error", yscale="log")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(study, path):
    config = study["configuration"]
    lines = ["# Experiment7: automatic two-stage feedback versus cosine annealing", "",
        f"Total tolerance {config['steady_tolerance']:g}; sampling allowance {config['sampling_tolerance']:g}.",
        "Both methods plan independently of trajectory seeds and R. No earlier schedules are loaded.", "",
        "| Method | Steps | Output time | Forward mean max | Planning seconds |",
        "|---|---:|---:|---:|---:|"]
    for method, record in study["methods"].items():
        s = record["selected"]
        lines.append(f"| {method} ({record['controller_status']}) | {s['steps']} | {s['final_time']:.8g} | {s['tail_mean_error']:.9g} | {record['planning_wall_seconds']:.3f} |")
    lines += ["", "| Method | Replay | Status | Sampled output error | Sampling distance | Two SE |",
              "|---|---|---|---:|---:|---:|"]
    for method, record in study["methods"].items():
        for key, row in record["replays"].items():
            a = row["assessment"]
            lines.append(f"| {method} | {key} | {row['status']} | {a['output_sampled_error']:.9g} | {a['max_sampling_distance']:.9g} | {a['max_two_standard_errors']:.9g} |")
    lines += ["", "Search jointly selects preparation length and timestep parameters with seeded differential evolution. The two-stage method replans after short committed prefixes and retains its own feasible continuation. Cosine selects its two endpoints and length, then follows that schedule, allowing early stopping only when the same output checks pass.",
        "", "Search observes a sparse, exactly advanced future window; every retained candidate and actual output is separately checked by direct mean evolution at every future integration node. Globally minimal depth is not established. The schedule search does not compile circuits; separately generated verification and resource measurements are in [VerifiedComparison.md](VerifiedComparison.md).",
        "", "Forward continuation uses the SAME final timestep for at least the configured physical-time window. It is classical work and is excluded from preparation steps. Sampling is checked at output only. Two-SE values are descriptive, not simultaneous confidence bounds. Initial mixture bits and measurement shots are not sampled; hardware noise is excluded."]
    if config["plan_only"]:
        lines += ["", "Plan only: no sampling-validated success is claimed."]
    path.write_text("\n".join(lines)+"\n")


def main(options):
    config = configuration(options)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = options.output_directory/f"comparison_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    prep = planning.preparations(options.model, physics.exact_stationary(options.model), [options.preparation])[0]
    study = {"experiment": "Model3/Experiment7", "configuration": config, "methods": {},
             "figure_generation_status": "disabled" if options.no_plots else "pending"}
    plans = {}
    for method in options.methods:
        directory = folder/method
        directory.mkdir(exist_ok=True)
        print(f"Planning {method}: mean budget={options.mean_tolerance:g}", flush=True)
        def progress(event):
            if not options.no_progress:
                print(f"Controller: {event}", flush=True)
        started = time.perf_counter()
        plan = schedules.plan(options.model, options.J, prep, options.settings, method=method,
            policy=options.sampling_policy, order=options.order, formula=options.system_formula, progress=progress)
        plan["planning_wall_seconds"] = time.perf_counter()-started
        physics.atomic_json(directory/"plan.json", plan)
        plans[method] = plan
        record = {"controller_status": plan["status"], "selected": plan["selected"],
            "planning_wall_seconds": plan["planning_wall_seconds"], "replays": {}}
        study["methods"][method] = record
        physics.atomic_json(folder/"study.json", study)
        write_report(study, folder/"Study.md")
        print(f"{method}: {plan['status']}, {plan['selected']['steps']} preparation steps", flush=True)
        if plan["selected"]["mean_target_met"] and not options.plan_only:
            for seed in options.seeds:
                for R in options.R_values:
                    replay_options = copy.copy(options)
                    replay_options.seed_trajectories = seed
                    started = time.perf_counter()
                    result = replay_engine.replay(replay_options, plan, R)
                    result.update(experiment="Model3/Experiment7", schedule_method=method,
                        controller_file="plan.json", replay_wall_seconds=time.perf_counter()-started,
                        figure_generation_status="disabled" if options.no_plots else "pending")
                    result["configuration"]["schedule_input"]["mode"] = method
                    filename = f"replay_seed{seed}_R{R}.json"
                    physics.atomic_json(directory/filename, result)
                    record["replays"][f"seed{seed}_R{R}"] = {"file": filename, "status": result["run_status"],
                                                              "assessment": result["assessment"]}
                    physics.atomic_json(folder/"study.json", study)
                    write_report(study, folder/"Study.md")
                    print(f"{method}: seed={seed}, R={R}, {result['run_status']}", flush=True)
        if not options.no_plots:
            from Model3.Experiment3.replot_selected_observables import plot_saved, selected_indices
            figures = directory/"figures"
            figures.mkdir(exist_ok=True)
            plot_plan(plan, figures/"schedule.png")
            for key, row in record["replays"].items():
                result = json.loads((directory/row["file"]).read_text())
                if plan["selected"]["steps"]:
                    plot_saved(result, selected_indices(options.n_qubits, None), selected_indices(options.n_qubits-1, None), figures/f"observables_{key}.png")
                    replay_engine.plot_forward_validation(result, options, figures/f"validation_{key}.png")
                result["figure_generation_status"] = "complete"
                physics.atomic_json(directory/row["file"], result)
    if not options.no_plots:
        plot_comparison(plans, options, folder/"comparison.png")
        study["figure_generation_status"] = "complete"
    physics.atomic_json(folder/"study.json", study)
    write_report(study, folder/"Study.md")
    print(f"Saved: {folder/'Study.md'}", flush=True)
    return study


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except ValueError as error:
        raise SystemExit(str(error)) from error
