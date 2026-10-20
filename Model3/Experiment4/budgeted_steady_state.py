"""Budget-first Model3 steady-state preparation and schedule search.

Classical mean-channel planning followed by sampled single-jump replay.
Trotter steps are a depth proxy; no circuits, transpilation, or hardware.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import reservoir_scaling as physics
from Model3.Experiment4 import planning

FAMILIES = ("populations", "magnetizations", "excitation_flows")


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--p-left", type=float, default=.9)
    parser.add_argument("--p-right", type=float, default=.1)
    parser.add_argument("--exchange-prefactor", type=float, choices=(.25, .5), default=.5)
    parser.add_argument("--step-budget", type=int, default=500)
    parser.add_argument("--step-grid", type=float, nargs="+", help="physical step candidates; default (1,.5,...,.015625)/max(J,gamma)")
    parser.add_argument("--preparations", nargs="+", choices=planning.PREPARATIONS, default=list(planning.PREPARATIONS))
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--block-steps", type=int, default=10, help="beam-search switching resolution")
    parser.add_argument("--beam-width", type=int, default=2, help="paths retained per preparation and final step size")
    parser.add_argument("--search-node-cap", type=int, default=50000, help="beam expansions; 0 requests constant baselines only")
    parser.add_argument("--steady-window", type=float, help="physical-time validation window; default 10/J")
    parser.add_argument("--validation-mode", choices=("trailing", "forward"), default="trailing",
                        help="forward validates constant-dt mean continuation after output; only preparation counts against budget")
    parser.add_argument("--baseline-study", type=Path, nargs="+", default=[],
                        help="retain compatible prior study.json schedules as mandatory baselines")
    parser.add_argument("--plan-only", action="store_true", help="save deterministic search without sampled replay")
    parser.add_argument("--steady-tolerance", type=float, default=.01)
    parser.add_argument("--sampling-tolerance", type=float, help="sampling allowance; default .4*steady-tolerance")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--trajectories", type=int, help="default 64")
    sampling.add_argument("--R-values", type=int, nargs="+", help="replay the SAME selected schedule at these trajectory counts")
    parser.add_argument("--seed-trajectories", "--seed", dest="seed_trajectories", type=int, default=29)
    parser.add_argument("--sampling-policy", choices=("rate-weighted", "uniform"), default="rate-weighted")
    parser.add_argument("--order", choices=("hamiltonian-first", "jump-first"), default="hamiltonian-first")
    parser.add_argument("--system-formula", choices=("even-odd", "exact"), default="even-odd")
    parser.add_argument("--estimated-step-depth", type=int, help="optional supplied constant circuit-depth cost per step")
    parser.add_argument("--depth-budget", type=int, help="optional additive modeled depth cap, including local preparation layer")
    parser.add_argument("--output-directory", "--output-dir", type=Path, default=DIRECTORY/"runs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    options = parser.parse_args(arguments)
    for name in ("J", "gamma", "steady_tolerance"):
        if not np.isfinite(getattr(options, name)) or getattr(options, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if options.n_qubits < 2 or not np.isfinite(options.gamma/options.J):
        raise ValueError("require n-qubits >= 2 and finite gamma/J")
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in (options.p_left, options.p_right)):
        raise ValueError("reservoir populations must lie in [0,1]")
    for name in ("step_budget", "max_stages", "block_steps", "beam_width"):
        if getattr(options, name) < 1:
            raise ValueError(f"{name.replace('_','-')} must be positive")
    if options.search_node_cap < 0 or options.seed_trajectories < 0:
        raise ValueError("search-node-cap and seed must be nonnegative")
    if len(set(options.preparations)) != len(options.preparations):
        raise ValueError("preparations must be distinct")
    options.step_grid = ([float(2.**(-k)/max(options.J, options.gamma)) for k in range(7)]
                         if options.step_grid is None else options.step_grid)
    if (any(not np.isfinite(dt) or dt <= 0 for dt in options.step_grid)
            or len(set(options.step_grid)) != len(options.step_grid)):
        raise ValueError("step-grid must contain distinct positive finite values")
    options.step_grid = sorted(options.step_grid, reverse=True)
    scaled = options.J*np.asarray(options.step_grid)
    if np.any(~np.isfinite(scaled)) or np.any(scaled <= 0) or np.any(~np.isfinite(options.gamma*np.asarray(options.step_grid))):
        raise ValueError("scaled timestep values must be positive and finite")
    options.steady_window = 10/options.J if options.steady_window is None else options.steady_window
    if not np.isfinite(options.steady_window) or options.steady_window <= 0:
        raise ValueError("steady-window must be positive and finite")
    if options.validation_mode == "forward" and any(not np.isfinite(options.steady_window/dt) for dt in options.step_grid):
        raise ValueError("forward window/step ratios must be finite")
    options.sampling_tolerance = .4*options.steady_tolerance if options.sampling_tolerance is None else options.sampling_tolerance
    if not np.isfinite(options.sampling_tolerance) or not 0 <= options.sampling_tolerance < options.steady_tolerance:
        raise ValueError("sampling-tolerance must be nonnegative and less than steady-tolerance")
    options.mean_tolerance = options.steady_tolerance-options.sampling_tolerance
    values = options.R_values if options.R_values is not None else [64 if options.trajectories is None else options.trajectories]
    if any(value < 1 for value in values) or len(set(values)) != len(values):
        raise ValueError("trajectory counts must be distinct positive integers")
    options.R_values = sorted(values)
    if (options.estimated_step_depth is None) != (options.depth_budget is None):
        raise ValueError("estimated-step-depth and depth-budget must be supplied together")
    if options.depth_budget is not None and (options.depth_budget < 1 or options.estimated_step_depth < 1):
        raise ValueError("modeled depth costs and budget must be positive integers")
    options.model = physics.Model(options.n_qubits, options.gamma/options.J,
                                  options.p_left, options.p_right, options.exchange_prefactor)
    return options


def configuration(options, baseline_manifest=None):
    names = ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor", "step_budget", "step_grid",
             "preparations", "max_stages", "block_steps", "beam_width", "search_node_cap", "steady_window",
             "steady_tolerance", "sampling_tolerance", "mean_tolerance", "R_values", "seed_trajectories",
             "sampling_policy", "order", "system_formula", "estimated_step_depth", "depth_budget")
    config = {"schema_version": 1, "algorithm": "budgeted_mean_beam_search_v1", **{name: getattr(options, name) for name in names}}
    if options.validation_mode != "trailing" or options.baseline_study or options.plan_only:
        config.update(schema_version=2, algorithm="budgeted_mean_beam_search_v2",
            validation_mode=options.validation_mode, baseline_studies=baseline_manifest or [], plan_only=options.plan_only)
    return config


def observables(C, options):
    populations = np.diagonal(C, axis1=-2, axis2=-1)
    currents = 4*options.exchange_prefactor*options.J*np.diagonal(C, offset=1, axis1=-2, axis2=-1)
    return dict(zip(FAMILIES, (populations, 1-2*populations, currents)))


def replay(options, search, trajectories):
    selected, prep = search["selected"], search["selected_preparation"]
    dts = selected["step_dts"]
    nodes = np.r_[0., np.cumsum(dts)]
    target = np.asarray(search["physical_stationary_real_gauge"])
    initial = np.diag(prep["populations"])[None]
    mean, exact = initial.copy(), initial.copy()
    sampled = np.repeat(initial[:, None], trajectories, axis=1)
    streams = [np.random.default_rng(np.random.SeedSequence([options.seed_trajectories, path])) for path in range(trajectories)]
    probabilities = (np.full(4, .25) if options.sampling_policy == "uniform" else
        np.array([options.p_left, 1-options.p_left, options.p_right, 1-options.p_right])/2)
    choices = np.empty((trajectories, len(dts)), dtype=np.uint8)
    stores = [{name: [] for name in FAMILIES} for _ in range(4)]
    cache, block = {}, None
    for step in range(len(nodes)):
        if step:
            dt = dts[step-1]
            if dt not in cache:
                cache[dt] = (physics.channel(options.model, options.J*dt, options.sampling_policy, options.order, options.system_formula),
                             expm(options.J*dt*physics.generator(options.model)))
            branch, E = cache[dt]
            offset = (step-1) % 1024
            if offset == 0:
                block = physics.draw_choice_block(streams, probabilities)
            choice = block[:, offset]
            choices[:, step-1] = choice
            sampled = physics.sampled_step(sampled, choice, branch)
            mean = physics.mean_step(mean, branch)
            exact = target+E@(exact-target)@E.T
        sample_values, mean_values, exact_values = (observables(C, options) for C in (sampled, mean, exact))
        for name in FAMILIES:
            values = sample_values[name]
            se = np.zeros_like(values[:, 0]) if trajectories == 1 else values.std(axis=1, ddof=1)/np.sqrt(trajectories)
            for store, value in zip(stores, (values.mean(axis=1), se, mean_values[name], exact_values[name])):
                store[name].append(value)
    means, errors, mean_ref, exact_ref = ({name: np.stack(rows, axis=1).tolist() for name, rows in store.items()} for store in stores)
    target_values = observables(target, options)
    steady = {}
    for label, data in (("sampled", means), ("mean_channel", mean_ref), ("exact", exact_ref)):
        distance = np.maximum(np.max(np.abs(np.asarray(data["populations"])-target_values["populations"]), axis=-1),
            np.max(np.abs(np.asarray(data["excitation_flows"])-target_values["excitation_flows"]), axis=-1)/options.J)
        steady[label] = {"local_observable_distance": distance.tolist(),
            "final_within_tolerance": (distance[:, -1] <= options.steady_tolerance).tolist()}
    first = selected["checked_from_step"]
    sampling_distance = max(float(np.abs(np.asarray(means[name])[:, first:]-np.asarray(mean_ref[name])[:, first:]).max())/divisor
                            for name, divisor in (("populations", 1), ("excitation_flows", options.J)))
    uncertainty = max(float((2*np.asarray(errors[name])[:, first:]).max())/divisor
                      for name, divisor in (("populations", 1), ("excitation_flows", options.J)))
    mean_error = max(steady["mean_channel"]["local_observable_distance"][0][first:])
    mean_validation = None
    if options.validation_mode == "forward":
        continuation = selected["forward_validation"]
        state = mean[0].copy()
        validation_errors = [planning.distance(state, target, options.model)]
        for dt in continuation["step_dts"]:
            if dt not in cache:
                cache[dt] = (physics.channel(options.model, options.J*dt, options.sampling_policy,
                    options.order, options.system_formula), None)
            state = physics.mean_step(state, cache[dt][0])
            validation_errors.append(planning.distance(state, target, options.model))
        mean_error = max(validation_errors)
        if not np.allclose(validation_errors, continuation["mean_errors"], rtol=1e-10, atol=1e-12):
            raise ValueError("independent forward validation disagrees with search")
        mean_validation = {"times": continuation["times"], "local_observable_distance": validation_errors,
            "step_dts": continuation["step_dts"], "sampled": False,
            "interpretation": continuation["interpretation"]}
    sampled_error = max(steady["sampled"]["local_observable_distance"][0][first:])
    sampling_passed = trajectories > 1 and sampling_distance <= options.sampling_tolerance and uncertainty <= options.sampling_tolerance
    if not selected["tail_window_covered"]:
        status = "tail_window_not_covered"
    elif mean_error > options.mean_tolerance:
        status = "mean_target_not_met"
    elif not sampling_passed or sampled_error > options.steady_tolerance:
        status = "sampling_budget_not_met"
    else:
        status = "passed"
    boundaries, count = [], 0
    for stage in selected["stages"]:
        count += stage["steps"]
        boundaries.append([float(nodes[count]), stage["dt"]])
    physical = {name: getattr(options, name) for name in ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor")}
    return {"experiment": "Model3/Experiment4", "run_status": status,
        "configuration": {"physics": {**physical, "delta": 0.}, "trajectories": trajectories,
            "seed_trajectories": options.seed_trajectories, "steady_tolerance": options.steady_tolerance,
            "sampling_policy": options.sampling_policy, "order": options.order, "system_formula": options.system_formula,
            "schedule_input": {"mode": "budget_search", "segments": boundaries, "constant_dt": None, "step_dts": None}},
        "preparation": prep, "initial_states": [prep["name"]], "times": nodes.tolist(),
        "schedule": {"time_nodes": nodes.tolist(), "step_dts": dts, "recorded_step_counts": list(range(len(nodes))),
            "jump_choices": choices.tolist(), "jump_labels": ["left_pump", "left_loss", "right_pump", "right_loss"],
            "jump_probabilities": probabilities.tolist()},
        "observable_axis_order": ["initial_state", "time", "site_or_bond"],
        "observables": means, "trajectory_standard_errors": errors,
        "mean_channel_observables": mean_ref, "exact_reference_observables": exact_ref,
        "stationary_observables": {name: values.tolist() for name, values in target_values.items()},
        "steady_state": {"tolerance": options.steady_tolerance, **steady},
        "assessment": {"tail_window_covered": selected["tail_window_covered"],
            "validation_mode": options.validation_mode,
            "window_start": selected["window_start"], "checked_from_time": float(nodes[first]),
            "window_end": float(mean_validation["times"][-1]) if mean_validation else float(nodes[-1]),
            "tail_mean_error": float(mean_error), "tail_sampled_error": float(sampled_error),
            "output_mean_error": float(steady["mean_channel"]["local_observable_distance"][0][-1]),
            "output_sampled_error": float(steady["sampled"]["local_observable_distance"][0][-1]),
            "max_sampling_distance": sampling_distance, "max_two_standard_errors": uncertainty,
            "sampling_budget_met": bool(sampling_passed),
            "sampling_scope": "output_only" if options.validation_mode == "forward" else "trailing_window",
            "interpretation": "finite-node mean window; descriptive sampling check, not simultaneous confidence or permanent settling"},
        "mean_forward_validation": mean_validation,
        "uncertainty": {"trajectory_standard_error_estimable": trajectories > 1,
            "measurement_shot_noise": False, "hardware_noise": False,
            "note": "product mixtures are analytically averaged per path; preparation-bit and measurement shot noise are not sampled; R=1 SE is a zero placeholder"},
        "workload": {"per_trajectory_steps": len(dts), "step_budget": options.step_budget,
            "classical_forward_validation_steps": len(mean_validation["step_dts"]) if mean_validation else 0,
            "estimated_total_depth": selected["estimated_total_depth"], "depth_budget": options.depth_budget,
            "local_preparation_depth": prep["local_preparation_depth"], "total_channel_applications": trajectories*len(dts),
            "compiled_depth": None, "note": "optional depth model = local preparation layer + supplied per-step cost; no compilation performed"},
        "final_covariances_real_gauge": sampled.tolist(),
        "covariance_gauge": "physical C_ij = (-i)^i i^j M_ij; C_ij = <c_j^dagger c_i>",
        "max_mean_covariance_replay_difference": float(np.abs(mean[0]-selected["final_mean_covariance_real_gauge"]).max()),
        "figure_generation_status": "pending"}


def plot_search(search, options, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    rows = search["best_per_preparation"]
    axes[0].bar(range(len(rows)), [max(row["tail_mean_error"] if row["tail_mean_error"] is not None else row["final_mean_error"], 1e-12) for row in rows])
    axes[0].set_xticks(range(len(rows)), [row["preparation"] for row in rows], rotation=30, ha="right")
    axes[0].set(yscale="log", ylabel="Worst local-observable deviation", title="Best tested schedule per preparation")
    axes[0].axhline(options.mean_tolerance, color="black", linestyle="--", label="Mean error allowance")
    axes[0].legend()
    for name in options.preparations:
        constants = [row for row in search["constant_baselines"] if row["preparation"] == name and row["tail_mean_error"] is not None]
        line = axes[1].scatter([row["steps"] for row in constants], [max(row["tail_mean_error"], 1e-12) for row in constants], alpha=.35)
        row = next(row for row in rows if row["preparation"] == name)
        if row["tail_mean_error"] is not None:
            axes[1].scatter(row["steps"], max(row["tail_mean_error"], 1e-12), marker="*", s=100, color=line.get_facecolor()[0], label=name)
    axes[1].axhline(options.mean_tolerance, color="black", linestyle="--")
    axes[1].set(yscale="log", xlabel="Preparation steps per trajectory", ylabel="Mean validation error", title="Faint: constants; stars: best tested")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    if any(not row["tail_window_covered"] for row in rows):
        figure.suptitle("Incomplete windows use endpoint error in the left panel; these are not passing steady-state tests")
    figure.tight_layout()
    try:
        figure.savefig(folder/"preparation_comparison.png", dpi=160)
    finally:
        plt.close(figure)
    if search["selected"]["step_dts"]:
        figure, axis = plt.subplots(figsize=(10, 3.5))
        axis.stairs(search["selected"]["step_dts"], search["selected"]["time_nodes"])
        axis.set(xlabel="Physical time", ylabel="Trotter timestep", title=f"Selected schedule: {search['selected']['steps']} / {options.step_budget} steps")
        figure.tight_layout()
        try:
            figure.savefig(folder/"schedule.png", dpi=160)
        finally:
            plt.close(figure)


def main(options):
    baseline_rows, baseline_manifest = planning.load_baselines(options)
    config = configuration(options, baseline_manifest)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    directory = options.output_directory/f"budget_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Experiment4: step budget={options.step_budget}, N={options.n_qubits}; deterministic schedule/preparation search", flush=True)
    def progress(event):
        if not options.no_progress and (event["phase"] == "baselines" or event["layer"] % 5 == 0):
            print(f"Search progress: {event}", flush=True)
    started = time.perf_counter()
    search = planning.search(options, progress=progress, baselines=baseline_rows)
    search["planning_wall_seconds"] = time.perf_counter()-started
    physics.atomic_json(directory/"search.json", search)
    selected = search["selected"]
    print(f"Selected {selected['preparation']}: {selected['steps']} preparation steps, mean validation error={selected['tail_mean_error']}; {search['status']}", flush=True)
    study = {"experiment": "Model3/Experiment4", "configuration": config, "search_status": search["status"],
        "search_limit_reached": search["search_limit_reached"], "selected": selected,
        "best_per_preparation": search["best_per_preparation"], "replays": {},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "figure_generation_status": "disabled" if options.no_plots else "pending"}
    physics.atomic_json(directory/"study.json", study)
    replay_counts = [] if options.plan_only else options.R_values
    for R in replay_counts:
        started = time.perf_counter()
        result = replay(options, search, R)
        result["replay_wall_seconds"] = time.perf_counter()-started
        result["search_file"] = "search.json"
        result["figure_generation_status"] = "disabled" if options.no_plots else "pending"
        path = directory/f"replay_R{R}.json"
        physics.atomic_json(path, result)
        study["replays"][str(R)] = {"file": path.name, "status": result["run_status"], "assessment": result["assessment"]}
        physics.atomic_json(directory/"study.json", study)
        print(f"R={R}: {result['run_status']}; {path}", flush=True)
    report = ["# Experiment4: budget-constrained steady-state search", "",
        f"Step budget per trajectory: {options.step_budget}. Search status: **{search['status']}**.", "",
        f"Selected preparation: `{selected['preparation']}`; steps: {selected['steps']}; final time: {selected['final_time']:.8g}.", "",
        "Steps are a circuit-depth proxy. Local preparation cost is reported separately; compiled depth is unmeasured.", "",
        "The profile mixture uses reference stationary populations and has zero initial current/coherence. Mixed preparation-bit noise and measurement shots are analytically averaged, not included in trajectory SE.", "",
        "| Preparation | Steps | Covered mean window | Mean window error |", "|---|---:|---|---:|"]
    for row in search["best_per_preparation"]:
        value = "unavailable" if row["tail_mean_error"] is None else f"{row['tail_mean_error']:.8g}"
        report.append(f"| {row['preparation']} | {row['steps']} | {row['tail_window_covered']} | {value} |")
    if options.validation_mode == "forward":
        report.extend(["", "Forward mode: the mean is validated after output using the same final timestep. This continuation is classical planning work and is excluded from preparation depth. Sampling checks apply only to the output state.",
            f"Classical continuation: {selected['classical_validation_steps']} steps; validation ends at physical time {selected['window_end']:.8g}."])
    if options.plan_only:
        report.extend(["", "Plan only: no sampled trajectories were replayed; this is not a sampling-validated success."])
    sampled_label = "Sampled output error" if options.validation_mode == "forward" else "Sampled tail error"
    report.extend(["", f"| R | Status | {sampled_label} | Two SE |", "|---:|---|---:|---:|"])
    for R, replay_row in study["replays"].items():
        assessment = replay_row["assessment"]
        report.append(f"| {R} | {replay_row['status']} | {assessment['tail_sampled_error']:.8g} | {assessment['max_two_standard_errors']:.8g} |")
    report.extend(["", "Constant baselines, search limits, and full decision data are in `search.json`. This is the best tested candidate, not a global optimum, confidence guarantee, or permanent-settling proof."])
    (directory/"Study.md").write_text("\n".join(report)+"\n")
    if not options.no_plots:
        from Model3.Experiment3.replot_selected_observables import plot_saved, selected_indices
        folder = directory/"figures"
        folder.mkdir(exist_ok=True)
        plot_search(search, options, folder)
        for R in replay_counts:
            path = directory/f"replay_R{R}.json"
            result = json.loads(path.read_text())
            if selected["steps"]:
                plot_saved(result, selected_indices(options.n_qubits, None), selected_indices(options.n_qubits-1, None), folder/f"observables_R{R}.png")
                if options.validation_mode == "forward":
                    plot_forward_validation(result, options, folder/f"forward_validation_R{R}.png")
                result["figure_generation_status"] = "complete"
            else:
                result["figure_generation_status"] = "not_generated_no_evolution"
            physics.atomic_json(path, result)
        study["figure_generation_status"] = "complete"
        physics.atomic_json(directory/"study.json", study)
    print(f"Saved study: {directory/'study.json'}", flush=True)
    return study


def plot_forward_validation(result, options, path):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(result["times"], result["steady_state"]["mean_channel"]["local_observable_distance"][0], label="Preparation mean")
    future = result["mean_forward_validation"]
    axis.plot(future["times"], future["local_observable_distance"], linestyle="--", label="Classical mean continuation")
    axis.scatter([result["times"][-1]], [result["assessment"]["output_sampled_error"]], color="C1", zorder=4, label="Sampled output error")
    axis.axvline(result["times"][-1], color="gray", alpha=.5, label="Circuit stops")
    axis.axhline(options.mean_tolerance, color="black", linestyle=":", label="Mean allowance")
    axis.axhline(options.steady_tolerance, color="C1", linestyle=":", label="Total tolerance")
    axis.set(xlabel="Physical time", ylabel="Maximum local observable error", yscale="log",
             title=f"{result['workload']['per_trajectory_steps']} preparation steps; forward mean check")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except ValueError as error:
        raise SystemExit(str(error)) from error
