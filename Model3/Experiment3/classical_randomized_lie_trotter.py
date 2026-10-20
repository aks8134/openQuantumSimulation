"""Classical Model3 randomized Lie–Trotter with manual or automatic scheduling.

The XX covariance reduction evaluates ideal finite-step single-jump
dilation/reset channels. Each step samples one of four pump/loss channels.
Automatic scheduling uses reference-assisted mean-channel stationary checks.
No circuit simulator or hardware is used.
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
from Model3.Experiment3 import adaptive_schedule as adaptive

FAMILIES = ("populations", "magnetizations", "excitation_flows")
JUMPS = ("left_pump", "left_loss", "right_pump", "right_loss")


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def build_schedule(*, segments=None, step_dts=None, t_final=None, maximum_dt=0.1,
                   step_cap=1_000_000):
    """Return physical-time nodes; END:DT segments start at t=0.

    Segment boundaries are reached exactly with a shorter remainder step.
    Observation settings never change these integration nodes.
    """
    if not isinstance(step_cap, (int, np.integer)) or step_cap < 1:
        raise ValueError("step-cap must be a positive integer")
    if segments is not None and step_dts is not None:
        raise ValueError("choose dt-schedule or step-dts")
    if t_final is not None:
        positive(t_final, "t-final")
    if step_dts is not None:
        dts = np.asarray(step_dts, dtype=float)
        if dts.ndim != 1 or not len(dts) or np.any(~np.isfinite(dts)) or np.any(dts <= 0):
            raise ValueError("step-dts must be a nonempty list of positive finite steps")
        if len(dts) > step_cap:
            raise ValueError("schedule exceeds step-cap")
        nodes = np.r_[0.0, np.cumsum(dts)]
    else:
        if segments is None:
            positive(maximum_dt, "trotter-delta-t")
            segments = [(20.0 if t_final is None else t_final, maximum_dt)]
        if not len(segments):
            raise ValueError("dt-schedule must contain at least one END:DT pair")
        pieces, start, used = [np.array([0.0])], 0.0, 0
        for end, dt in segments:
            positive(end, "schedule endpoint")
            positive(dt, "schedule step size")
            if end <= start:
                raise ValueError("schedule endpoints must increase strictly from zero")
            quotient = (end - start) / dt
            if not np.isfinite(quotient) or quotient > step_cap - used + 1:
                raise ValueError("schedule exceeds step-cap; increase --step-cap if intended")
            full = int(np.floor(quotient))
            piece = start + np.arange(1, full + 1) * dt
            if full and np.isclose(piece[-1], end, rtol=0,
                                    atol=8*np.finfo(float).eps*max(end, dt)):
                piece[-1] = end
            else:
                piece = np.r_[piece, end]
            used += len(piece)
            if used > step_cap:
                raise ValueError("schedule exceeds step-cap")
            pieces.append(piece)
            start = end
        nodes = np.concatenate(pieces)
    if np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0):
        raise ValueError("schedule must give finite, distinguishable increasing time nodes")
    if t_final is not None and not np.isclose(nodes[-1], t_final, rtol=1e-12, atol=0):
        raise ValueError("t-final must equal the supplied schedule's final time")
    return nodes


def parse_segments(tokens):
    segments = []
    for token in tokens:
        try:
            end, dt = token.split(":")
            segments.append((float(end), float(dt)))
        except ValueError as error:
            raise ValueError(f"invalid schedule entry {token!r}; expected END:DT") from error
    return segments


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--J", "--j", dest="J", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--p-left", type=float, default=0.9)
    parser.add_argument("--p-right", type=float, default=0.1)
    parser.add_argument("--exchange-prefactor", type=float, choices=(0.25, 0.5), default=0.5)
    parser.add_argument("--initial-states", nargs="+", help="bitstrings, leftmost bit = left site; default: all zero, all one, left excitation")
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--seed-trajectories", "--seed", dest="seed_trajectories", type=int, default=29)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--dt-schedule", nargs="+", metavar="END:DT",
                          help="step size until each increasing physical-time endpoint")
    schedule.add_argument("--step-dts", nargs="+", type=float,
                          help="explicit consecutive physical-time step sizes")
    schedule.add_argument("--trotter-delta-t", type=float,
                          help="constant-step baseline (default 0.1)")
    schedule.add_argument("--auto-schedule", action="store_true",
                          help="plan a common schedule with deterministic steady-state checks, then replay trajectories")
    parser.add_argument("--t-final", type=float,
                        help="default 20 for constant steps; otherwise inferred from schedule")
    parser.add_argument("--record-every", type=int, default=1,
                        help="record every kth step; always include zero and final time")
    parser.add_argument("--step-cap", type=int, default=1_000_000)
    parser.add_argument("--sampling-policy", choices=("rate-weighted", "uniform"), default="rate-weighted")
    parser.add_argument("--order", choices=("hamiltonian-first", "jump-first"), default="hamiltonian-first")
    parser.add_argument("--system-formula", choices=("even-odd", "exact"), default="even-odd")
    parser.add_argument("--steady-tolerance", type=float, default=0.01)
    controller = parser.add_argument_group("automatic schedule controls (require --auto-schedule)")
    controller.add_argument("--initial-dt", type=float, help="default 0.5/max(J,gamma)")
    controller.add_argument("--minimum-dt", type=float, help="minimum nominal step; default 1e-5/max(J,gamma)")
    controller.add_argument("--check-interval", type=float, help="physical-time interval between decisions; default 5/J")
    controller.add_argument("--steady-window", type=float, help="passing-window duration; default twice check-interval")
    controller.add_argument("--max-time", type=float, help="automatic physical-time limit; default 200/J")
    controller.add_argument("--bias-tolerance", type=float, help="default 0.3*steady-tolerance")
    controller.add_argument("--relaxation-tolerance", type=float, help="default 0.3*steady-tolerance")
    controller.add_argument("--sampling-tolerance", type=float, help="default 0.4*steady-tolerance")
    controller.add_argument("--refine-factor", type=float,
                            help="refine when relaxation <= max(relaxation budget, factor*bias); default 1")
    parser.add_argument("--output-directory", "--output-dir", type=Path, default=DIRECTORY)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    options = parser.parse_args(arguments)
    if options.n_qubits < 2:
        raise ValueError("n-qubits must be at least two")
    positive(options.J, "J")
    positive(options.gamma, "gamma")
    positive(options.steady_tolerance, "steady-tolerance")
    if not np.isfinite(options.gamma / options.J):
        raise ValueError("gamma/J must be finite")
    for name, value in (("p-left", options.p_left), ("p-right", options.p_right)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie between zero and one")
    if options.trajectories < 1 or options.seed_trajectories < 0 or options.record_every < 1:
        raise ValueError("trajectories and record-every must be positive; seed must be nonnegative")
    if options.initial_states is None:
        options.initial_states = ["0"*options.n_qubits, "1"*options.n_qubits,
                                 "1"+"0"*(options.n_qubits-1)]
    if (len(set(options.initial_states)) != len(options.initial_states)
            or any(len(label) != options.n_qubits or set(label) - {"0", "1"}
                   for label in options.initial_states)):
        raise ValueError("initial-states must be distinct n-qubits-long binary strings")
    options.segments = None if options.dt_schedule is None else parse_segments(options.dt_schedule)
    options.model = physics.Model(options.n_qubits, options.gamma/options.J,
        options.p_left, options.p_right, options.exchange_prefactor)
    if options.step_cap < 1:
        raise ValueError("step-cap must be positive")
    if options.auto_schedule:
        if options.t_final is not None:
            raise ValueError("use --max-time rather than --t-final with --auto-schedule")
        options.controller_settings = adaptive.settings_from_options(options)
        # Planning is deliberately deferred to main(); parsing does not evolve.
        options.nodes = np.array([0.0])
    else:
        names = ("initial_dt", "minimum_dt", "check_interval", "steady_window", "max_time",
                 "bias_tolerance", "relaxation_tolerance", "sampling_tolerance", "refine_factor")
        if any(getattr(options, name) is not None for name in names):
            raise ValueError("automatic controller options require --auto-schedule")
        options.nodes = build_schedule(segments=options.segments, step_dts=options.step_dts,
            t_final=options.t_final, maximum_dt=0.1 if options.trotter_delta_t is None else options.trotter_delta_t,
            step_cap=options.step_cap)
    scaled_steps = options.J * np.diff(options.nodes)
    if np.any(~np.isfinite(scaled_steps)) or np.any(scaled_steps <= 0):
        raise ValueError("J times each step must be positive and finite")
    return options


def observables(covariances, options):
    """Covariance is in the real gauge used by reservoir_scaling."""
    populations = np.diagonal(covariances, axis1=-2, axis2=-1)
    currents = 4*options.exchange_prefactor*options.J*np.diagonal(
        covariances, offset=1, axis1=-2, axis2=-1)
    return dict(zip(FAMILIES, (populations, 1-2*populations, currents)))


def standard_error(values):
    count = values.shape[1]
    return np.zeros_like(values[:, 0]) if count == 1 else values.std(axis=1, ddof=1)/np.sqrt(count)


def configuration(options):
    config = {"schema_version": 1, "physics": {"n_qubits": options.n_qubits,
        "J": options.J, "gamma": options.gamma, "p_left": options.p_left,
        "p_right": options.p_right, "exchange_prefactor": options.exchange_prefactor,
        "delta": 0.0}, "initial_states": options.initial_states,
        "trajectories": options.trajectories, "seed_trajectories": options.seed_trajectories,
        "sampling_policy": options.sampling_policy, "order": options.order,
        "system_formula": options.system_formula, "time_nodes": options.nodes.tolist(),
        "record_every": options.record_every, "steady_tolerance": options.steady_tolerance,
        "schedule_input": {"segments": options.segments, "step_dts": options.step_dts,
            "constant_dt": (0.1 if options.trotter_delta_t is None else options.trotter_delta_t)
                           if options.segments is None and options.step_dts is None else None}}
    if options.auto_schedule:
        config.update(schema_version=2, automatic_controller=options.controller_settings.payload(),
                      schedule_input={"mode": "automatic", "segments": None, "step_dts": None, "constant_dt": None})
    return config


def evolve(options, progress=None):
    nodes, model = options.nodes, options.model
    dts = np.diff(nodes)
    counts = np.unique(np.r_[np.arange(0, len(nodes), options.record_every), len(dts)])
    controller = getattr(options, "controller_plan", None)
    if controller is not None:
        # Preserve decision nodes and every tail node for sampling assessment,
        # even when transient output is thinned with record-every.
        required = [entry["step_count"] for entry in controller["checkpoints"]]
        if controller["window_start"] is not None:
            required.extend(np.flatnonzero(nodes >= controller["window_start"]-1e-10).tolist())
        counts = np.unique(np.r_[counts, np.asarray(required, dtype=int)])
    initial = np.asarray([np.diag([int(bit) for bit in label])
                          for label in options.initial_states], dtype=float)
    sampled = np.repeat(initial[:, None], options.trajectories, axis=1)
    averaged, exact = initial.copy(), initial.copy()
    target = physics.exact_stationary(model)
    # Independent RNG per path preserves existing paths when R increases.
    streams = [np.random.default_rng(np.random.SeedSequence([options.seed_trajectories, path]))
               for path in range(options.trajectories)]
    probabilities = (np.full(4, .25) if options.sampling_policy == "uniform" else
        np.array([options.p_left, 1-options.p_left, options.p_right, 1-options.p_right])/2)
    choices = np.empty((options.trajectories, len(dts)), dtype=np.uint8)
    means, errors, mean_reference, exact_reference = ({name: [] for name in FAMILIES} for _ in range(4))
    sampled_local, mean_local, exact_local = [], [], []
    target_observables = observables(target, options)
    record_index, blocks, cache = 0, None, {}

    def local_error(values):
        return np.maximum(np.max(np.abs(values["populations"]-target_observables["populations"]), axis=-1),
            np.max(np.abs(values["excitation_flows"]-target_observables["excitation_flows"]), axis=-1)/options.J)

    for step in range(len(nodes)):
        if step:
            dt = float(dts[step-1])
            if dt not in cache:
                cache[dt] = (physics.channel(model, options.J*dt, options.sampling_policy,
                                            options.order, options.system_formula),
                             expm(options.J*dt*physics.generator(model)))
            branch, E = cache[dt]
            offset = (step-1) % 1024
            if offset == 0:
                blocks = physics.draw_choice_block(streams, probabilities)
            selected = blocks[:, offset]
            choices[:, step-1] = selected
            sampled = physics.sampled_step(sampled, selected, branch)
            averaged = physics.mean_step(averaged, branch)
            exact = target + E @ (exact-target) @ E.T
        if step == counts[record_index]:
            sampled_values = observables(sampled, options)
            averaged_values, exact_values = observables(averaged, options), observables(exact, options)
            sample_means = {name: values.mean(axis=1) for name, values in sampled_values.items()}
            for name in FAMILIES:
                means[name].append(sample_means[name])
                errors[name].append(standard_error(sampled_values[name]))
                mean_reference[name].append(averaged_values[name])
                exact_reference[name].append(exact_values[name])
            sampled_local.append(local_error(sample_means))
            mean_local.append(local_error(averaged_values))
            exact_local.append(local_error(exact_values))
            record_index += 1
            if progress is not None:
                progress(record_index, len(counts), float(nodes[step]))
    convert = lambda data: {name: np.stack(rows, axis=1).tolist() for name, rows in data.items()}
    mean_data, errors_data = convert(means), convert(errors)
    mean_ref, exact_ref = convert(mean_reference), convert(exact_reference)
    all_errors = {}
    for label, left, right in (("sampled_vs_exact", mean_data, exact_ref),
                               ("mean_channel_vs_exact", mean_ref, exact_ref),
                               ("sampled_vs_mean_channel", mean_data, mean_ref)):
        all_errors[label] = {name: float(np.max(np.abs(np.asarray(left[name])-right[name])))
                            for name in FAMILIES}
    steady = {}
    for name, rows in (("sampled", sampled_local), ("mean_channel", mean_local), ("exact", exact_local)):
        distances = np.stack(rows, axis=1)
        steady[name] = {"local_observable_distance": distances.tolist(),
            "final_within_tolerance": (distances[:, -1] <= options.steady_tolerance).tolist(),
            "first_sampled_time_within_tolerance": [next((float(nodes[counts[i]])
                for i, value in enumerate(row) if value <= options.steady_tolerance), None) for row in distances]}
    return {"experiment": "Model3/Experiment3/classical", "configuration": configuration(options),
        "initial_states": options.initial_states, "times": nodes[counts].tolist(),
        "schedule": {"time_nodes": nodes.tolist(), "step_dts": dts.tolist(),
            "recorded_step_counts": counts.tolist(), "jump_labels": list(JUMPS),
            "jump_probabilities": probabilities.tolist(), "jump_choices": choices.tolist()},
        "observable_axis_order": ["initial_state", "recorded_time", "site_or_bond"],
        "observables": mean_data, "trajectory_standard_errors": errors_data,
        "mean_channel_observables": mean_ref, "exact_reference_observables": exact_ref,
        "stationary_observables": {name: values.tolist() for name, values in target_observables.items()},
        "max_absolute_observable_errors": all_errors,
        "steady_state": {"tolerance": options.steady_tolerance,
            "distance_metric": "max(max site population error, max bond current error / J) against physical stationary observables",
            "interpretation": "sampled local-observable checks, not full-state trace distance or a permanent-settling guarantee", **steady},
        "uncertainty": {"trajectory_standard_error_estimable": options.trajectories > 1,
            "note": "sample standard deviation / sqrt(R); R=1 uses zero placeholders",
            "measurement_shot_noise": False, "hardware_noise": False},
        "final_covariances_real_gauge": sampled.tolist(),
        "covariance_gauge": "physical C_ij = (-i)^i i^j M_ij; C_ij = <c_j^dagger c_i>",
        "figure_generation_status": "pending"}


def plot_result(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(payload["times"])
    preparations = payload["initial_states"]
    figure, axes = plt.subplots(len(preparations), 3, figsize=(17, 4*len(preparations)), squeeze=False)
    for prep, label in enumerate(preparations):
        for column, (name, title) in enumerate((("populations", "Site populations"),
                                                ("excitation_flows", "Bond excitation current"))):
            axis = axes[prep, column]
            means = np.asarray(payload["observables"][name])[prep]
            errors = np.asarray(payload["trajectory_standard_errors"][name])[prep]
            exact = np.asarray(payload["exact_reference_observables"][name])[prep]
            averaged = np.asarray(payload["mean_channel_observables"][name])[prep]
            for index in sorted({0, (means.shape[-1]-1)//2, means.shape[-1]-1}):
                line, = axis.plot(times, means[:, index], label=f"{'site' if column == 0 else 'bond'} {index}")
                color = line.get_color()
                axis.fill_between(times, means[:, index]-errors[:, index], means[:, index]+errors[:, index], color=color, alpha=.15)
                axis.plot(times, exact[:, index], "--", color=color, alpha=.8)
                axis.plot(times, averaged[:, index], ":", color=color, alpha=.8)
                axis.axhline(payload["stationary_observables"][name][index], color=color, alpha=.2)
            axis.set_title(f"{label}: {title}")
            axis.legend(fontsize="small")
        axis = axes[prep, 2]
        for name, style in (("sampled", "-"), ("mean_channel", ":"), ("exact", "--")):
            values = payload["steady_state"][name]["local_observable_distance"][prep]
            axis.semilogy(times, np.maximum(values, np.finfo(float).tiny), style, label=name)
        axis.axhline(payload["steady_state"]["tolerance"], color="black", linestyle=":", label="tolerance")
        axis.set_title(f"{label}: Local distance to stationary observables")
        axis.legend(fontsize="small")
    for axis in axes.flat:
        axis.set_xlabel("Physical time")
        axis.grid(alpha=.3)
    cfg = payload["configuration"]
    model = cfg["physics"]
    mode = "automatic" if cfg["schedule_input"].get("mode") == "automatic" else "manual"
    status = (f"\nController: {payload['automatic_controller']['status']}; sampling: {payload['sampling_assessment']['status']}"
              if "automatic_controller" in payload else "")
    figure.suptitle(f"Model3 randomized Lie–Trotter (selected sites/bonds): N={model['n_qubits']}, J={model['J']}, γ={model['gamma']}, R={cfg['trajectories']}\n"
                    "solid = sampled mean ± SE; dashed = exact Lindblad; dotted = mean channel"+status)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 4))
    nodes = payload["schedule"]["time_nodes"]
    dts = payload["schedule"]["step_dts"]
    axis.stairs(dts, nodes)
    axis.set(xlabel="Physical time", ylabel="Trotter step size", title=f"Executed {mode} timestep schedule")
    axis.grid(alpha=.3)
    figure.tight_layout()
    try:
        figure.savefig(path.with_name(path.stem.replace("_observables", "_schedule")+".png"), dpi=160)
    finally:
        plt.close(figure)


def main(options):
    planning_seconds = 0.0
    if options.auto_schedule:
        def controller_progress(entry):
            if options.no_progress:
                return
            if entry["event"] == "candidate":
                detail = f"bias={entry['bias']:.6g}" if entry["status"] == "valid" else entry["reason"]
                print(f"Candidate dt={entry['dt']:g}: {detail}", flush=True)
            else:
                print(f"Controller t={entry['time']:g}, dt={entry['dt']:g}: {entry['action']}; "
                      f"relaxation={entry['relaxation_distance']:.6g}, bias={entry['bias']:.6g}", flush=True)
        print("Planning automatic schedule using deterministic mean channels", flush=True)
        initial = np.asarray([np.diag([int(bit) for bit in label]) for label in options.initial_states], dtype=float)
        started = time.perf_counter()
        options.nodes, options.controller_plan = adaptive.plan(options.model, initial, options.J,
            options.controller_settings, policy=options.sampling_policy, order=options.order,
            formula=options.system_formula, step_cap=options.step_cap, progress=controller_progress)
        planning_seconds = time.perf_counter()-started
    cfg = configuration(options)
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    stem = f"classical_randomized_N{options.n_qubits}_J{options.J:g}_g{options.gamma:g}_R{options.trajectories}_{digest}"
    result = options.output_directory / "results" / f"{stem}.json"
    figure = options.output_directory / "figures" / f"{stem}_observables.png"
    if options.auto_schedule:
        plan_path = options.output_directory / "plans" / f"{stem}_plan.json"
        physics.atomic_json(plan_path, {"configuration": cfg, "automatic_controller": options.controller_plan,
                                       "planning_wall_seconds": planning_seconds})
        print(f"Controller: {options.controller_plan['status']}; saved plan: {plan_path}", flush=True)
    mode = "automatic" if options.auto_schedule else "manual"
    print(f"Model3 {mode} schedule: N={options.n_qubits}, J={options.J:g}, gamma={options.gamma:g}, "
          f"R={options.trajectories}, steps={len(options.nodes)-1}, t_final={options.nodes[-1]:g}", flush=True)
    def progress(index, count, t):
        if index == 1 or index == count or index % max(1, count//20) == 0:
            print(f"Sample {index}/{count}: t={t:g}", flush=True)
    started = time.perf_counter()
    payload = evolve(options, None if options.no_progress else progress)
    payload["simulation_wall_seconds"] = time.perf_counter()-started
    if options.auto_schedule:
        payload["planning_wall_seconds"] = planning_seconds
        payload["automatic_controller"] = options.controller_plan
        payload["sampling_assessment"] = adaptive.assess_sampling(payload, options.controller_plan)
        status = options.controller_plan["status"]
        payload["run_status"] = payload["sampling_assessment"]["status"] if status == "converged" else status
        planned = observables(np.asarray(options.controller_plan["final_mean_covariances_real_gauge"]), options)
        differences = {name: float(np.max(np.abs(np.asarray(payload["mean_channel_observables"][name])[:, -1]-planned[name])))
                       for name in FAMILIES}
        payload["automatic_controller"]["observable_replay_differences"] = differences
        payload["automatic_controller"]["max_population_replay_difference"] = differences["populations"]
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    physics.atomic_json(result, payload)
    if options.no_plots:
        payload["figure_generation_status"] = "disabled"
    elif options.auto_schedule and len(options.nodes) == 1:
        payload["figure_generation_status"] = "not_generated_no_evolution"
    else:
        plot_result(payload, figure)
        payload["figure_generation_status"] = "complete"
        payload["figures"] = {"observables": str(figure),
            "schedule": str(figure.with_name(figure.stem.replace("_observables", "_schedule")+".png"))}
    physics.atomic_json(result, payload)
    print(f"Saved results: {result}", flush=True)
    if payload["figure_generation_status"] == "complete":
        print(f"Saved figures: {figure.parent}", flush=True)
    print("Maximum sampled population error vs exact: "
          f"{payload['max_absolute_observable_errors']['sampled_vs_exact']['populations']:.6g}", flush=True)
    if options.auto_schedule:
        print(f"Automatic run status: {payload['run_status']}; sampling: {payload['sampling_assessment']['status']}", flush=True)
        print(f"Mean controller: {options.controller_plan['stop_reason']}", flush=True)
        assessment = payload["sampling_assessment"]
        if assessment["status"] != "not_assessed":
            print(f"Sampling window [{assessment['window_start']:g}, {assessment['window_end']:g}]: "
                  f"realized discrepancy={assessment['max_realized_sampling_distance']:.6g}, "
                  f"2 SE={assessment['max_two_standard_errors']:.6g}, "
                  f"budget={options.controller_settings.sampling_tolerance:g}", flush=True)
            if assessment["reasons"]:
                print("Sampling outcome: "+"; ".join(assessment["reasons"]), flush=True)
    return payload


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except ValueError as error:
        raise SystemExit(str(error)) from error
