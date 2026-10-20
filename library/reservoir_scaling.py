"""Classical, covariance-sector scaling studies for boundary-driven XX chains.

Pure numerical functions implement exact references and genuine sampled
single-jump dilations. CLI/persistence are separate orchestration functions.
No density-matrix truncation, circuit simulator, hardware, or QEM is used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import LinearOperator, gmres


class Model(NamedTuple):
    n: int
    ratio: float
    p_left: float
    p_right: float
    prefactor: float


class Channel(NamedTuple):
    unitary: np.ndarray
    weights: np.ndarray
    injection: np.ndarray
    probabilities: np.ndarray
    cosines: np.ndarray
    order: str


def initial_covariances(n):
    """Preparations |00...0>, |11...1>, |10...0>; left site first."""
    single = np.zeros((n, n))
    single[0, 0] = 1
    return np.asarray([np.zeros((n, n)), np.eye(n), single])


def generator(model):
    """Real gauge C_ij=(-i)^i i^j M_ij, dimensionless J=1."""
    a = 2 * model.prefactor
    G = np.diag(np.full(model.n - 1, -a), 1) + np.diag(np.full(model.n - 1, a), -1)
    G[0, 0] = G[-1, -1] = -model.ratio / 2
    return G


def exact_stationary(model):
    source = np.zeros((model.n, model.n))
    source[0, 0] = model.ratio * model.p_left
    source[-1, -1] = model.ratio * model.p_right
    target = solve_continuous_lyapunov(generator(model), -source)
    return (target + target.T) / 2


def exact_at(model, tau, initial=None):
    initial = initial_covariances(model.n) if initial is None else initial
    target = exact_stationary(model)
    E = expm(tau * generator(model))
    return target + E @ (initial - target) @ E.T


def observable_error(left, right, prefactor):
    """Max absolute population error and current error/J, all preparations."""
    difference = np.asarray(left) - np.asarray(right)
    populations = float(np.max(np.abs(np.diagonal(difference, axis1=-2, axis2=-1))))
    currents = float(4 * prefactor * np.max(np.abs(
        np.diagonal(difference, offset=1, axis1=-2, axis2=-1))))
    return {"population": populations, "current_divided_by_J": currents}


def envelope(covariances, target, prefactor):
    """Contractive upper bound on local errors, NOT many-body trace distance."""
    return float(max(1, 4 * prefactor) * np.max(np.linalg.norm(
        np.asarray(covariances) - target, axis=(-2, -1))))


def exact_crossing(model, tolerance, cap):
    target = exact_stationary(model)
    initial = initial_covariances(model.n)
    def distance(t):
        E = expm(t * generator(model))
        return envelope(target + E @ (initial - target) @ E.T, target, model.prefactor)
    if distance(0) <= tolerance:
        return 0.0
    upper = min(1.0, cap)
    while distance(upper) > tolerance and upper < cap:
        upper = min(2 * upper, cap)
    if distance(upper) > tolerance:
        return None
    return float(brentq(lambda t: distance(t) - tolerance, 0, upper, xtol=1e-9))


def system_unitary(model, step, formula):
    hopping = generator(model).copy()
    np.fill_diagonal(hopping, 0)
    if formula == "exact":
        return expm(step * hopping)
    if formula != "even-odd":
        raise ValueError("system formula must be exact or even-odd")
    U = np.eye(model.n)
    cosine, sine = np.cos(2 * model.prefactor * step), np.sin(2 * model.prefactor * step)
    for parity in (0, 1):
        for j in range(parity, model.n - 1, 2):
            left, right = U[j].copy(), U[j + 1].copy()
            U[j], U[j + 1] = cosine * left - sine * right, sine * left + cosine * right
    return U


def channel(model, step, policy="rate-weighted", order="jump-first", formula="exact"):
    """Four random alternatives, one jump per step; never four sequential jumps.

    A reset-zero two-level dilation gives cos(sqrt(rate*s/mu)), not an
    exponential damping factor. Signed cosines are retained.
    """
    if step <= 0 or not np.isfinite(step) or policy not in ("uniform", "rate-weighted"):
        raise ValueError("invalid step or policy")
    if order not in ("jump-first", "hamiltonian-first"):
        raise ValueError("invalid factor order")
    fractions = np.array([model.p_left, 1 - model.p_left, model.p_right, 1 - model.p_right])
    probabilities = np.full(4, 0.25) if policy == "uniform" else fractions / 2
    # Zero-rate jumps are never selected by rate-weighted sampling. Their
    # unused branches are identities, including at p_left/right = 0 or 1.
    angles_squared = np.divide(model.ratio * fractions * step, probabilities,
                               out=np.zeros(4), where=probabilities > 0)
    cosines = np.cos(np.sqrt(angles_squared))
    masks = np.ones((4, model.n))
    injection = np.zeros((model.n, model.n))
    for k, boundary in enumerate((0, 0, model.n - 1, model.n - 1)):
        masks[k, boundary] = cosines[k]
        if k % 2 == 0:
            injection[boundary, boundary] += probabilities[k] * (1 - cosines[k]**2)
    weights = np.einsum("k,ki,kj->ij", probabilities, masks, masks)
    return Channel(system_unitary(model, step, formula), weights, injection,
                   probabilities, cosines, order)


def mean_step(C, branch, *, affine=True):
    U, weights, B = branch.unitary, branch.weights, branch.injection
    if branch.order == "jump-first":
        return U @ (weights * C + (B if affine else 0)) @ U.T
    return weights * (U @ C @ U.T) + (B if affine else 0)


def sampled_step(C, choices, branch):
    """C has shape (preparation, R, N, N); all R paths evolve independently."""
    choices = np.asarray(choices, dtype=int)
    if np.any((choices < 0) | (choices > 3)):
        raise ValueError("jump choices must be 0, 1, 2 or 3")
    if C.shape[-3] != len(choices):
        raise ValueError("one jump choice is needed per path")
    U = branch.unitary
    C = U @ C @ U.T if branch.order == "hamiltonian-first" else C.copy()
    boundaries = np.where(choices < 2, 0, C.shape[-1] - 1)
    columns = np.arange(len(choices))
    factors = branch.cosines[choices]
    C[:, columns, boundaries, :] *= factors[None, :, None]
    # Separated NumPy advanced indices move the path axis to the front here.
    C[:, columns, :, boundaries] *= factors[:, None, None]
    C[:, columns, boundaries, boundaries] += np.where(choices % 2 == 0, 1 - factors**2, 0)
    return U @ C @ U.T if branch.order == "jump-first" else C


def draw_choice_block(streams, probabilities):
    """Fixed-size draws per independent path preserve prefixes when R increases."""
    uniforms = np.asarray([stream.random(1024) for stream in streams])
    return np.searchsorted(np.cumsum(probabilities), uniforms, side="right")


def mean_stationary(branch, n):
    """Matrix-free affine fixed-point solve; residual and positivity are audited."""
    source = mean_step(np.zeros((n, n)), branch)
    operator = LinearOperator((n*n, n*n), dtype=float,
        matvec=lambda x: (x.reshape(n, n) - mean_step(x.reshape(n, n), branch, affine=False)).ravel())
    vector, info = gmres(operator, source.ravel(), rtol=1e-10, atol=1e-12,
                         restart=min(n*n, 80), maxiter=200)
    if info != 0:
        raise ValueError(f"averaged stationary solve did not converge (GMRES info={info})")
    stationary = vector.reshape(n, n)
    stationary = (stationary + stationary.T) / 2
    residual = float(np.linalg.norm(mean_step(stationary, branch) - stationary))
    values = np.linalg.eigvalsh(stationary)
    if residual > 1e-9 or values[0] < -1e-7 or values[-1] > 1 + 1e-7:
        raise ValueError("averaged fixed point failed residual/physicality checks")
    return stationary, {"residual": residual, "minimum_eigenvalue": float(values[0]),
                        "maximum_eigenvalue": float(values[-1])}


def update_maxima(previous, new):
    return {key: max(previous[key], new[key]) for key in previous}


def evaluate_step(model, step, exact_time, config):
    started = time.perf_counter()
    branch = channel(model, step, config["policy"], config["order"], config["system_formula"])
    target = exact_stationary(model)
    base = {"N": model.n, "ratio": model.ratio, "dimensionless_step": step,
            "horizon": config["horizon_factor"] * exact_time}
    try:
        stationary, checks = mean_stationary(branch, model.n)
    except ValueError as error:
        return {**base, "status": "stationary_solve_failed", "error": str(error),
                "passed_infinite_R": False, "passed_recommended": False}
    bias = max(observable_error(stationary, target, model.prefactor).values())
    initial = initial_covariances(model.n)
    mean, exact = initial.copy(), initial.copy()
    E = expm(step * generator(model))
    peaks = {"population": 0.0, "current_divided_by_J": 0.0}
    crossings = {"infinite_R": None, "recommended": None}
    # Same permanent settling bound for both dynamic budgets.
    crossing = None
    total_steps = int(np.ceil(base["horizon"] / step))
    tail_bound, status, curve = None, "complete", []
    stride = max(1, total_steps // 250)
    completed = 0
    for k in range(min(total_steps, config["step_cap"]) + 1):
        completed = k
        error = observable_error(mean, exact, model.prefactor)
        peaks = update_maxima(peaks, error)
        settling_bound = bias + envelope(mean, stationary, model.prefactor)
        if crossing is None and settling_bound <= config["steady_tolerance"]:
            crossing = float(k * step)
        if k % stride == 0 or k == total_steps:
            curve.append({"tau": k * step, "error": max(error.values()),
                          "steady_bound": settling_bound})
        if max(peaks.values()) > config["dynamic_tolerance"]:
            status = "transient_error_rejected"
            break
        if bias > config["steady_tolerance"]:
            status = "stationary_bias_rejected"
            break
        bound = bias + envelope(mean, stationary, model.prefactor) + envelope(exact, target, model.prefactor)
        if crossing is not None and bound <= config["bias_tolerance"]:
            tail_bound = bound
            break  # Future-node accuracy follows by contractivity, including beyond horizon.
        if k == total_steps or k == config["step_cap"]:
            break
        mean = mean_step(mean, branch)
        exact = target + E @ (exact - target) @ E.T
        if config.get("progress_every", 0) and (k + 1) % config["progress_every"] == 0:
            print(f"    {k+1}/{total_steps} mean steps; error={max(peaks.values()):.5g}", flush=True)
    covered = completed == total_steps or tail_bound is not None
    if status == "complete" and not covered:
        status = "step_cap_reached"
    timing_error = None if crossing is None else abs(crossing - exact_time) / max(exact_time, 1e-300)
    for label, error_limit, timing_limit in (
            ("infinite_R", config["dynamic_tolerance"], config["timing_tolerance"]),
            ("recommended", config["bias_tolerance"], config["timing_tolerance"] / 2)):
        crossings[label] = bool(covered and status == "complete" and crossing is not None
            and max(peaks.values()) <= error_limit and bias <= config["steady_tolerance"]
            and timing_error <= timing_limit)
    return {**base, "status": status, "completed_steps": completed, "max_errors": peaks,
            "stationary_bias": bias, "stationary_checks": checks,
            "steady_time_bracket": None if crossing is None else [max(0, crossing-step), crossing],
            "relative_steady_time_error": timing_error, "tail_error_bound": tail_bound,
            "passed_infinite_R": crossings["infinite_R"], "passed_recommended": crossings["recommended"],
            "averaged_stationary_covariance": stationary.tolist(), "curve": curve,
            "wall_seconds": time.perf_counter() - started,
            "error_audit": "all evaluated nodes; unvisited tail only accepted with contractive upper bound"}


def evaluate_ensemble(model, step, exact_time, R, config):
    """Each seed must pass throughout a finite horizon; never certify an infinite noisy tail."""
    # Dense covariances, matrix products, and outputs: conservative working-set estimate.
    memory_mb = 8 * 3 * R * model.n**2 * 6 / 1024**2
    if memory_mb > config["memory_limit_mb"]:
        return {"R": R, "passed_all_seeds": False, "status": "memory_cap_reached",
                "estimated_memory_mb": memory_mb, "per_seed": []}
    branch = channel(model, step, config["policy"], config["order"], config["system_formula"])
    target = exact_stationary(model)
    E = expm(step * generator(model))
    horizon = config["horizon_factor"] * exact_time
    total_steps = int(np.ceil(horizon / step))
    results = []
    for seed in config["seeds"]:
        print(f"      R={R}, seed={seed}: {total_steps} steps", flush=True)
        initial = initial_covariances(model.n)
        paths = np.repeat(initial[:, None], R, axis=1)
        mean, exact = initial.copy(), initial.copy()
        streams = [np.random.default_rng(np.random.SeedSequence([seed, path])) for path in range(R)]
        blocks = None
        peaks = {"population": 0.0, "current_divided_by_J": 0.0}
        sampling = peaks.copy()
        tail_peak, status = 0.0, "complete"
        last_outside = None
        completed = 0
        for k in range(min(total_steps, config["step_cap"]) + 1):
            completed = k
            estimate = paths.mean(axis=1)
            peaks = update_maxima(peaks, observable_error(estimate, exact, model.prefactor))
            sampling = update_maxima(sampling, observable_error(estimate, mean, model.prefactor))
            steady_error = max(observable_error(estimate, target, model.prefactor).values())
            if steady_error > config["steady_tolerance"]:
                last_outside = k
            if k * step >= exact_time * (1 + config["timing_tolerance"]):
                tail_peak = max(tail_peak, steady_error)
            if max(peaks.values()) > config["dynamic_tolerance"] or max(sampling.values()) > config["sampling_tolerance"]:
                status = "transient_error_rejected"
                break
            if config["finite_R_steady_check"] and tail_peak > config["steady_tolerance"]:
                status = "finite_window_steady_rejected"
                break
            if k == total_steps or k == config["step_cap"]:
                break
            if k % 1024 == 0:
                blocks = draw_choice_block(streams, branch.probabilities)
            choices = blocks[:, k % 1024]
            paths = sampled_step(paths, choices, branch)
            mean = mean_step(mean, branch)
            exact = target + E @ (exact - target) @ E.T
            if config["progress_every"] and (k + 1) % config["progress_every"] == 0:
                print(f"        {k+1}/{total_steps} steps", flush=True)
        if status == "complete" and completed < total_steps:
            status = "step_cap_reached"
        record = {"seed": seed, "passed": status == "complete", "status": status,
                  "completed_steps": completed, "max_errors_vs_exact": peaks,
                  "max_errors_vs_average": sampling, "maximum_post_deadline_steady_error": tail_peak,
                  "finite_window_settling_time": None if last_outside == completed else
                      (0.0 if last_outside is None else (last_outside + 1) * step)}
        results.append(record)
        if not record["passed"]:
            break  # One failing seed suffices; unvisited seeds are NOT labeled as failed.
    return {"R": R, "passed_all_seeds": len(results) == len(config["seeds"]) and all(r["passed"] for r in results),
            "status": "complete" if len(results) == len(config["seeds"]) and all(r["passed"] for r in results) else "rejected_or_capped",
            "horizon": total_steps * step, "per_seed": results,
            "untested_seeds": config["seeds"][len(results):],
            "RNG": "independent SeedSequence([seed,path]), fixed 1024-draw blocks; stable R prefixes"}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def parse_arguments(arguments=None, *, model_name, directory, unequal=False):
    parser = argparse.ArgumentParser(description=f"{model_name}: classical XX randomized-dilation scaling study")
    parser.add_argument("--sizes", type=int, nargs="+", default=(2,4,7,10,15,20,30,50))
    parser.add_argument("--ratios", type=float, nargs="+", default=(.25,.5,1,1.5,2,2.5,3,4,6,8))
    parser.add_argument("--steps", type=float, nargs="+", default=(.1,.08,.06,.04,.03,.02,.015,.01,.0075,.005,.003,.002,.001,.0005))
    parser.add_argument("--R-values", type=int, nargs="+", default=(4,8,16,32,64,128,256))
    parser.add_argument("--extend-R-values", type=int, nargs="+", default=())
    parser.add_argument("--seeds", type=int, nargs="+", default=(29,101,307,509,701))
    if unequal:
        parser.add_argument("--p-left", type=float, default=.9)
        parser.add_argument("--p-right", type=float, default=.1)
    else:
        parser.add_argument("--p", type=float, default=.25)
    parser.add_argument("--exchange-prefactor", type=float, default=.5 if unequal else .25)
    parser.add_argument("--policy", choices=("uniform", "rate-weighted"), default="rate-weighted")
    parser.add_argument("--order", choices=("jump-first", "hamiltonian-first"), default="jump-first")
    parser.add_argument("--system-formula", choices=("exact", "even-odd"), default="exact")
    parser.add_argument("--steady-tolerance", type=float, default=1e-3)
    parser.add_argument("--dynamic-tolerance", type=float, default=.01)
    parser.add_argument("--timing-tolerance", type=float, default=.05)
    parser.add_argument("--target-time", type=float, default=20)
    parser.add_argument("--rate-margin", type=float, default=1.1)
    parser.add_argument("--horizon-factor", type=float, default=1.5)
    parser.add_argument("--promising-ratios", type=int, default=2)
    parser.add_argument("--step-refinements", type=int, default=2)
    parser.add_argument("--time-cap", type=float, default=100000)
    parser.add_argument("--step-cap", type=int, default=3000000)
    parser.add_argument("--memory-limit-mb", type=float, default=1024)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--no-finite-R-steady-check", action="store_true",
                        help="select R by transient/sampling error only, without a finite-window steady-tail test")
    parser.add_argument("--output-directory", type=Path, default=Path(directory) / "study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    options = parser.parse_args(arguments)
    cfg = {key: getattr(options, key) for key in (
        "policy", "order", "system_formula", "steady_tolerance", "dynamic_tolerance", "timing_tolerance",
        "target_time", "rate_margin", "horizon_factor", "promising_ratios", "time_cap", "step_cap", "memory_limit_mb")}
    cfg.update(schema_version=1, model_name=model_name, delta=0,
        p_left=options.p_left if unequal else options.p, p_right=options.p_right if unequal else options.p,
        exchange_prefactor=options.exchange_prefactor, finite_R_steady_check=not options.no_finite_R_steady_check,
        sizes=sorted(set(options.sizes)), ratios=sorted(set(options.ratios)),
        steps=sorted(set(options.steps), reverse=True), R_values=sorted(set(options.R_values)), seeds=list(options.seeds),
        bias_tolerance=options.dynamic_tolerance/2, sampling_tolerance=options.dynamic_tolerance/2)
    if min(cfg["sizes"]) < 2 or min(cfg["R_values"]) < 1 or min(cfg["seeds"]) < 0 or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("N>=2, R>=1, distinct nonnegative seeds required")
    positive = (*cfg["ratios"], *cfg["steps"], cfg["exchange_prefactor"], cfg["target_time"], cfg["time_cap"], cfg["memory_limit_mb"])
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        raise ValueError("ratios, steps, prefactor, target time, and caps must be positive and finite")
    if any(not np.isfinite(x) or not 0 < x < 1 for x in (cfg["p_left"], cfg["p_right"],
            cfg["steady_tolerance"], cfg["dynamic_tolerance"], cfg["timing_tolerance"])):
        raise ValueError("reservoir populations and tolerances must lie strictly between zero and one")
    if (not np.isfinite(cfg["rate_margin"]) or cfg["rate_margin"] < 1 or not np.isfinite(cfg["horizon_factor"])
            or cfg["horizon_factor"] <= max(cfg["rate_margin"], 1 + cfg["timing_tolerance"])
            or cfg["step_cap"] < 1 or cfg["promising_ratios"] < 1 or options.step_refinements < 0
            or options.progress_every < 0 or any(R < 1 for R in options.extend_R_values)):
        raise ValueError("invalid search caps/margins; horizon must exceed rate margin and steady deadline")
    options.config = cfg
    return options


def write_report(payload, directory):
    cfg = payload["configuration"]
    lines = [f"# {cfg['model_name']}: randomized XX reservoir scaling study", "",
        f"Status: `{payload['status']}`. Classical covariance calculation only; no circuit or hardware execution.", "",
        f"H = {cfg['exchange_prefactor']:g} J sum(XX+YY), Delta=0, pL={cfg['p_left']:g}, pR={cfg['p_right']:g}.",
        f"Sampling: {cfg['policy']}; order: {cfg['order']}; Hamiltonian: {cfg['system_formula']}.", "",
        "## Definitions and acceptance", "",
        r"$r=\Gamma/J$, $\tau=Jt$, $s=J\Delta t$. Preparations are all-zero, all-one, and a left-end excitation.", "",
        r"Approximation error is $\max_{k,\mathrm{prep},j}\{|n_j^{app}-n_j^{ex}|,|I_j^{app}-I_j^{ex}|/J\}$, "
        "accumulated at every algorithmic node, not the one-step local error.",
        f"Dynamic tolerance={cfg['dynamic_tolerance']:g}; averaged bias and finite-R sampling budgets "
        f"are each {cfg['bias_tolerance']:g}. All {len(cfg['seeds'])} seeds must pass.", "",
        r"The exact settling criterion is $q\max_{\mathrm{prep}}\|C(t)-C_*\|_F\le\epsilon_{ss}$, "
        r"where $q=\max(1,4c)$ and $H=cJ\sum(XX+YY)$. It bounds both populations and current/J, "
        "including at every later time. This is a conservative covariance/observable criterion, "
        "NOT full-state trace distance, and NOT Model 2 Experiment 1's population-only criterion.",
        f"Steady tolerance={cfg['steady_tolerance']:g}; timing tolerance={100*cfg['timing_tolerance']:g}% "
        "(half that budget for the recommended averaged step).",
        r"For the mean dilation use $b+q\max\|C_k-C_s\|_F$, where $C_s$ is its affine fixed point "
        "and b is its local stationary bias relative to the physical C*. Affine-map residual/positivity checks are saved.", "",
        "Finite-R trajectories do not settle permanently: pumping/loss choices keep fluctuating. "
        f"They are tested to {cfg['horizon_factor']:g} times the exact dimensionless settling time. "
        + ("Additionally, every checked node after the exact settling time plus its timing allowance must meet "
           "the local steady tolerance through that horizon. " if cfg['finite_R_steady_check'] else
           "The optional finite-window steady-tail check is disabled; R is accepted only for approximation accuracy. ")
        + "No infinite-time or probabilistic guarantee follows for finite R.", "",
        "## Recommendations", "",
        "| N | Gamma/J | Exact Jt_ss bound | Largest tested s (infinite R) | Recommended s | Smallest passing tested R | J | Gamma | Delta t | Steps to target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["recommendations"]:
        lines.append("| " + " | ".join("unresolved" if row.get(key) is None else
            f"{row[key]:.8g}" for key in ("N", "ratio", "exact_dimensionless_steady_time",
            "largest_tested_infinite_R_step", "recommended_dimensionless_step", "minimum_tested_R",
            "J_for_target", "gamma_for_target", "delta_t_for_target", "substeps_to_target_time")) + " |")
    lines += ["", "Rates place the *exact covariance-bound* settling time before the target with the configured "
        "margin. They do not remove the dimensionless step cost. The mean settling time is also included "
        "in rate selection. Finite-R steady claims are limited to the validated horizon/seeds.", "",
        "## Search audit and limitations", "",
        f"Sizes: {cfg['sizes']}; ratio grid: {cfg['ratios']}; s grid: {cfg['steps']}; R grid: {cfg['R_values']}; "
        f"supplemental R: {payload['supplemental_R_values']}; seeds: {cfg['seeds']}.",
        "Ratios are ranked by exact settling time, optionally locally refined. The promising ratios "
        "are compared by mean step count; s is refined locally at the selected ratio. This is not a global "
        "optimization over all r,s,R or a proof of minimal total R-times-step cost. Dilation resonances "
        "may make accuracy nonmonotonic in s; bisection is a local heuristic only.",
        "Cases save after completion; interruption restarts an unfinished case. Cap or fixed-point-solve "
        "failures are unresolved, not successful. Later seeds skipped after a rejection are explicitly untested.",
        "The Frobenius norm contracts under the exact dissipative propagator and the mean homogeneous "
        "map (an average of A C A^T with each A a contraction). A bias-plus-two-decaying-envelopes bound "
        "can certify an unevaluated mean-error tail. Reported peaks then cover evaluated nodes only. "
        "Bounds and crossings are floating-point results, not interval-arithmetic certificates.",
        "The covariance reduction is exact for these XX observables and endpoint channels, not for the "
        "full generally non-Gaussian many-body density matrix. Delta!=0 is deliberately unsupported. "
        "Signed dilation cosines are preserved; jumps are not replaced with exact reservoir exponentials.",
        "The exact-H option preserves Model 2 Experiment 2's system factor. Even-odd is an explicit "
        "optional change of the approximate method; the physical reference always exponentiates the full H.",
        "Seeds use independent path streams, different from other drivers' row-major random draws. "
        "Step boundaries do not depend on a plotting grid. No measurement shot noise, routing, hardware "
        "gate depth or pulse fidelity is included.", "",
        "[Complete checkpoint and cases](study.json).", ""]
    if payload.get("figures"):
        lines += ["Figures: [scaling](scaling.png), [ratio landscape](ratio_landscape.png), "
                  "[rates and workload](target_workload.png).", ""]
    (directory / "Study.md").write_text("\n".join(lines), encoding="utf-8")


def plots(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = payload["recommendations"]
    names = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, label in zip(axes, ("exact_dimensionless_steady_time", "recommended_dimensionless_step", "minimum_tested_R"),
                             ("Exact covariance-bound Jt_ss", "Recommended s = J delta t", "Smallest passing tested R")):
        group = [r for r in rows if r.get(key) is not None]
        if group:
            ax.loglog([r["N"] for r in group], [r[key] for r in group], "o-")
        ax.set(xlabel="N", ylabel=label)
        ax.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(directory / "scaling.png", dpi=170); plt.close(figure)
    names.append("scaling.png")
    figure, ax = plt.subplots(figsize=(8, 5))
    for n in payload["configuration"]["sizes"]:
        group = sorted([r for r in payload["exact_landscape"] if r["N"] == n and r["steady_time"] is not None], key=lambda r:r["ratio"])
        if group:
            ax.loglog([r["ratio"] for r in group], [r["steady_time"] for r in group], "o-", label=f"N={n}")
    ax.set(xlabel="Gamma/J", ylabel="Exact covariance-bound Jt_ss"); ax.grid(alpha=.25)
    if ax.lines:
        ax.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(directory / "ratio_landscape.png", dpi=170); plt.close(figure)
    names.append("ratio_landscape.png")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for key in ("J_for_target", "gamma_for_target"):
        if rows:
            axes[0].loglog([r["N"] for r in rows], [r[key] for r in rows], "o-", label=key)
    if rows:
        axes[0].legend()
        axes[1].loglog([r["N"] for r in rows], [r["substeps_to_target_time"] for r in rows], "o-")
    axes[0].set(xlabel="N", ylabel="Physical rates")
    axes[1].set(xlabel="N", ylabel="Substeps per trajectory to target time")
    for ax in axes:
        ax.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(directory / "target_workload.png", dpi=170); plt.close(figure)
    return names + ["target_workload.png"]


def main(options):
    config = options.config
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    directory = options.output_directory / f"scaling_{digest}"
    path = directory / "study.json"
    if path.exists():
        if not options.resume:
            raise ValueError(f"study already exists; use --resume: {path}")
        payload = json.loads(path.read_text())
        if payload["configuration"] != config:
            raise ValueError("resume configuration differs")
    elif options.resume:
        raise FileNotFoundError(f"no checkpoint to resume: {path}")
    else:
        payload = {"configuration": config, "status": "running", "exact_landscape": [],
                   "ratio_selections": {}, "averaged_cases": {}, "ensemble_cases": {}, "recommendations": []}
    payload["step_refinement_rounds"] = max(options.step_refinements, payload.get("step_refinement_rounds", 0))
    payload["supplemental_R_values"] = sorted(set(payload.get("supplemental_R_values", [])) | set(options.extend_R_values))
    evaluation_config = {**config, "progress_every": options.progress_every}
    def save():
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, payload)
    def model(n, ratio):
        return Model(n, ratio, config["p_left"], config["p_right"], config["exchange_prefactor"])
    def averaged(n, ratio, step, exact_time):
        key = f"{n}:{ratio:.17g}:{step:.17g}"
        if key not in payload["averaged_cases"]:
            print(f"Averaged N={n}, Gamma/J={ratio:.7g}, s={step:g}", flush=True)
            payload["averaged_cases"][key] = evaluate_step(model(n, ratio), step, exact_time, evaluation_config)
            save()
        return payload["averaged_cases"][key]
    save()
    print(f"Study checkpoint: {path}", flush=True)
    for n in config["sizes"]:
        for ratio in config["ratios"]:
            if any(row["N"] == n and row["ratio"] == ratio for row in payload["exact_landscape"]):
                continue
            crossing = exact_crossing(model(n, ratio), config["steady_tolerance"], config["time_cap"])
            payload["exact_landscape"].append({"N": n, "ratio": ratio, "steady_time": crossing})
            save(); print(f"Exact N={n}, Gamma/J={ratio:g}: Jt_ss={crossing}", flush=True)
        if str(n) not in payload["ratio_selections"]:
            rows = sorted([row for row in payload["exact_landscape"] if row["N"] == n
                and row["steady_time"] is not None and row["steady_time"] > 0], key=lambda r:r["steady_time"])
            candidates, boundary = [], None
            if rows:
                best = rows[0]
                index = config["ratios"].index(best["ratio"])
                boundary = index in (0, len(config["ratios"])-1)
                if not boundary:
                    fit = minimize_scalar(lambda log_r: exact_crossing(model(n, float(np.exp(log_r))),
                        config["steady_tolerance"], config["time_cap"]) or config["time_cap"],
                        bounds=np.log([config["ratios"][index-1], config["ratios"][index+1]]), method="bounded",
                        options={"xatol": 1e-5})
                    ratio = float(np.exp(fit.x))
                    crossing = exact_crossing(model(n, ratio), config["steady_tolerance"], config["time_cap"])
                    if crossing is not None and crossing > 0:
                        payload["exact_landscape"].append({"N": n, "ratio": ratio, "steady_time": crossing, "locally_refined": True})
                        candidates.append(ratio)
                candidates += [row["ratio"] for row in rows]
            payload["ratio_selections"][str(n)] = {"ratios": candidates[:config["promising_ratios"]], "grid_optimum_at_boundary": boundary}
            save()
        candidates = []
        for ratio in payload["ratio_selections"][str(n)]["ratios"]:
            exact_time = next(r["steady_time"] for r in payload["exact_landscape"] if r["N"] == n and r["ratio"] == ratio)
            found = {}
            for step in config["steps"]:
                case = averaged(n, ratio, step, exact_time)
                for label in ("infinite_R", "recommended"):
                    if label not in found and case[f"passed_{label}"]:
                        found[label] = case
                if "recommended" in found:
                    break
            if "recommended" in found:
                candidates.append({"ratio": ratio, "exact_time": exact_time, "found": found,
                    "cost": found["recommended"]["steady_time_bracket"][1]/found["recommended"]["dimensionless_step"]})
        if not candidates:
            print(f"N={n}: no passing averaged step within tested grid/caps", flush=True)
            continue
        best = min(candidates, key=lambda r:r["cost"])
        ratio, exact_time, found = best["ratio"], best["exact_time"], best["found"]
        brackets = {}
        for label, case in list(found.items()):
            lower = case["dimensionless_step"]
            upper = min((s for s in config["steps"] if s > lower), default=None)
            for _ in range(payload["step_refinement_rounds"] if upper is not None else 0):
                midpoint = (lower + upper)/2
                case = averaged(n, ratio, midpoint, exact_time)
                if case[f"passed_{label}"]:
                    lower, found[label] = midpoint, case
                else:
                    upper = midpoint
            brackets[label] = [lower, upper]
        step = found["recommended"]["dimensionless_step"]
        minimum_R = None
        for R in sorted(set(config["R_values"]) | set(payload["supplemental_R_values"])):
            key = f"{n}:{ratio:.17g}:{step:.17g}:R{R}"
            if key not in payload["ensemble_cases"]:
                print(f"Finite R: N={n}, r={ratio:.7g}, s={step:g}, R={R}", flush=True)
                payload["ensemble_cases"][key] = evaluate_ensemble(model(n, ratio), step, exact_time, R, evaluation_config)
                save()
            if payload["ensemble_cases"][key]["passed_all_seeds"]:
                minimum_R = R
                break
        J = max(config["rate_margin"] * exact_time, found["recommended"]["steady_time_bracket"][1]) / config["target_time"]
        recommendation = {"N": n, "ratio": ratio, "exact_dimensionless_steady_time": exact_time,
            "largest_tested_infinite_R_step": found["infinite_R"]["dimensionless_step"],
            "recommended_dimensionless_step": step, "minimum_tested_R": minimum_R,
            "J_for_target": J, "gamma_for_target": ratio * J, "delta_t_for_target": step / J,
            "substeps_to_target_time": int(np.ceil(config["target_time"] * J / step)),
            "step_refinement_brackets": brackets,
            "grid_optimum_at_boundary": payload["ratio_selections"][str(n)]["grid_optimum_at_boundary"]}
        payload["recommendations"] = sorted([r for r in payload["recommendations"] if r["N"] != n] + [recommendation], key=lambda r:r["N"])
        save(); write_report(payload, directory)
        print(f"RECOMMENDATION: {recommendation}", flush=True)
    valid = len(payload["recommendations"]) == len(config["sizes"]) and all(r["minimum_tested_R"] is not None for r in payload["recommendations"])
    payload["status"] = "complete" if valid else "complete_with_search_limits"
    save(); write_report(payload, directory)
    if not options.no_plots:
        payload["figures"] = plots(payload, directory)
        save()
    write_report(payload, directory)
    print(f"Saved study: {directory}; status={payload['status']}", flush=True)
    return payload
