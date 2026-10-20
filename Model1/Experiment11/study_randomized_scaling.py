"""Classical dimensionless study of size, boundary loss, timestep and R.

No Aer, hardware or other experiments. The deterministic covariance update
averages the two randomized alternatives; it does not compose both jumps.
Each completed case is atomically saved, allowing configuration-safe resume.
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
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse import csr_matrix, diags, kron

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Model1.Experiment11.classical_randomized_lie_trotter import atomic_json

DIRECTORY = Path(__file__).resolve().parent


def generator(n, ratio):
    """Real gauge: a_i=(-i)^(i-site) x_i; J=1 and h=0."""
    G = np.diag(-np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
    G[0, 0] = G[-1, -1] = -ratio/2
    return G


def exact_crossing(n, ratio, tolerance, cap):
    G = generator(n, ratio)
    initial = np.zeros(n)
    initial[n//2] = 1
    def distance(t):
        amplitude = expm(t*G) @ initial
        return float(amplitude @ amplitude)
    upper = min(1.0, cap)
    while distance(upper) > tolerance and upper < cap:
        upper = min(2*upper, cap)
    if distance(upper) > tolerance:
        return None
    return float(brentq(lambda t: distance(t)-tolerance, 0, upper, xtol=1e-9))


def averaged_map(n, ratio, step):
    """Sparse linear map on vec(M), preserving the original signed cosine."""
    U = np.eye(n)
    c, s = np.cos(step), np.sin(step)
    for parity in (0, 1):
        left, right = U[parity:-1:2].copy(), U[parity+1::2].copy()
        U[parity:-1:2] = c*left-s*right
        U[parity+1::2] = s*left+c*right
    damping = np.cos(np.sqrt(2*ratio*step))
    left, right = np.ones(n), np.ones(n)
    left[0], right[-1] = damping, damping
    weights = (np.outer(left, left)+np.outer(right, right))/2
    operator = kron(csr_matrix(U), csr_matrix(U), format="csr")
    operator = operator.multiply(weights.reshape(-1, 1)).tocsr()
    return operator, U, float(damping)


def evaluate_step(n, ratio, step, exact_time, config, *, error_limit=None, timing_limit=None):
    """All algorithmic nodes checked; no save grid inserts additional substeps."""
    error_limit = config["dynamic_tolerance"] if error_limit is None else error_limit
    timing_limit = config["timing_tolerance"] if timing_limit is None else timing_limit
    started = time.perf_counter()
    operator, _, damping = averaged_map(n, ratio, step)
    reference_step = expm(step*generator(n, ratio))
    amplitude = np.zeros(n); amplitude[n//2] = 1
    covariance = np.outer(amplitude, amplitude).ravel()
    diagonal = np.arange(n)*(n+1)
    neighbor = np.arange(n-1)*(n+1)+1
    horizon = min(config["time_cap"], config["horizon_factor"]*exact_time)
    maximum = min(config["step_cap"], int(np.ceil(horizon/step)))
    sample_stride = max(1, maximum//300)
    peaks = {"population": 0.0, "normalized_flow": 0.0}
    peak_times = dict.fromkeys(peaks, 0.0)
    crossing = None
    curve = {"times": [0.0], "mean_excitation": [1.0], "exact_excitation": [1.0]}
    certified_tail = False
    tail_limit = min(error_limit, config["bias_tolerance"])
    status = "cap_reached"
    previous_time = 0.0
    for k in range(1, maximum+1):
        covariance = operator @ covariance
        t = k*step
        P = float(covariance[diagonal].sum())
        if not certified_tail:
            amplitude = reference_step @ amplitude
            exact_P = float(amplitude @ amplitude)
            errors = {"population": float(np.max(np.abs(covariance[diagonal]-amplitude**2))),
                      "normalized_flow": float(np.max(np.abs(2*covariance[neighbor]-2*amplitude[:-1]*amplitude[1:])))}
            for name, error in errors.items():
                if error > peaks[name]:
                    peaks[name], peak_times[name] = error, t
            # Any later population/current difference is <= P_approx+P_exact.
            certified_tail = P+exact_P <= tail_limit
        else:
            exact_P = None
        if k % sample_stride == 0 or k == maximum:
            curve["times"].append(t); curve["mean_excitation"].append(P)
            # Reference samples from expm, not display-grid evolution changes.
            a = expm(t*generator(n, ratio))[:, n//2] if exact_P is None else amplitude
            curve["exact_excitation"].append(float(a @ a))
        if crossing is None and P <= config["steady_tolerance"]:
            crossing = [previous_time, t]
        if max(peaks.values()) > error_limit:
            status = "transient_error_rejected"
            break
        if crossing is not None and certified_tail:
            status = "complete"
            break
        if k % 100000 == 0:
            print(f"    N={n} r={ratio:.5g} s={step:g}: {k:,} steps, τ={t:.3g}, P={P:.3g}", flush=True)
        previous_time = t
    timing_error = None if crossing is None else abs(crossing[1]-exact_time)/exact_time
    return {"N": n, "ratio": ratio, "dimensionless_step": step, "status": status,
        "max_errors": peaks, "error_peak_times": peak_times,
        "steady_time_bracket": crossing, "relative_steady_time_error": timing_error,
        "tail_error_certified_below": tail_limit if certified_tail else None,
        "error_audit": "reported peaks are evaluated-node maxima; all later-node errors are certified below the tail bound",
        "passed": bool(status == "complete" and timing_error <= timing_limit),
        "tested_error_limit": error_limit, "tested_timing_limit": timing_limit,
        "completed_steps": k, "boundary_cosine": damping, "curve": curve,
        "wall_seconds": time.perf_counter()-started}


def advance_real(amplitudes, ratio, step, boundaries):
    c, s = np.cos(step), np.sin(step)
    for parity in (0, 1):
        left, right = amplitudes[:, parity:-1:2].copy(), amplitudes[:, parity+1::2].copy()
        amplitudes[:, parity:-1:2] = c*left-s*right
        amplitudes[:, parity+1::2] = s*left+c*right
    sites = np.where(boundaries == 0, 0, amplitudes.shape[1]-1)
    amplitudes[np.arange(len(amplitudes)), sites] *= np.cos(np.sqrt(2*ratio*step))


def evaluate_ensembles(n, ratio, step, exact_time, config, *, early_reject=False):
    """Nested R prefixes; independent per-path RNG streams stable across R caps."""
    started = time.perf_counter()
    R_values, seeds = config["R_values"], config["seeds"]
    S, largest = len(seeds), max(R_values)
    amplitudes = np.zeros((S*largest, n)); amplitudes[:, n//2] = 1
    operator, _, _ = averaged_map(n, ratio, step)
    covariance = np.zeros(n*n); covariance[(n//2)*(n+1)] = 1
    reference = np.zeros(n); reference[n//2] = 1
    reference_step = expm(step*generator(n, ratio))
    diagonal, neighbor = np.arange(n)*(n+1), np.arange(n-1)*(n+1)+1
    selected = np.asarray(R_values)-1
    denominators = np.asarray(R_values)[None, :, None]
    crossing = np.full((S, len(R_values)), np.nan)
    errors_vs_average = np.zeros_like(crossing)
    errors_vs_exact = np.zeros_like(crossing)
    generators = [np.random.default_rng(np.random.SeedSequence([seed, path]))
                  for seed in seeds for path in range(largest)]
    horizon = min(config["time_cap"], config["horizon_factor"]*exact_time)
    maximum = min(config["step_cap"], int(np.ceil(horizon/step)))
    block_size, choices = 2048, None
    certified_tail = False
    rejection = False
    for k in range(1, maximum+1):
        position = (k-1) % block_size
        if position == 0:
            choices = np.stack([rng.integers(0, 2, size=block_size, dtype=np.int8) for rng in generators], axis=1)
        advance_real(amplitudes, ratio, step, choices[position])
        if not certified_tail:
            covariance = operator @ covariance
            populations = (amplitudes**2).reshape(S, largest, n)
            prefix = populations.cumsum(axis=1)[:, selected, :]/denominators
            P = prefix.sum(axis=-1)
        else:
            # Tail errors are already bounded; only finite-R survival/crossing
            # needs further propagation. Avoid an unused covariance update.
            path_norms = (amplitudes**2).sum(axis=1).reshape(S, largest)
            P = path_norms.cumsum(axis=1)[:, selected]/np.asarray(R_values)[None, :]
        crossing[np.isnan(crossing) & (P <= config["steady_tolerance"])] = k*step
        if not certified_tail:
            reference = reference_step @ reference
            currents = (2*amplitudes[:, :-1]*amplitudes[:, 1:]).reshape(S, largest, n-1)
            flow = currents.cumsum(axis=1)[:, selected, :]/denominators
            mean_P = float(covariance[diagonal].sum())
            exact_P = float(reference @ reference)
            errors_vs_average = np.maximum(errors_vs_average, np.maximum(
                np.max(np.abs(prefix-covariance[diagonal]), axis=-1),
                np.max(np.abs(flow-2*covariance[neighbor]), axis=-1)))
            errors_vs_exact = np.maximum(errors_vs_exact, np.maximum(
                np.max(np.abs(prefix-reference**2), axis=-1),
                np.max(np.abs(flow-2*reference[:-1]*reference[1:]), axis=-1)))
            certified_tail = (float(P.max())+mean_P <= config["sampling_tolerance"]
                              and float(P.max())+exact_P <= config["dynamic_tolerance"])
        if early_reject and (np.any(errors_vs_average > config["sampling_tolerance"])
                or np.any(errors_vs_exact > config["dynamic_tolerance"])
                or np.any(np.isfinite(crossing) & (np.abs(crossing-exact_time)/exact_time > config["timing_tolerance"]))
                or (k*step > exact_time*(1+config["timing_tolerance"]) and np.any(np.isnan(crossing)))):
            rejection = True
            break
        if np.all(np.isfinite(crossing)) and certified_tail:
            break
        if k % 100000 == 0:
            print(f"    finite R, N={n}: {k:,} steps; crossed {np.isfinite(crossing).sum()}/{crossing.size}", flush=True)
    records = []
    for column, R in enumerate(R_values):
        per_seed = []
        for row, seed in enumerate(seeds):
            t = crossing[row, column]
            relative = None if not np.isfinite(t) else float(abs(t-exact_time)/exact_time)
            passed = bool(certified_tail and relative is not None
                and relative <= config["timing_tolerance"]
                and errors_vs_average[row, column] <= config["sampling_tolerance"]
                and errors_vs_exact[row, column] <= config["dynamic_tolerance"])
            failure_confirmed = (errors_vs_average[row, column] > config["sampling_tolerance"]
                or errors_vs_exact[row, column] > config["dynamic_tolerance"]
                or (relative is not None and relative > config["timing_tolerance"])
                or (relative is None and k*step > exact_time*(1+config["timing_tolerance"])))
            if not passed and not failure_confirmed:
                passed = None  # Other seeds may be unfinished after one seed rejects R.
            per_seed.append({"seed": seed, "passed": passed,
                "evaluation_status": "passed" if passed is True else "failed" if passed is False else "not_fully_evaluated",
                "max_error_vs_averaged_channel": float(errors_vs_average[row, column]),
                "max_error_vs_exact": float(errors_vs_exact[row, column]),
                "steady_time_bracket": None if relative is None else [float(t-step), float(t)],
                "relative_steady_time_error": relative})
        records.append({"R": R, "passed_all_seeds": all(row["passed"] is True for row in per_seed),
                        "passed_seed_count": sum(row["passed"] is True for row in per_seed),
                        "unfinished_seed_count": sum(row["passed"] is None for row in per_seed), "per_seed": per_seed})
    return {"N": n, "ratio": ratio, "dimensionless_step": step, "results": records,
        "minimum_tested_R_passing_all_seeds": next((row["R"] for row in records if row["passed_all_seeds"]), None),
        "tail_certified": certified_tail, "completed_steps": k, "wall_seconds": time.perf_counter()-started,
        "status": "early_rejected" if rejection else "complete" if certified_tail and np.all(np.isfinite(crossing)) else "cap_reached",
        "error_audit": "peaks cover evaluated nodes only; early rejection proves this R cannot pass every seed",
        "RNG": "independent SeedSequence([seed,path]) streams; differs from circuit driver's row-major seeded draw"}


def find_ensemble_size(n, ratio, step, exact_time, config, cache, on_complete):
    """Evaluate ascending R, cheaply rejecting provably failing candidates."""
    records, seconds = [], 0.0
    for R in config["R_values"]:
        key = f"{n}:{ratio:.12g}:{step:.12g}:R{R}"
        if key not in cache:
            print(f"    checking R={R} across {len(config['seeds'])} seeds", flush=True)
            cache[key] = evaluate_ensembles(n, ratio, step, exact_time, {**config, "R_values": [R]}, early_reject=True)
            on_complete()
        result = cache[key]
        records.extend(result["results"])
        seconds += result["wall_seconds"]
        print(f"    R={R}: {result['status']}", flush=True)
        if result["minimum_tested_R_passing_all_seeds"] is not None:
            return {**result, "results": records, "wall_seconds": seconds,
                    "search_mode": "ascending R, permanent-error early rejection, case-level persistence"}
    return {**result, "results": records, "wall_seconds": seconds,
            "search_mode": "ascending R, permanent-error early rejection, case-level persistence"}


def plots(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recommendations = payload["recommendations"]
    if not recommendations:
        return []
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    sizes = [row["N"] for row in recommendations]
    for axis, key, title in zip(axes.flat, ("ratio", "exact_dimensionless_steady_time", "recommended_dimensionless_step", "minimum_tested_R"),
            ("Selected γ/J", "Exact dimensionless steady time Jt", "Recommended dimensionless step JΔt", "Smallest tested R passing all seeds")):
        points = [(row["N"], row[key]) for row in recommendations if row[key] is not None]
        if points:
            axis.plot(*zip(*points), "o-")
        axis.set_xlabel("N"); axis.set_title(title); axis.grid(alpha=0.3)
        if key in ("exact_dimensionless_steady_time", "minimum_tested_R"):
            axis.set_yscale("log")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scaling.png"
    figure.savefig(path, dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 6))
    for n in sizes:
        rows = sorted((row for row in payload["exact_landscape"] if row["N"] == n and row["steady_time"] is not None), key=lambda row: row["ratio"])
        axis.plot([row["ratio"] for row in rows], [row["steady_time"] for row in rows], "o-", label=f"N={n}")
    axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("γ/J")
    axis.set_ylabel("Exact steady time Jt, tolerance 10⁻³" if payload["configuration"]["steady_tolerance"] == 1e-3 else "Exact steady time Jt")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout()
    second = directory / "ratio_landscape.png"
    figure.savefig(second, dpi=180); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    target_steps = [int(np.ceil(payload["configuration"]["target_time"]*row["J_for_target"]/row["recommended_dimensionless_step"])) for row in recommendations]
    axes[0, 0].semilogy(sizes, [row["J_for_target"] for row in recommendations], "o-", label="J")
    axes[0, 0].semilogy(sizes, [row["gamma_for_target"] for row in recommendations], "o-", label="γ")
    axes[0, 0].set_title("Rates for the fixed physical-time target"); axes[0, 0].legend()
    axes[0, 1].semilogy(sizes, [row["delta_t_for_target"] for row in recommendations], "o-")
    axes[0, 1].set_title("Recommended physical Δt")
    axes[1, 0].semilogy(sizes, target_steps, "o-")
    axes[1, 0].set_title("Substeps per trajectory through target T")
    valid = [(row["N"], count*row["minimum_tested_R"]) for row, count in zip(recommendations, target_steps) if row["minimum_tested_R"] is not None]
    if valid:
        axes[1, 1].semilogy(*zip(*valid), "o-")
    axes[1, 1].set_title("R × substeps (excluding shots/bases)")
    for axis in axes.flat:
        axis.set_xlabel("N"); axis.grid(alpha=0.3)
    figure.tight_layout()
    third = directory / "target_workload.png"
    figure.savefig(third, dpi=180); plt.close(figure)
    return [path.name, second.name, third.name]


def write_report(payload, path):
    cfg = payload["configuration"]
    lines = ["# Experiment 11 dimensionless randomized-scaling study", "",
        f"Status: {payload['status']}. Initial excitation: N//2; J=1, h=0 during the sweep.", "",
        f"Vacuum tolerance: {cfg['steady_tolerance']:g}. Normalized transient tolerance: {cfg['dynamic_tolerance']:g}.",
        f"Recommended timestep bias budget: {cfg['bias_tolerance']:g}; sampling budget: {cfg['sampling_tolerance']:g}.",
        f"Timing error limit: {100*cfg['timing_tolerance']:g}%; averaged-map timing budget: {100*cfg['bias_timing_tolerance']:g}%.", "",
        "Transient metric is max over algorithmic nodes and sites/bonds of population error and |current error|/J. Uniform h and zero XY correlations are not optimization axes.", "",
        "All future-node errors are bounded once remaining excitation sums fall below the error budget. No plotting/save interval inserts steps.", "",
        "Largest steps are largest *tested* passing values, not a proof of continuous/global optimality. Ratios are refined locally around the best exact grid point; only promising ratios receive timestep sweeps.", "",
        "Finite-R acceptance requires every listed seed to pass, but this is empirical validation, not a probabilistic guarantee. Random streams are independent per trajectory and differ from existing circuit seeded prefixes.", "",
        "| N | γ/J | exact Jt_ss | largest tested s (infinite R) | recommended s | tested R | J for target T | γ | Δt | approximate substeps |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    def number(value):
        return "not found" if value is None else f"{value:.6g}"
    for row in payload["recommendations"]:
        lines.append("| "+" | ".join(number(row[key]) for key in ("N", "ratio", "exact_dimensionless_steady_time",
            "largest_tested_infinite_R_step", "recommended_dimensionless_step", "minimum_tested_R", "J_for_target",
            "gamma_for_target", "delta_t_for_target", "approximate_substep_count"))+" |")
    extended = {str(row["N"]): row["supplemental_R_extension"] for row in payload["ensemble_cases"].values() if "supplemental_R_extension" in row}
    lines += ["", f"Local timestep refinement rounds: {payload.get('step_refinement_rounds', 0)}.", "",
              f"Supplemental trajectory-count extensions: {extended if extended else 'none'}.", "",
              f"Target T={cfg['target_time']:g}. Rates include a {100*(cfg['rate_margin']-1):g}% speed margin: J=margin·(Jt_ss)/T, γ=rJ, Δt=s/J.", "",
              "Rate scaling compresses physical time but does not reduce the dimensionless substep count. Missing R means no tested ensemble size passed; do not treat that recommendation as fully validated.", "",
              "## Search configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
              "## Workload through the fixed physical-time target", "",
              "These counts are algorithmic, not measured transpilation metrics. Each substep has N exchange blocks (N-1 hopping bonds and one selected boundary), plus a reset.", "",
              "| N | substeps per trajectory to T | R × substeps | exchange blocks per trajectory |",
              "|---|---|---|---|"]
    for row in payload["recommendations"]:
        steps = int(np.ceil(cfg["target_time"]*row["J_for_target"]/row["recommended_dimensionless_step"]))
        work = None if row["minimum_tested_R"] is None else steps*row["minimum_tested_R"]
        lines.append(f"| {row['N']} | {steps:,} | {number(work)} | {steps*row['N']:,} |")
    fit_rows = [row for row in payload["recommendations"] if row["N"] >= 10]
    if len(fit_rows) >= 3:
        exponent = np.polyfit(np.log([row["N"] for row in fit_rows]), np.log([row["exact_dimensionless_steady_time"] for row in fit_rows]), 1)[0]
        lines += ["", f"A descriptive log-log fit over the studied N>=10 gives τ_ss proportional to N^{exponent:.3f}. This finite-size fit is not an asymptotic scaling proof."]
    lines += ["", "R is selected for local absolute population/current errors: spreading one excitation over more sites can reduce this metric. A lower tested R at higher N is not evidence that generic large-N states are easier to simulate.", "",
        "N=2 is exceptional: both sites are dissipative boundaries. The fastest exact grid point is at the ratio search boundary, so no finite globally optimal loss ratio is claimed.", "",
        "Results describe ideal classical expectations, not measurement-shot noise, gate errors or hardware feasibility. The recommended physical rates accelerate the model but leave a potentially very large circuit workload.", "",
        "Figures: [size scaling](scaling.png), [ratio landscape](ratio_landscape.png), [target rates and workload](target_workload.png). Full case errors, peak times, crossing brackets, seed results and timings are in study.json.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=(2,4,7,10,15,20,30,50))
    parser.add_argument("--ratios", nargs="+", type=float, default=(0.25,0.5,1,1.5,2,2.5,3,4,6,8))
    parser.add_argument("--steps", nargs="+", type=float, default=(0.1,0.08,0.06,0.04,0.03,0.02,0.015,0.01,0.0075,0.005,0.003,0.002))
    parser.add_argument("--R-values", nargs="+", type=int, default=(4,8,16,32,64,128,256))
    parser.add_argument("--extend-R-values", nargs="+", type=int, default=(),
                        help="on resume, extend only ensemble cases with no passing R; does not rerun exact/timestep cases")
    parser.add_argument("--seeds", nargs="+", type=int, default=(29,101,307,509,701))
    parser.add_argument("--promising-ratios", type=int, default=2)
    parser.add_argument("--step-refinements", type=int, default=2,
                        help="local bisection rounds at the selected ratio; may be increased on resume")
    parser.add_argument("--steady-tolerance", type=float, default=1e-3)
    parser.add_argument("--dynamic-tolerance", type=float, default=0.01)
    parser.add_argument("--timing-tolerance", type=float, default=0.05)
    parser.add_argument("--target-time", type=float, default=20)
    parser.add_argument("--rate-margin", type=float, default=1.1)
    parser.add_argument("--time-cap", type=float, default=100000)
    parser.add_argument("--step-cap", type=int, default=3000000)
    parser.add_argument("--horizon-factor", type=float, default=1.5)
    parser.add_argument("--output-directory", type=Path, default=DIRECTORY / "study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    options = parser.parse_args(arguments)
    config = {key: getattr(options, key) for key in ("steady_tolerance", "dynamic_tolerance", "timing_tolerance", "target_time", "rate_margin", "time_cap", "step_cap", "horizon_factor", "promising_ratios")}
    config.update({"schema_version": 1, "sizes": sorted(set(options.sizes)), "ratios": sorted(set(options.ratios)),
        "steps": sorted(set(options.steps), reverse=True), "R_values": sorted(set(options.R_values)), "seeds": list(options.seeds),
        "bias_tolerance": options.dynamic_tolerance/2, "sampling_tolerance": options.dynamic_tolerance/2,
        "bias_timing_tolerance": options.timing_tolerance/2})
    if not config["sizes"] or min(config["sizes"]) < 2 or min(config["R_values"]) < 1 or min(config["seeds"]) < 0:
        raise ValueError("invalid sizes, R values or seeds")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be distinct")
    if any(value < 1 for value in options.extend_R_values):
        raise ValueError("extended R values must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in (*config["ratios"], *config["steps"], options.target_time, options.time_cap)):
        raise ValueError("ratios, steps, target time and time cap must be positive and finite")
    if not all(np.isfinite(value) and 0 < value < 1 for value in (options.steady_tolerance, options.dynamic_tolerance, options.timing_tolerance)):
        raise ValueError("tolerances must lie between zero and one")
    if not np.isfinite(options.rate_margin) or options.rate_margin < 1 or options.step_cap < 1 or options.promising_ratios < 1 or options.step_refinements < 0 or not np.isfinite(options.horizon_factor) or options.horizon_factor <= 1:
        raise ValueError("invalid margin or search caps")
    options.config = config
    return options


def main(options):
    config = options.config
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    directory = options.output_directory / f"scaling_{digest}"
    path = directory / "study.json"
    if path.exists():
        if not options.resume:
            raise ValueError(f"study exists; use --resume: {path}")
        payload = json.loads(path.read_text())
        if payload["configuration"] != config:
            raise ValueError("resume configuration differs")
    elif options.resume:
        raise FileNotFoundError(f"no checkpoint to resume: {path}")
    else:
        payload = {"configuration": config, "status": "running", "exact_landscape": [], "ratio_selections": {},
                   "averaged_cases": {}, "ensemble_cases": {}, "R_case_cache": {}, "recommendations": []}
    payload.setdefault("R_case_cache", {})
    payload["step_refinement_rounds"] = max(payload.get("step_refinement_rounds", 0), options.step_refinements)
    payload["supplemental_R_values"] = sorted(set(payload.get("supplemental_R_values", [])) | set(options.extend_R_values))
    # Completed simultaneous-R checks can seed the ascending-R cache unchanged.
    for ensemble in payload["ensemble_cases"].values():
        if not ensemble.get("tail_certified", False):
            continue
        for record in ensemble["results"]:
            cache_key = f"{ensemble['N']}:{ensemble['ratio']:.12g}:{ensemble['dimensionless_step']:.12g}:R{record['R']}"
            if cache_key not in payload["R_case_cache"]:
                payload["R_case_cache"][cache_key] = {**ensemble, "results": [record], "wall_seconds": 0.0,
                    "minimum_tested_R_passing_all_seeds": record["R"] if record["passed_all_seeds"] else None,
                    "status": "reused_completed_ensemble"}
    def save():
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, payload)
    save()
    print(f"Study checkpoint: {path}", flush=True)
    for n in config["sizes"]:
        for ratio in config["ratios"]:
            if any(row["N"] == n and row["ratio"] == ratio for row in payload["exact_landscape"]):
                continue
            t = exact_crossing(n, ratio, config["steady_tolerance"], config["time_cap"])
            payload["exact_landscape"].append({"N": n, "ratio": ratio, "steady_time": t})
            save(); print(f"Exact N={n}, γ/J={ratio:g}: Jt_ss={t}", flush=True)
        key = str(n)
        if key not in payload["ratio_selections"]:
            rows = sorted((row for row in payload["exact_landscape"] if row["N"] == n and row["steady_time"] is not None), key=lambda row: row["steady_time"])
            if not rows:
                payload["ratio_selections"][key] = {"ratios": [], "status": "no_exact_crossing_within_cap"}; save(); continue
            best = rows[0]; index = config["ratios"].index(best["ratio"])
            candidates = [best["ratio"]]
            boundary = index in (0, len(config["ratios"])-1)
            if not boundary:
                fit = minimize_scalar(lambda log_r: exact_crossing(n, float(np.exp(log_r)), config["steady_tolerance"], config["time_cap"]) or config["time_cap"],
                    bounds=(np.log(config["ratios"][index-1]), np.log(config["ratios"][index+1])), method="bounded", options={"xatol": 1e-5})
                refined = float(np.exp(fit.x))
                t = exact_crossing(n, refined, config["steady_tolerance"], config["time_cap"])
                if t is not None:
                    payload["exact_landscape"].append({"N": n, "ratio": refined, "steady_time": t, "locally_refined": True})
                    candidates.insert(0, refined)
            candidates += [row["ratio"] for row in rows[1:]]
            payload["ratio_selections"][key] = {"ratios": candidates[:config["promising_ratios"]], "grid_optimum_at_boundary": boundary}
            save()
        passing = []
        for ratio in payload["ratio_selections"][key]["ratios"]:
            exact_time = next(row["steady_time"] for row in payload["exact_landscape"] if row["N"] == n and row["ratio"] == ratio)
            found = {}
            for budget, error, timing in (("infinite_R", config["dynamic_tolerance"], config["timing_tolerance"]),
                                           ("recommended", config["bias_tolerance"], config["bias_timing_tolerance"])):
                for step in config["steps"]:
                    case_key = f"{n}:{ratio:.12g}:{step:.12g}:{budget}"
                    computed = case_key not in payload["averaged_cases"]
                    if case_key not in payload["averaged_cases"]:
                        print(f"Averaged N={n}, r={ratio:.6g}, s={step:g}, budget={budget}", flush=True)
                        loose_key = f"{n}:{ratio:.12g}:{step:.12g}:infinite_R"
                        previous = payload["averaged_cases"].get(loose_key)
                        if budget == "recommended" and previous is not None and previous["status"] != "cap_reached":
                            case = {**previous, "tested_error_limit": error, "tested_timing_limit": timing,
                                "reused_infinite_R_case": True,
                                "passed": bool(previous["status"] == "complete" and max(previous["max_errors"].values()) <= error
                                    and previous["tail_error_certified_below"] <= error
                                    and previous["relative_steady_time_error"] <= timing)}
                            if max(case["max_errors"].values()) > error:
                                case["status"] = "transient_error_rejected"
                            payload["averaged_cases"][case_key] = case
                        else:
                            payload["averaged_cases"][case_key] = evaluate_step(n, ratio, step, exact_time, config, error_limit=error, timing_limit=timing)
                        save()
                    case = payload["averaged_cases"][case_key]
                    if computed:
                        print(f"  {case['status']}; error={max(case['max_errors'].values()):.5g}, passed={case['passed']}", flush=True)
                    if case["passed"]:
                        found[budget] = case; break
            if "recommended" in found:
                case = found["recommended"]
                passing.append({"ratio": ratio, "exact_time": exact_time, "case": case,
                                "infinite_R_step": found.get("infinite_R", {}).get("dimensionless_step"),
                                "cost": case["steady_time_bracket"][1]/case["dimensionless_step"]})
        if not passing:
            print(f"No recommended step found for N={n} within search caps", flush=True)
            continue
        best = min(passing, key=lambda row: row["cost"])
        refinement_brackets = {}
        for budget, error, timing, initial in (
                ("infinite_R", config["dynamic_tolerance"], config["timing_tolerance"], best["infinite_R_step"]),
                ("recommended", config["bias_tolerance"], config["bias_timing_tolerance"], best["case"]["dimensionless_step"])):
            lower = initial
            upper = min((s for s in config["steps"] if s > lower), default=None)
            if upper is not None:
                for _ in range(payload["step_refinement_rounds"]):
                    midpoint = (lower+upper)/2
                    case_key = f"{n}:{best['ratio']:.12g}:{midpoint:.12g}:{budget}"
                    if case_key not in payload["averaged_cases"]:
                        print(f"Refining N={n}, r={best['ratio']:.6g}, s={midpoint:g}, budget={budget}", flush=True)
                        payload["averaged_cases"][case_key] = evaluate_step(n, best["ratio"], midpoint, best["exact_time"], config, error_limit=error, timing_limit=timing)
                        save()
                    case = payload["averaged_cases"][case_key]
                    if case["passed"]:
                        lower = midpoint
                        if budget == "infinite_R":
                            best["infinite_R_step"] = midpoint
                        else:
                            best["case"] = case
                    else:
                        upper = midpoint
            refinement_brackets[budget] = [lower, upper]
        best["cost"] = best["case"]["steady_time_bracket"][1]/best["case"]["dimensionless_step"]
        ratio, step = best["ratio"], best["case"]["dimensionless_step"]
        ensemble_key = f"{n}:{ratio:.12g}:{step:.12g}"
        if ensemble_key not in payload["ensemble_cases"]:
            print(f"Testing finite R for N={n}, r={ratio:.6g}, s={step:g}", flush=True)
            ensemble = find_ensemble_size(n, ratio, step, best["exact_time"], config, payload["R_case_cache"], save)
            payload["ensemble_cases"][ensemble_key] = ensemble
            save()
        ensemble = payload["ensemble_cases"][ensemble_key]
        tested_Rs = [row["R"] for row in ensemble["results"]]
        extended_Rs = sorted(set(config["R_values"]) | set(payload["supplemental_R_values"]) | set(tested_Rs))
        if ensemble["minimum_tested_R_passing_all_seeds"] is None and max(extended_Rs) > max(tested_Rs):
            print(f"Extending unresolved N={n} ensemble case to R={max(extended_Rs)}", flush=True)
            ensemble = find_ensemble_size(n, ratio, step, best["exact_time"], {**config, "R_values": extended_Rs}, payload["R_case_cache"], save)
            ensemble["supplemental_R_extension"] = list(payload["supplemental_R_values"])
            payload["ensemble_cases"][ensemble_key] = ensemble
            save()
        observed_max_tau = best["case"]["steady_time_bracket"][1]
        if ensemble["minimum_tested_R_passing_all_seeds"] is not None:
            validated = next(row for row in ensemble["results"] if row["R"] == ensemble["minimum_tested_R_passing_all_seeds"])
            observed_max_tau = max(observed_max_tau, max(row["steady_time_bracket"][1] for row in validated["per_seed"]))
        J = max(config["rate_margin"]*best["exact_time"], observed_max_tau)/config["target_time"]
        recommendation = {"N": n, "ratio": ratio, "exact_dimensionless_steady_time": best["exact_time"],
            "largest_tested_infinite_R_step": best["infinite_R_step"], "recommended_dimensionless_step": step,
            "minimum_tested_R": ensemble["minimum_tested_R_passing_all_seeds"], "J_for_target": J,
            "gamma_for_target": ratio*J, "delta_t_for_target": step/J,
            "approximate_substep_count": int(round(best["cost"])),
            "substeps_to_target_time": int(np.ceil(config["target_time"]*J/step)),
            "observed_max_dimensionless_steady_time": observed_max_tau,
            "step_refinement_brackets": refinement_brackets,
            "grid_optimum_at_boundary": payload["ratio_selections"][key]["grid_optimum_at_boundary"]}
        payload["recommendations"] = sorted([row for row in payload["recommendations"] if row["N"] != n]+[recommendation], key=lambda row: row["N"])
        save(); print(f"RECOMMENDATION: {recommendation}", flush=True)
        write_report(payload, directory / "Study.md")
    payload["all_recommendations_validated_at_tested_R"] = all(row["minimum_tested_R"] is not None for row in payload["recommendations"]) and len(payload["recommendations"]) == len(config["sizes"])
    payload["status"] = "complete" if payload["all_recommendations_validated_at_tested_R"] else "complete_with_search_limits"
    save()
    if not options.no_plots:
        payload["figures"] = plots(payload, directory)
        save()
    write_report(payload, directory / "Study.md")
    print(f"Saved study: {directory}; status={payload['status']}", flush=True)
    return payload


if __name__ == "__main__":
    main(parse_arguments())
