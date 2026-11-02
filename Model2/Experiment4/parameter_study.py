"""Checkpointed Model2 rate/timestep study with fixed-count annealing fits.

For equal reservoirs and a half-mixture, C=pI+(1/2-p)K. The homogeneous
map for K is independent of p. Batched bond rotations accelerate search;
selected schedules are checked with the original affine channel separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library import reservoir_scaling as physics
from Model2.Experiment4 import cosine_search as original


def matrices(dts, n, ratio):
    """Even/odd Model2 rotations and rate-weighted homogeneous Schur masks."""
    dts = np.atleast_1d(dts)
    U = np.broadcast_to(np.eye(n), (len(dts), n, n)).copy()
    c, s = np.cos(dts/2)[:, None, None], np.sin(dts/2)[:, None, None]
    for parity in (0, 1):
        left = np.arange(parity, n-1, 2)
        a, b = U[:, left, :].copy(), U[:, left+1, :].copy()
        U[:, left, :], U[:, left+1, :] = c*a-s*b, s*a+c*b
    bath = np.cos(np.sqrt(2*ratio*dts))
    W = np.ones_like(U)
    off = (1+bath)/2
    for boundary in (0, n-1):
        W[:, boundary, 1:-1] = off[:, None]
        W[:, 1:-1, boundary] = off[:, None]
        W[:, boundary, boundary] = (1+bath*bath)/2
    W[:, 0, -1] = W[:, -1, 0] = bath
    return U, W


def advance(K, U, W, order="jump-first"):
    if order == "jump-first":
        return U@(W*K)@np.swapaxes(U, -1, -2)
    return W*(U@K@np.swapaxes(U, -1, -2))


def fast_output(dts, config):
    U, W = matrices(dts, config["n_qubits"], config["gamma"])
    K = np.eye(config["n_qubits"])
    for u, w in zip(U, W):
        K = advance(K, u, w, config["order"])
    return K


def error(K, config):
    # K is PSD. For c=1/4, |K_ij|<=max diag(K), so this checks currents too.
    return float(abs(.5-config["p"])*np.max(np.diag(K)))


def objective(parameters, family, count, config):
    dts = original.schedule(family, count, parameters)
    if np.any(dts < config["minimum_dt"]-1e-12) or np.any(dts > config["maximum_dt"]+1e-12):
        return 1.
    if dts.sum() > config["maximum_time"]:
        return 1.+float(dts.sum()/config["maximum_time"])
    K = fast_output(dts, config)
    maximum = error(K, config)
    if maximum <= config["mean_tolerance"]:
        # Evaluate the entire final-dt window only near a feasible output.
        U, W = matrices([dts[-1]], config["n_qubits"], config["gamma"])
        for _ in range(int(np.ceil(config["forward_window"]/dts[-1]))):
            K = advance(K, U[0], W[0], config["order"])
            maximum = max(maximum, error(K, config))
    return maximum


def constant_search(config):
    """Batched timestep scan, followed by multiple bounded local refinements."""
    low, high = config["minimum_dt"], config["maximum_dt"]
    r = config["gamma"]
    special = [np.pi]+[(2*k+1)**2*np.pi**2/(8*r) for k in range(10)]
    grid = np.unique(np.r_[np.linspace(low, high, config["grid_points"]),
        np.geomspace(low, high, 80), [x for x in special if low <= x <= high]])
    U, W = matrices(grid, config["n_qubits"], r)
    K = np.broadcast_to(np.eye(config["n_qubits"]), U.shape).copy()
    counts = np.zeros(len(grid), dtype=int)
    for count in range(1, config["step_budget"]+1):
        K = advance(K, U, W, config["order"])
        errors = abs(.5-config["p"])*np.max(np.diagonal(K, axis1=-2, axis2=-1), axis=-1)
        passing = (errors <= config["mean_tolerance"]) & (counts == 0)
        counts[passing] = count
    if not np.any(counts):
        return None
    count = int(counts[counts > 0].min())
    seed_indices = np.flatnonzero(counts == count)
    best = None
    probes = []
    for candidate_count in (count, count-1):
        if candidate_count < 1:
            continue
        def f(dt):
            return error(fast_output(np.full(candidate_count, dt), config), config)
        indices = sorted(seed_indices, key=lambda i: f(grid[i]))[:8]
        trial = min((f(grid[i]), float(grid[i])) for i in indices)
        for index in indices:
            lo, hi = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
            opt = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
            trial = min(trial, (float(opt.fun), float(opt.x)))
        probes.append({"steps": candidate_count, "error": trial[0], "dt": trial[1]})
        if trial[0] <= config["mean_tolerance"]:
            best = {"steps": candidate_count, "dt": trial[1], "error": trial[0]}
    return {"best": best, "probes": probes, "grid_points": len(grid)}


def run_case(config, progress):
    baseline = constant_search(config)
    if baseline is None:
        return {"constant": None, "trials": [], "selected": None}
    b = baseline["best"]
    best = {"family": "constant", "steps": b["steps"], "parameters": [float(np.log(b["dt"]))], "error": b["error"]}
    trials = []
    anchor = b["steps"]
    counts = sorted(set([anchor-1, int(np.floor(.9*anchor)), int(np.floor(.75*anchor))]), reverse=True)
    counts = sorted(set(counts+list(range(anchor-1, anchor-config.get("neighbor_counts", 1)-1, -1))), reverse=True)
    counts = [c for c in counts if c >= 1]
    for family in config["families"]:
        bounds = original.parameter_bounds(family, config)
        for count in counts:
            for seed in config["optimizer_seeds"]:
                started = time.perf_counter()
                retained = None
                calls = 0
                def f(parameters):
                    nonlocal retained, calls
                    calls += 1
                    value = objective(parameters, family, count, config)
                    if retained is None or value < retained["error"]:
                        retained = {"family": family, "steps": count,
                                    "parameters": list(map(float, parameters)), "error": value}
                    return value
                optimum = differential_evolution(f, bounds, seed=seed, maxiter=config["iterations"],
                    popsize=8, polish=False, tol=1e-6)
                constant = [np.log(b["dt"]), np.log(b["dt"])]
                if family == "warped_cosine":
                    constant.append(0.)
                # Independent global point and a constant-family control.
                for start in (optimum.x, constant):
                    minimize(f, start, method="Nelder-Mead", bounds=bounds,
                             options={"maxiter": 120, "xatol": 1e-7, "fatol": 1e-10})
                row = {**retained, "seed": seed, "evaluations": calls,
                       "wall_seconds": time.perf_counter()-started}
                trials.append(row)
                if retained["error"] <= config["mean_tolerance"] and (count, retained["error"]) < (best["steps"], best["error"]):
                    best = retained.copy()
                progress({"constant": baseline, "trials": trials, "selected": best}, row)
    dts = original.schedule(best["family"], best["steps"], best["parameters"])
    model = physics.Model(config["n_qubits"], config["gamma"], config["p"], config["p"], .25)
    validation = original.evaluate(model, dts, config)
    validation.update(family=best["family"], parameters=best["parameters"])
    if not validation["passed"]:
        raise ValueError("selected accelerated candidate failed original affine validation")
    fast_C = config["p"]*np.eye(model.n)+(.5-config["p"])*fast_output(dts, config)
    agreement = float(np.max(np.abs(fast_C-np.asarray(validation["output_covariance"]))))
    if agreement > 1e-11:
        raise ValueError("accelerated and original covariance disagree")
    return {"constant": baseline, "trials": trials, "selected": validation, "affine_agreement": agreement}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=[.25, np.pi/8, .5, 1., 2., 4., 8.])
    parser.add_argument("--p-values", type=float, nargs="+", default=[.25])
    parser.add_argument("--orders", nargs="+", choices=["jump-first", "hamiltonian-first"], default=["jump-first"])
    parser.add_argument("--maximum-dt", type=float, default=8.)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[2718, 3141])
    parser.add_argument("--families", nargs="+", choices=["cosine", "warped_cosine"], default=["cosine", "warped_cosine"])
    parser.add_argument("--grid-points", type=int, default=321)
    parser.add_argument("--neighbor-counts", type=int, default=1)
    parser.add_argument("--maximum-power", type=float, default=12.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent/"runs")
    args = parser.parse_args(arguments)
    if (not np.all(np.isfinite(args.ratios+args.p_values+[args.maximum_dt,args.maximum_power]))
            or min(args.ratios) <= 0 or any(not 0 < p < .5 for p in args.p_values) or args.maximum_dt <= .001):
        raise ValueError("positive ratios, 0<p<0.5 and maximum dt>0.001 required")
    if args.iterations < 1 or args.neighbor_counts < 1 or args.grid_points < 3 or args.maximum_power < .03:
        raise ValueError("invalid search effort, grid or power bounds")
    if min(args.optimizer_seeds) < 0 or len(set(args.optimizer_seeds)) != len(args.optimizer_seeds):
        raise ValueError("optimizer seeds must be distinct and nonnegative")
    _, config = original.parse_arguments(["--maximum-dt", str(args.maximum_dt)])
    config.update(iterations=args.iterations, optimizer_seeds=args.optimizer_seeds,
                  grid_points=args.grid_points, maximum_time=2000., families=args.families,
                  minimum_dt=.001, minimum_power=.03, maximum_power=args.maximum_power, neighbor_counts=args.neighbor_counts)
    specification = {"base": config, "ratios": args.ratios, "p_values": args.p_values, "orders": args.orders}
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:12]
    directory = args.output_directory/f"parameter_study_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"study.json"
    study = json.loads(path.read_text()) if args.resume and path.exists() else {"specification": specification, "cases": {}}
    for ratio in args.ratios:
        for p in args.p_values:
            for order in args.orders:
                key = f"ratio{ratio:.12g}_p{p:g}_{order}"
                if study["cases"].get(key, {}).get("complete"):
                    continue
                case_config = {**config, "gamma": ratio, "p": p, "order": order}
                def progress(result, row):
                    study["cases"][key] = {"configuration": case_config, **result, "complete": False}
                    physics.atomic_json(path, study)
                    print(f"{key}, {row['family']} M={row['steps']} seed={row['seed']}: {row['error']:.9g}", flush=True)
                started = time.perf_counter()
                result = run_case(case_config, progress)
                study["cases"][key] = {"configuration": case_config, **result, "complete": True,
                                       "wall_seconds": time.perf_counter()-started}
                physics.atomic_json(path, study)
                if result["selected"] is None:
                    print(f"COMPLETE {key}: no passing constant-grid baseline found", flush=True)
                else:
                    print(f"COMPLETE {key}: constant={result['constant']['best']}, selected={result['selected']['steps']}", flush=True)
    study["complete"] = True
    physics.atomic_json(path, study)
    print(f"Saved {path}", flush=True)
    return study


if __name__ == "__main__":
    main()
