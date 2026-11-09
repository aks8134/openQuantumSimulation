"""Offline XX scaling: continuous references, digital schedules and resources.

Run from the repository root; each stage checkpoints one (model, N) case.
Only resources imports Qiskit. No execution/provider API is used.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library import reservoir_scaling as physics


def validate(config):
    if not config["sizes"] or any(type(n) is not int or not 2 <= n <= 100 for n in config["sizes"]):
        raise ValueError("sizes must contain system-qubit counts between 2 and 100")
    positive = ("mean_tolerance", "synthesis_allowance", "time_cap", "fixed_dt", "minimum_dt",
                "maximum_dt", "minimum_power", "maximum_power", "forward_window")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive):
        raise ValueError("tolerances, times and power limits must be positive and finite")
    if not np.isfinite(config["sampling_allowance"]) or config["sampling_allowance"] < 0:
        raise ValueError("sampling_allowance must be finite and nonnegative")
    for key in ("step_budget", "validation_step_cap", "iterations", "population_multiplier",
                "resource_paths", "maximum_compile_exchanges"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["minimum_dt"] > config["maximum_dt"] or config["minimum_power"] > config["maximum_power"]:
        raise ValueError("reversed search bounds")
    if not config["optimizer_seeds"] or any(type(s) is not int or s < 0 for s in config["optimizer_seeds"]):
        raise ValueError("optimizer_seeds must be nonnegative integers")
    for key in ("resource_seed", "transpiler_seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not config["models"]:
        raise ValueError("at least one model is required")
    for spec in config["models"].values():
        if any(not np.isfinite(spec[k]) or spec[k] <= 0 for k in ("J", "gamma", "prefactor")):
            raise ValueError("positive finite J, gamma and prefactor required")
        if any(not np.isfinite(spec[k]) or not 0 <= spec[k] <= 1 for k in ("p_left", "p_right")):
            raise ValueError("reservoir populations must be in [0,1]")
        if spec["preparation"] not in ("center_excitation", "half_mixed", "vacuum", "filled"):
            raise ValueError("unsupported preparation")


def model_for(n, spec):
    return physics.Model(n, spec["gamma"]/spec["J"], spec["p_left"], spec["p_right"], spec["prefactor"])


@lru_cache(maxsize=128)
def stationary(model):
    return physics.exact_stationary(model)


def initial(n, preparation):
    if preparation == "half_mixed":
        return np.eye(n)/2
    if preparation == "filled":
        return np.eye(n)
    value = np.zeros((n, n))
    if preparation == "center_excitation":
        value[n//2, n//2] = 1
    elif preparation != "vacuum":
        raise ValueError("unknown preparation")
    return value


def error(state, target, model):
    return max(physics.observable_error(state, target, model.prefactor).values())


def exact_case(n, spec, config):
    """Crossing of a permanent sufficient bound, not first local crossing."""
    started = time.perf_counter()
    model = model_for(n, spec)
    A = spec["J"]*physics.generator(model)
    target = stationary(model)
    delta = initial(n, spec["preparation"])-target
    factor = max(1., 4*model.prefactor)

    def at(t):
        E = expm(t*A)
        difference = E@delta@E.T
        return factor*float(np.linalg.norm(difference, "fro")), target+difference

    low, high = 0., min(1./spec["J"], config["time_cap"])
    if at(0)[0] <= config["mean_tolerance"]:
        high = 0.
    else:
        while at(high)[0] > config["mean_tolerance"] and high < config["time_cap"]:
            low, high = high, min(2*high, config["time_cap"])
        if at(high)[0] > config["mean_tolerance"]:
            return {"status": "time_cap_reached", "settling_time_upper_bound": None}
        for _ in range(60):
            if high-low <= 1e-8*max(1., high):
                break
            middle = (low+high)/2
            if at(middle)[0] <= config["mean_tolerance"]:
                high = middle
            else:
                low = middle
    bound, state = at(high)
    source = np.zeros_like(A)
    source[0, 0], source[-1, -1] = spec["gamma"]*model.p_left, spec["gamma"]*model.p_right
    return {"status": "permanent_bound_met", "settling_time_upper_bound": high,
            "crossing_bracket": [low, high], "bound_at_upper": bound,
            "local_error_at_upper": error(state, target, model),
            "stationary_residual": float(np.linalg.norm(A@target+target@A.T+source)),
            "wall_seconds": time.perf_counter()-started,
            "criterion": "max(1,4c)*||C(t)-C_star||_F <= mean_tolerance; permanent local-error upper bound"}


def fast_step(state, model, dt, J):
    """O(N^2) mean step; same rate-weighted single-jump map as the library.

    Real-gauge Hamiltonian parity rotations, followed by one random bath.
    Pump/loss initial mixtures are averaged, not chosen deterministically.
    """
    state = state.copy()
    angle = 2*model.prefactor*J*dt
    c, s = np.cos(angle), np.sin(angle)
    for parity in (0, 1):
        left = np.arange(parity, model.n-1, 2)
        a, b = state[left].copy(), state[left+1].copy()
        state[left], state[left+1] = c*a-s*b, s*a+c*b
        a, b = state[:, left].copy(), state[:, left+1].copy()
        state[:, left], state[:, left+1] = c*a-s*b, s*a+c*b
    bath = np.cos(np.sqrt(2*model.ratio*J*dt))
    for boundary, p in ((0, model.p_left), (model.n-1, model.p_right)):
        state[boundary, 1:-1] *= (1+bath)/2
        state[1:-1, boundary] *= (1+bath)/2
        state[boundary, boundary] = (1+bath*bath)/2*state[boundary, boundary]+p*(1-bath*bath)/2
    state[0, -1] *= bath
    state[-1, 0] *= bath
    return state


def schedule(family, count, parameters):
    if family == "constant":
        return np.full(count, np.exp(parameters[0]))
    low, high = sorted(np.exp(parameters[:2]))
    if count == 1:
        return np.array([low])
    power = np.exp(parameters[2])
    result = low+(high-low)/2*(1+np.cos(np.pi*np.linspace(0, 1, count)**power))
    result[0], result[-1] = high, low
    return result


def evaluate(dts, model, spec, config, *, independent=False):
    state = initial(model.n, spec["preparation"])
    target = stationary(model)
    def advance(C, dt):
        if independent:
            branch = physics.channel(model, spec["J"]*dt, "rate-weighted", "hamiltonian-first", "even-odd")
            return physics.mean_step(C, branch)
        return fast_step(C, model, dt, spec["J"])
    for dt in dts:
        state = advance(state, float(dt))
    output = state.copy()
    maximum = error(state, target, model)
    count = int(np.ceil(config["forward_window"]/dts[-1]))
    if maximum <= config["mean_tolerance"] and count <= config["validation_step_cap"]:
        for _ in range(count):
            state = advance(state, float(dts[-1]))
            maximum = max(maximum, error(state, target, model))
            if maximum > config["mean_tolerance"]:
                break
    passed = maximum <= config["mean_tolerance"] and count <= config["validation_step_cap"]
    eigenvalues = np.linalg.eigvalsh(output) if independent else None
    if eigenvalues is not None:
        passed = passed and eigenvalues[0] >= -1e-9 and eigenvalues[-1] <= 1+1e-9
    return {"passed": bool(passed), "output_error": error(output, target, model),
            "maximum_forward_error": maximum, "forward_steps": count,
            "forward_duration": float(count*dts[-1]),
            "output_covariance": output.tolist() if independent else None}


def schedules_case(n, spec, config):
    started = time.perf_counter()
    model = model_for(n, spec)
    results = {}
    # A fixed-step baseline counts real channel applications, not ceil(t_exact/dt).
    state = initial(n, spec["preparation"])
    target = stationary(model)
    fixed = None
    for count in range(config["step_budget"]+1):
        if error(state, target, model) <= config["mean_tolerance"] and count > 0:
            dts = np.full(count, config["fixed_dt"])
            check = evaluate(dts, model, spec, config)
            if check["passed"]:
                fixed = dts
                break
        if (count+1)*config["fixed_dt"] > config["time_cap"]:
            break
        state = fast_step(state, model, config["fixed_dt"], spec["J"])
    candidates = [("fixed", fixed, [])]
    for family in ("constant", "warped_cosine"):
        incumbent, trials = None, []
        if family == "constant" and fixed is not None and config["minimum_dt"] <= config["fixed_dt"] <= config["maximum_dt"]:
            incumbent = fixed.copy()
        bounds = [(1, config["step_budget"]), (np.log(config["minimum_dt"]), np.log(config["maximum_dt"]))]
        if family == "warped_cosine":
            bounds += [bounds[1], (np.log(config["minimum_power"]), np.log(config["maximum_power"]))]
        # Retain a passing constant as a valid equal-endpoint warped candidate.
        previous = results.get("constant", {}).get("step_dts")
        if family == "warped_cosine" and previous:
            incumbent = np.asarray(previous)
        for seed in config["optimizer_seeds"]:
            calls = 0
            def objective(parameters):
                nonlocal incumbent, calls
                calls += 1
                dts = schedule(family, int(round(parameters[0])), parameters[1:])
                if dts.sum() > config["time_cap"]:
                    return 2.+dts.sum()/config["time_cap"]
                check = evaluate(dts, model, spec, config)
                if check["passed"]:
                    if incumbent is None or len(dts) < len(incumbent):
                        incumbent = dts.copy()
                    return len(dts)/(config["step_budget"]+1)
                return 1.+check["maximum_forward_error"]/config["mean_tolerance"]
            optimum = differential_evolution(objective, bounds, integrality=[True]+[False]*(len(bounds)-1),
                seed=seed, maxiter=config["iterations"], popsize=config["population_multiplier"], polish=False)
            trials.append({"seed": seed, "calls": calls, "optimizer_converged": bool(optimum.success),
                           "objective": float(optimum.fun)})
        candidates.append((family, incumbent, trials))
        # Make the constant available before constructing the warped search.
        results[family] = {"step_dts": None if incumbent is None else incumbent.tolist()}
    for family, dts, trials in candidates:
        record = {"status": "no_passing_schedule_found", "selected_steps": None, "trials": trials}
        if dts is not None:
            check = evaluate(dts, model, spec, config, independent=True)
            if not check["passed"]:
                raise RuntimeError("Selected schedule failed independent dense-channel validation")
            record.update(status="mean_forward_window_passed", selected_steps=len(dts),
                          physical_time=float(dts.sum()), step_dts=dts.tolist(), validation=check)
        results[family] = record
    return {"status": "search_complete", "families": results, "wall_seconds": time.perf_counter()-started,
            "minimality": "smallest passing count found within this search; no minimum proof",
            "sampling": "not run; deterministic mean only",
            "settling": "finite constant-final-step mean window, not permanent convergence"}


def build_circuit(n, spec, dts, seed):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    circuit = QuantumCircuit(n+2, 2*n)
    if spec["preparation"] == "center_excitation":
        circuit.x(n//2+1)
    elif spec["preparation"] == "filled":
        circuit.x(list(range(1, n+1)))
    elif spec["preparation"] == "half_mixed":
        circuit.h(list(range(1, n+1)))
        circuit.measure(list(range(1, n+1)), list(range(n)))
    probabilities = np.array([spec["p_left"], 1-spec["p_left"], spec["p_right"], 1-spec["p_right"]])/2
    choices = np.random.default_rng(seed).choice(4, size=len(dts), p=probabilities)
    for dt, choice in zip(dts, choices):
        for parity in (0, 1):
            for j in range(parity, n-1, 2):
                circuit.append(XXPlusYYGate(4*spec["prefactor"]*spec["J"]*dt), [j+1, j+2])
        bath, site = (0, 1) if choice < 2 else (n+1, n)
        if choice % 2 == 0:
            circuit.x(bath)
        circuit.append(XXPlusYYGate(2*np.sqrt(2*spec["gamma"]*dt)), [site, bath])
        circuit.reset(bath)
    circuit.measure(list(range(1, n+1)), list(range(n, 2*n)))
    return circuit


def rotation_estimate(angles, allowance, prefactor):
    """Leading-order Ross--Selinger estimate, NOT synthesized T gates."""
    generic, exact_t = 0, 0
    for angle in angles:
        units = angle/(np.pi/4)
        if abs(units-round(units)) < 1e-10:
            exact_t += int(round(units)) % 2
        else:
            generic += 1
    # ||O|| <= max(1,4c); telescoping + expectation error <=2||O|| R eta.
    eta = None if generic == 0 else allowance/(2*max(1., 4*prefactor)*generic)
    per_rotation = 0 if eta is None else int(np.ceil(3*np.log2(1/eta)))
    return {"generic_rotations": generic, "exact_T_or_Tdg_rotations": exact_t,
            "per_rotation_operator_error_budget": eta,
            "T_count_leading_order_estimate": exact_t+generic*per_rotation,
            "method": "3 log2(1/eta), rounded up; omitted asymptotic terms; not an upper bound or actual synthesis",
            "T_depth": None, "physical_qubits": None, "Clifford_count_after_T_synthesis": None}


def resources_case(n, spec, config, schedules):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    records = []
    for family, selected in schedules["families"].items():
        if selected["selected_steps"] is None:
            records.append({"family": family, "status": "no_validated_schedule"})
            continue
        dts = selected["step_dts"]
        if n*len(dts) > config["maximum_compile_exchanges"]:
            records.append({"family": family, "status": "compile_size_cap", "logical_exchanges": n*len(dts)})
            continue
        for path in range(config["resource_paths"]):
            circuit = build_circuit(n, spec, dts, config["resource_seed"]+path)
            for topology in ("all_to_all", "line"):
                started = time.perf_counter()
                coupling = None if topology == "all_to_all" else CouplingMap.from_line(n+2)
                native = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], coupling_map=coupling,
                    initial_layout=list(range(n+2)), optimization_level=1, seed_transpiler=config["transpiler_seed"])
                axial = transpile(circuit, basis_gates=["rz", "h", "s", "sdg", "x", "z", "cx"],
                    coupling_map=coupling, initial_layout=list(range(n+2)), optimization_level=1,
                    seed_transpiler=config["transpiler_seed"])
                angles = [float(i.operation.params[0]) for i in axial.data if i.operation.name == "rz"]
                record = {"family": family, "path": path, "topology": topology, "status": "compiled",
                    "system_qubits": n, "logical_qubits": n+2, "steps": len(dts),
                    "logical_exchanges": n*len(dts), "native_operations": dict(native.count_ops()),
                    "native_depth": native.depth(),
                    "two_qubit_depth": native.depth(filter_function=lambda i: i.operation.num_qubits == 2),
                    "clifford_plus_rz_operations": dict(axial.count_ops()),
                    "fault_tolerant": rotation_estimate(angles, config["synthesis_allowance"], spec["prefactor"]),
                    "wall_seconds": time.perf_counter()-started}
                records.append(record)
    return {"status": "resource_study_complete", "records": records,
            "scope": "per Z-readout trajectory; compiled circuits and leading-order logical T estimates; no execution"}


def write_summary(directory, stage, records):
    rows = []
    for item in records:
        base = {"model": item["model"], "N": item["N"]}
        result = item["result"]
        if stage == "exact":
            rows.append({**base, **{k: result.get(k) for k in ("status", "settling_time_upper_bound", "bound_at_upper", "local_error_at_upper", "wall_seconds")}})
        elif stage == "schedules":
            for family, value in result["families"].items():
                rows.append({**base, "family": family, **{k: value.get(k) for k in ("status", "selected_steps", "physical_time")}})
        else:
            for value in result["records"]:
                rows.append({**base, **{k: value.get(k) for k in ("family", "status", "path", "topology", "logical_exchanges", "native_depth", "two_qubit_depth")},
                             "native_cx": value.get("native_operations", {}).get("cx"),
                             "T_estimate": value.get("fault_tolerant", {}).get("T_count_leading_order_estimate")})
    if not rows:
        return
    with (directory/f"{stage}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {stage.capitalize()} scaling results", "", "See Experiment1.md for definitions, assumptions and commands.", "",
             "| "+" | ".join(rows[0])+" |", "|"+"|".join("---" for _ in rows[0])+"|"]
    lines += ["| "+" | ".join(str(v) if v is not None else "—" for v in row.values())+" |" for row in rows]
    (directory/f"{stage.capitalize()}.md").write_text("\n".join(lines)+"\n")
    if stage == "exact":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in sorted({r["model"] for r in rows}):
            group = sorted((r for r in rows if r["model"] == name and r["settling_time_upper_bound"] is not None), key=lambda r:r["N"])
            ax.loglog([r["N"] for r in group], [r["settling_time_upper_bound"] for r in group], "o-", label=name)
        ax.set(xlabel="System qubits N", ylabel="Permanent local-settling time upper bound")
        ax.legend()
        ax.grid(alpha=.25, which="both")
        fig.tight_layout()
        fig.savefig(directory/"exact.png", dpi=180)
        fig.savefig(directory/"exact.pdf")
        plt.close(fig)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("exact", "schedules", "resources"))
    parser.add_argument("--config", type=Path, default=DIRECTORY/"config.json")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--models", nargs="+", choices=("model1", "model2", "model3"))
    parser.add_argument("--output", type=Path, default=DIRECTORY/"runs")
    args = parser.parse_args(arguments)
    config = json.loads(args.config.read_text())
    if args.sizes is not None:
        config["sizes"] = args.sizes
    if args.models:
        config["models"] = {k: config["models"][k] for k in args.models}
    validate(config)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "qiskit")}
    provenance = {"configuration": config, "versions": versions,
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__).resolve(), ROOT/"library/reservoir_scaling.py")}}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    directory = args.output/f"scaling_{signature}"
    directory.mkdir(parents=True, exist_ok=True)
    physics.atomic_json(directory/"configuration.json", provenance)
    records = []
    for name, spec in config["models"].items():
        for n in config["sizes"]:
            path = directory/f"{args.stage}_{name}_N{n}.json"
            if path.exists():
                item = json.loads(path.read_text())
            else:
                print(f"{args.stage}: {name}, N={n}", flush=True)
                if args.stage == "exact":
                    result = exact_case(n, spec, config)
                elif args.stage == "schedules":
                    result = schedules_case(n, spec, config)
                else:
                    source = directory/f"schedules_{name}_N{n}.json"
                    if not source.exists():
                        raise ValueError(f"Run schedules with the same configuration first: {source}")
                    result = resources_case(n, spec, config, json.loads(source.read_text())["result"])
                item = {"model": name, "N": n, "parameters": spec, "result": result}
                physics.atomic_json(path, item)
            records.append(item)
    write_summary(directory, args.stage, records)
    print(directory/f"{args.stage.capitalize()}.md", flush=True)
    return directory


if __name__ == "__main__":
    main()
