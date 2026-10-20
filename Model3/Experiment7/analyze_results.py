"""Independently verify saved schedules and compare with existing baselines.

No sampled trajectory evolution or new schedule optimization is performed.
The known earlier schedules are read ONLY here, after controller selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import reservoir_scaling as physics
from Model3.Experiment5.circuit_depth import validate_source


def read(path, hashes):
    raw = path.read_bytes()
    hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def distance(A, B, prefactor):
    delta = np.asarray(A)-np.asarray(B)
    return float(max(np.max(np.abs(np.diagonal(delta, axis1=-2, axis2=-1))),
                     4*prefactor*np.max(np.abs(np.diagonal(delta, offset=1, axis1=-2, axis2=-1)))))


def depth_summary(path, hashes):
    data = read(path, hashes)
    assert data["status"] == "complete"
    source = Path(data["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == data["configuration"]["source_sha256"]
    records = [r for r in data["records"] if r["steps"] == max(data["configuration"]["prefix_steps"])]
    depths = [r["readouts"][basis]["depth"] for r in records for basis in ("z", "xy", "yx")]
    unitary = [sum(count for gate, count in r["readouts"][basis]["operation_counts"].items()
                   if gate not in ("reset", "measure", "barrier"))
               for r in records for basis in ("z", "xy", "yx")]
    return {"file": str(path.resolve()), "full_depth_range": [min(depths), max(depths)],
        "unitary_gate_range_including_rz": [min(unitary), max(unitary)],
        "cz_range": [min(r["native_body"]["operation_counts"].get("cz", 0) for r in records),
                     max(r["native_body"]["operation_counts"].get("cz", 0) for r in records)],
        "two_qubit_depth_range": [min(r["native_body"]["two_qubit_depth"] for r in records),
                                  max(r["native_body"]["two_qubit_depth"] for r in records)],
        "active_qubits": sorted({r["native_body"]["active_qubits"] for r in records}),
        "trajectory_count": len(records)}


def main(folder):
    hashes = {}
    study = read(folder/"study.json", hashes)
    config = study["configuration"]
    assert set(study["methods"]) == set(config["methods"])
    model = physics.Model(config["n_qubits"], config["gamma"]/config["J"],
                          config["p_left"], config["p_right"], config["exchange_prefactor"])
    target = physics.exact_stationary(model)
    report = {"experiment": "Model3/Experiment7", "methods": {}, "baselines": {}, "source_sha256": hashes}
    for method, record in study["methods"].items():
        directory = folder/method
        plan = read(directory/"plan.json", hashes)
        selected = plan["selected"]
        dts = np.asarray(selected["step_dts"])
        assert len(dts) == selected["steps"] <= config["controller"]["step_budget"]
        assert np.all(dts >= config["controller"]["minimum_dt"]*(1-1e-14))
        assert np.all(dts <= config["controller"]["maximum_dt"]*(1+1e-14))
        state = np.diag(plan["selected_preparation"]["populations"])
        errors = [distance(state, target, model.prefactor)]
        for dt in dts:
            state = physics.mean_step(state, physics.channel(model, config["J"]*dt,
                config["sampling_policy"], config["order"], config["system_formula"]))
            errors.append(distance(state, target, model.prefactor))
        np.testing.assert_allclose(state, selected["final_mean_covariance_real_gauge"], atol=1e-13)
        np.testing.assert_allclose(errors, plan["preparation_mean_errors"], atol=1e-13)
        future = selected["forward_validation"]
        tail = state.copy()
        future_errors = [errors[-1]]
        for dt in future["step_dts"]:
            assert dt == dts[-1]
            tail = physics.mean_step(tail, physics.channel(model, config["J"]*dt,
                config["sampling_policy"], config["order"], config["system_formula"]))
            future_errors.append(distance(tail, target, model.prefactor))
        np.testing.assert_allclose(future_errors, future["mean_errors"], atol=1e-13)
        assert max(future_errors) <= config["controller"]["mean_tolerance"]
        assert future["times"][-1]-future["times"][0] >= config["controller"]["steady_window"]-1e-10
        method_result = {"steps": len(dts), "physical_time": float(dts.sum()),
            "mean_output_error": errors[-1], "mean_window_error": max(future_errors),
            "final_dt": float(dts[-1]), "classical_forward_steps": len(future["step_dts"]),
            "planning_wall_seconds": plan["planning_wall_seconds"], "replays": {}}
        for key, row in record["replays"].items():
            replay = read(directory/row["file"], hashes)
            validate_source(replay)
            assert replay["schedule"]["step_dts"] == dts.tolist()
            covs = np.asarray(replay["final_covariances_real_gauge"]).reshape(-1, model.n, model.n)
            R = len(covs)
            assert R > 1
            average = covs.mean(axis=0)
            sampled_error = distance(average, target, model.prefactor)
            sampling_distance = distance(average, state, model.prefactor)
            two_se = 2*max(np.diagonal(covs, axis1=-2, axis2=-1).std(axis=0, ddof=1).max(),
                4*model.prefactor*np.diagonal(covs, offset=1, axis1=-2, axis2=-1).std(axis=0, ddof=1).max())/np.sqrt(R)
            values = {"output_sampled_error": sampled_error, "max_sampling_distance": sampling_distance,
                      "max_two_standard_errors": float(two_se)}
            for name, value in values.items():
                np.testing.assert_allclose(value, row["assessment"][name], atol=1e-13)
            assert sampled_error <= config["steady_tolerance"]
            assert sampling_distance <= config["sampling_tolerance"]
            assert two_se <= config["sampling_tolerance"]
            assert row["status"] == replay["run_status"] == "passed"
            np.testing.assert_allclose(replay["mean_forward_validation"]["local_observable_distance"], future_errors, atol=1e-13)
            method_result["replays"][key] = values
        assert len(method_result["replays"]) == len(config["R_values"])*len(config["seeds"])
        depths = sorted(directory.glob("depth/depth_*/depth.json"))
        if depths:
            assert len(depths) == 1
            method_result["depth"] = depth_summary(depths[0], hashes)
        report["methods"][method] = method_result

    for name, path in (("earlier_automatic", ROOT/"Model3/Experiment6/runs/adaptive_6ea9f15a44d8/study.json"),
                       ("earlier_searched", ROOT/"Model3/Experiment4/runs/budget_419029a6f9bc/study.json")):
        baseline = read(path, hashes)
        bconfig, selected = baseline["configuration"], baseline["selected"]
        for field in ("n_qubits", "J", "gamma", "p_left", "p_right", "exchange_prefactor", "steady_tolerance",
                      "sampling_tolerance", "sampling_policy", "order", "system_formula"):
            assert bconfig[field] == config[field]
        assert selected["preparation"] == config["preparation"]
        assert selected["validation_mode"] == "forward"
        assert selected["window_end"]-selected["final_time"] >= config["controller"]["steady_window"]-1e-10
        report["baselines"][name] = {"file": str(path.resolve()), "steps": selected["steps"],
            "mean_window_error": selected["tail_mean_error"]}
    report["baselines"]["earlier_searched"]["depth"] = depth_summary(
        ROOT/"Model3/Experiment5/runs/depth_153aa2163bcf/depth.json", hashes)
    lines = ["# Experiment7: verified comparison", "",
        "N=10, J=gamma=1, half_mixed, total tolerance 0.02, mean allowance 0.016, sampling allowance 0.004.",
        "The new planners did not read baseline schedules. This analysis reads them only after schedule selection.", "",
        "| Method | Preparation steps | Reduction vs 273 | Reduction vs 117 | Forward mean maximum |",
        "|---|---:|---:|---:|---:|"]
    for method, result in report["methods"].items():
        result["step_reduction_vs_273_percent"] = 100*(1-result["steps"]/273)
        result["step_reduction_vs_117_percent"] = 100*(1-result["steps"]/117)
        lines.append(f"| {method} | {result['steps']} | {result['step_reduction_vs_273_percent']:.2f}% | {result['step_reduction_vs_117_percent']:.2f}% | {result['mean_window_error']:.9g} |")
    lines += ["", "| Method | Replay | Sampled error | Sampling distance | Two SE |", "|---|---|---:|---:|---:|"]
    for method, result in report["methods"].items():
        for key, row in result["replays"].items():
            lines.append(f"| {method} | {key} | {row['output_sampled_error']:.9g} | {row['max_sampling_distance']:.9g} | {row['max_two_standard_errors']:.9g} |")
    if any("depth" in result for result in report["methods"].values()):
        lines += ["", "Same FakeKingston chain, two boundary baths, optimization level 3, transpiler seed 11; four saved trajectories, three final readout settings.",
            "", "| Method | Full native depth | CZ gates | Two-qubit depth | Unitary gates incl. RZ |",
            "|---|---|---|---|---|"]
        for name, row in [("earlier 117-step schedule", report["baselines"]["earlier_searched"]), *report["methods"].items()]:
            if "depth" not in row:
                continue
            d = row["depth"]
            cells = ["–".join(map(str, d[field])) for field in ("full_depth_range", "cz_range", "two_qubit_depth_range", "unitary_gate_range_including_rz")]
            lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "Preparation and every future mean node were independently recomputed by direct stepping. Final sampled covariance means and standard errors were recomputed from saved per-trajectory covariances. All replay files passed circuit-source validation. Source hashes are recorded in VerifiedResults.json.",
        "", "Only preparation steps enter circuits. In particular, a small final timestep can cause thousands of classical validation steps without adding preparation gates. The comparison keeps the earlier finite-window mean/output-sampling protocol; it does not establish permanent settling or noisy hardware accuracy. Three seeds and descriptive two-SE checks do not constitute simultaneous confidence bounds. Search is heuristic; optimizer convergence and global optimality are not established."]
    (folder/"VerifiedComparison.md").write_text("\n".join(lines)+"\n")
    physics.atomic_json(folder/"VerifiedResults.json", report)
    print(json.dumps({method: {key: value for key, value in result.items() if key not in ("replays", "depth")}
                      for method, result in report["methods"].items()}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)
