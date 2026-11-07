"""Paired population/current analysis; refuses randomized cloud dry-run counts."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from library.reservoir_scaling import atomic_json
from Model2.Experiment7 import run_hardware as run, schedule_tools as timing
from Model3.Experiment5.analyze_hardware import moments, reconstruct, serializable


def analyze(manifest_path):
    folder, m, source = run.read_saved(manifest_path)
    c = m["configuration"]
    if m.get("submission_mode") != "hardware":
        raise ValueError("physical analysis requires real hardware counts; dry-run outputs are randomized mock data")
    if c["stage"] != "paired" or m["status"] != "collected":
        raise ValueError("collect the complete matched baseline/DD experiment before analysis")
    records = {arm: [] for arm in c["arms"]}
    observations = {arm: [] for arm in c["arms"]}
    controls, server, paired, expected = [], [], {}, {}
    for row, pub in run.saved_pubs(folder, m):
        counts = pub["outcome_counts"]
        if sum(counts.values()) != row["shots"]:
            raise ValueError("saved shot count mismatch")
        if row["kind"] == "control":
            values, covariance, shots = moments(counts, 10)
            controls.append(dict(batch=row["batch"], arm=row["arm"], control=row["control"],
                populations=values, shot_standard_errors=np.sqrt(np.maximum(0, np.diag(covariance))),
                maximum_error=float(np.abs(values-run.CONTROLS[row["control"]]).max()), shots=shots))
        else:
            records[row["arm"]].append(row)
            observations[row["arm"]].append(dict(outcome_counts=counts))
            paired.setdefault((row["trajectory"], row["basis"]), {})[row["arm"]] = timing.parse_timing(pub["timing"])
            if row["arm"] != "baseline":
                expected[row["trajectory"], row["basis"], row["arm"]] = row["dd_check"]["added_x_by_qubit"]
    server = run.compare_arm_timings(c, paired, expected, require_complete=True)
    physics = source["configuration"]["physics"]
    prefactor, J = physics["exchange_prefactor"], physics["J"]
    C = np.asarray(source["final_covariances_real_gauge"])
    ideal = dict(populations=np.diagonal(C, axis1=-2, axis2=-1),
                 excitation_flows=4*prefactor*J*np.diagonal(C, offset=1, axis1=-2, axis2=-1))
    measured = {arm: reconstruct(records[arm], observations[arm], 10, prefactor, J) for arm in c["arms"]}
    if any(values["indices"] != list(range(c["trajectory_count"])) for values in measured.values()):
        raise ValueError("matched saved path coverage/order changed")
    report, csv_rows = {}, []
    for family, covariance_name in (("populations", "population_shot_covariance"),
                                     ("excitation_flows", "current_shot_covariance")):
        reference = ideal[family].mean(axis=0)
        scale = J if family == "excitation_flows" else 1.
        arms = {}
        for arm in c["arms"]:
            mean = measured[arm][family].mean(axis=0)
            arms[arm] = dict(mean=mean, maximum_error_same_paths=float(np.abs(mean-reference).max()/scale),
                rmse_same_paths=float(np.sqrt(np.mean((mean-reference)**2))/scale),
                shot_standard_errors=np.sqrt(np.maximum(0, np.diag(measured[arm][covariance_name]))))
        contrasts = {}
        for reference_arm, treatment_arm in combinations(c["arms"], 2):
            delta = measured[treatment_arm][family]-measured[reference_arm][family]
            # Only these two arms contribute shot noise. In particular, the
            # direct DD-vs-DD contrast does not carry baseline shot variance.
            conditional_se = np.sqrt(np.maximum(0, np.diag(
                measured[reference_arm][covariance_name]+measured[treatment_arm][covariance_name])))
            contrasts[f"{treatment_arm}_minus_{reference_arm}"] = dict(
                reference=reference_arm, treatment=treatment_arm, difference=delta.mean(axis=0),
                shot_standard_errors_fixed_paths=conditional_se,
                across_path_standard_errors_descriptive=delta.std(axis=0, ddof=1)/np.sqrt(len(delta)))
        report[family] = dict(ideal_same_paths=reference, arms=arms, comparisons=contrasts)
        if c["arms"] == ["baseline", "dd"]:
            legacy = contrasts["dd_minus_baseline"]
            report[family].update(dd_minus_baseline=legacy["difference"],
                difference_shot_standard_errors_fixed_paths=legacy["shot_standard_errors_fixed_paths"],
                difference_across_path_standard_errors_descriptive=legacy["across_path_standard_errors_descriptive"])
        for i, value in enumerate(reference):
            row = dict(observable=family, index_1based=i+1, ideal_same_paths=float(value))
            row.update({a: float(arms[a]["mean"][i]) for a in c["arms"]})
            for label, contrast in contrasts.items():
                row.update({label: float(contrast["difference"][i]),
                    label+"_shot_se": float(contrast["shot_standard_errors_fixed_paths"][i]),
                    label+"_path_se_descriptive": float(contrast["across_path_standard_errors_descriptive"][i])})
            if c["arms"] == ["baseline", "dd"]:
                row.update(difference_shot_se=row["dd_minus_baseline_shot_se"],
                           difference_across_path_se_descriptive=row["dd_minus_baseline_path_se_descriptive"])
            csv_rows.append(row)
    errors = {}
    for arm in c["arms"]:
        residual = np.concatenate([report[f]["arms"][arm]["mean"]-report[f]["ideal_same_paths"]
            if f == "populations" else (report[f]["arms"][arm]["mean"]-report[f]["ideal_same_paths"])/J
            for f in ("populations", "excitation_flows")])
        errors[arm] = dict(maximum_error_same_paths=float(np.abs(residual).max()),
                          combined_rmse_same_paths=float(np.sqrt(np.mean(residual**2))))
    error_contrasts = [dict(reference=a, treatment=b,
        maximum_error_difference=errors[b]["maximum_error_same_paths"]-errors[a]["maximum_error_same_paths"],
        combined_rmse_difference=errors[b]["combined_rmse_same_paths"]-errors[a]["combined_rmse_same_paths"])
        for a, b in combinations(c["arms"], 2)]
    result = dict(experiment=run.EXPERIMENT, configuration=c, plan_sha256=m["plan_sha256"],
        job_ids=[b["job_id"] for b in m["batches"]], observables=report, controls=controls,
        method_errors=errors, error_comparisons=error_contrasts,
        server_timing_checks=server,
        all_non_x_server_anchors_equal=all(r["non_x_anchor_timing_equal"] for r in server),
        all_added_x_counts_match=all(r["added_x_counts_match"] for r in server),
        per_path={a: {k: measured[a][k] for k in ("indices", "populations", "excitation_flows")} for a in c["arms"]},
        scope="Raw paired counts, all bases, same-path ideal reference. No SPAM inversion, twirling or ZNE. "
              "Shot SE conditions on these paths; across-path SE is descriptive and includes shot fluctuations. "
              "Neither SE includes calibration drift/systematic hardware error or certifies a maximum-error tolerance.")
    out = folder/"analysis"; out.mkdir(exist_ok=True)
    atomic_json(out/"analysis.json", serializable(result))
    timing.write_csv(out/"observables.csv", csv_rows)
    timing.write_csv(out/"method_errors.csv", [dict(arm=a, **v) for a, v in errors.items()])
    timing.write_csv(out/"error_comparisons.csv", error_contrasts)
    lines = ["# Model2 Experiment7 hardware DD comparison", "",
        f"{c['trajectory_count']} paired paths, {c['shots']} shots per basis/arm; {len(m['batches'])} jobs.",
        "Errors below compare each measured ensemble to the ideal ensemble of those exact saved paths.", "",
        "| Observable | "+" | ".join(c["arms"])+" |",
        "| --- | "+" | ".join("---:" for _ in c["arms"])+" |"]
    for family, values in report.items():
        lines.append(f"| {family} | "+" | ".join(f"{values['arms'][a]['maximum_error_same_paths']:.6g}" for a in c["arms"])+" |")
    lines += ["", "Direct DD-policy contrasts and comparisons to baseline are in observables.csv, with shot and descriptive paired-path SE vectors.",
        "Combined maximum errors and RMSEs are in method_errors.csv. Error differences in error_comparisons.csv are treatment minus reference; negative favors treatment in this observed sample.",
        "These error rankings are descriptive: maximum-error differences do not have confidence intervals, and a small pilot does not establish a statistically decisive winner."]
    lines += ["", f"All compared non-X server timing anchors equal: **{result['all_non_x_server_anchors_equal']}**.",
        f"Compiler-returned added X pulse counts match construction: **{result['all_added_x_counts_match']}**.",
        "A timing mismatch needs inspection before attributing a change to DD alone.",
        "Job-local zero/one/half-mixture diagnostics and uncertainty vectors are in analysis.json; no correction or job exclusion applied.",
        "Currents in the table are divided by J. Small pilots do not establish the full-run 0.02 tolerance.", ""]
    (out/"Analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Hardware analysis: {(out/'Analysis.md').resolve()}")
    return result
