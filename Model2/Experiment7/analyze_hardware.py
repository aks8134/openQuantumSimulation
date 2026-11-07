"""Paired population/current analysis; refuses randomized cloud dry-run counts."""
from __future__ import annotations

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
            if row["arm"] == "dd":
                expected[row["trajectory"], row["basis"]] = row["dd_check"]["added_x_by_qubit"]
    for (path, basis), arms in paired.items():
        server.append(dict(trajectory=path, basis=basis,
            **timing.server_comparison(arms["baseline"], arms["dd"],
                expected_added_x_by_qubit=expected[path, basis])))
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
        baseline, dd = (measured[arm][family] for arm in ("baseline", "dd"))
        delta = dd-baseline
        conditional_se = np.sqrt(np.maximum(0, np.diag(sum(measured[a][covariance_name] for a in c["arms"]))))
        across_se = delta.std(axis=0, ddof=1)/np.sqrt(len(delta))
        scale = J if family == "excitation_flows" else 1.
        arms = {}
        for arm in c["arms"]:
            mean = measured[arm][family].mean(axis=0)
            arms[arm] = dict(mean=mean, maximum_error_same_paths=float(np.abs(mean-reference).max()/scale),
                rmse_same_paths=float(np.sqrt(np.mean((mean-reference)**2))/scale),
                shot_standard_errors=np.sqrt(np.maximum(0, np.diag(measured[arm][covariance_name]))))
        report[family] = dict(ideal_same_paths=reference, arms=arms, dd_minus_baseline=delta.mean(axis=0),
            difference_shot_standard_errors_fixed_paths=conditional_se,
            difference_across_path_standard_errors_descriptive=across_se)
        for i, value in enumerate(reference):
            csv_rows.append(dict(observable=family, index_1based=i+1, ideal_same_paths=float(value),
                baseline=float(arms["baseline"]["mean"][i]), dd=float(arms["dd"]["mean"][i]),
                dd_minus_baseline=float(delta[:, i].mean()), difference_shot_se=float(conditional_se[i]),
                difference_across_path_se_descriptive=float(across_se[i])))
    result = dict(experiment=run.EXPERIMENT, configuration=c, plan_sha256=m["plan_sha256"],
        job_ids=[b["job_id"] for b in m["batches"]], observables=report, controls=controls,
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
    lines = ["# Model2 Experiment7 hardware DD comparison", "",
        f"{c['trajectory_count']} paired paths, {c['shots']} shots per basis/arm; {len(m['batches'])} jobs.",
        "Errors below compare each measured ensemble to the ideal ensemble of those exact saved paths.", "",
        "| Observable | DD-off max error | XX max error |", "| --- | ---: | ---: |"]
    for family, values in report.items():
        lines.append(f"| {family} | {values['arms']['baseline']['maximum_error_same_paths']:.6g} | {values['arms']['dd']['maximum_error_same_paths']:.6g} |")
    lines += ["", f"All compared non-X server timing anchors equal: **{result['all_non_x_server_anchors_equal']}**.",
        f"Compiler-returned added X pulse counts match construction: **{result['all_added_x_counts_match']}**.",
        "A timing mismatch needs inspection before attributing a change to DD alone.",
        "Job-local zero/one/half-mixture diagnostics and uncertainty vectors are in analysis.json; no correction or job exclusion applied.",
        "Currents in the table are divided by J. Small pilots do not establish the full-run 0.02 tolerance.", ""]
    (out/"Analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Hardware analysis: {(out/'Analysis.md').resolve()}")
    return result
