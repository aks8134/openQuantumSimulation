"""Offline analysis of final-register counts and same-job preparation controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5.run_hardware import read_saved
from Model3.Experiment5.analyze_hardware import moments
from Model3.Experiment5.run_hardware import count_observables


def features(counts, n, pairs=False):
    if not counts or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("counts must be nonempty nonnegative integers")
    singles, correlators, shots = count_observables(counts, n)
    mean = correlators if pairs else (1-singles)/2
    covariance = moments(counts, n, pairs=pairs)[1] if shots > 1 else None
    return mean, covariance


def norm_error(left, right, J):
    return float(max(np.max(np.abs(np.asarray(value)-right[key]))/(J if key == "excitation_flows" else 1.)
                     for key, value in left.items()))


def reconstruct(manifest, source):
    c = manifest["configuration"]
    p = source["configuration"]["physics"]
    n, paths = p["n_qubits"], c["trajectories"]
    grouped, controls = {}, []
    expected_start = 0
    for job, batch in enumerate(manifest["batches"]):
        if (batch["status"] != "collected" or batch["start"] != expected_start or not batch.get("job_id")
                or len(batch["observations"]) != batch["stop"]-batch["start"]):
            raise ValueError("incomplete, unordered or inconsistent job results")
        expected_start = batch["stop"]
        per_job = {}
        for index, observation in zip(range(batch["start"],batch["stop"]), batch["observations"]):
            row = manifest["circuits"][index]
            if row["pub_index"] != index or row["batch"] != job:
                raise ValueError("record job or publication index mismatch")
            if sum(observation["outcome_counts"].values()) != row["shots"]:
                raise ValueError("observed shot count differs from the saved plan")
            values, covariance = features(observation["outcome_counts"], n, pairs=row["basis"] != "z")
            if row["kind"] == "control":
                if row["basis"] != "z" or row["control"] in per_job:
                    raise ValueError("duplicate or non-Z preparation control")
                per_job[row["control"]] = {"populations": values.tolist(),
                    "standard_errors": np.sqrt(np.maximum(np.diag(covariance),0)).tolist(),
                    "expected": row["expected_populations"], "shots": row["shots"]}
            else:
                key = row["trajectory"]
                if row["basis"] in grouped.get(key, {}):
                    raise ValueError("duplicate trajectory/basis result")
                grouped.setdefault(key, {})[row["basis"]] = (values, covariance, job)
        if set(per_job) != {"all_zero","all_one","half_mixed"}:
            raise ValueError("every job must contain all three preparation controls")
        a, b, m = [np.asarray(per_job[key]["populations"]) for key in ("all_zero","all_one","half_mixed")]
        se_a, se_b, se_m = [np.asarray(per_job[key]["standard_errors"]) for key in ("all_zero","all_one","half_mixed")]
        residual = m-(a+b)/2
        controls.append({"batch": job, "job_id": batch["job_id"], "controls": per_job,
            "maximum_zero_error": float(np.max(np.abs(a))), "maximum_one_error": float(np.max(np.abs(b-1))),
            "maximum_mixture_error": float(np.max(np.abs(m-.5))),
            "held_out_mixture_residual": residual.tolist(),
            "held_out_residual_standard_errors": np.sqrt(se_m**2+(se_a**2+se_b**2)/4).tolist(),
            "maximum_held_out_residual": float(np.max(np.abs(residual)))})
    if expected_start != len(manifest["circuits"]) or set(grouped) != set(range(paths)):
        raise ValueError("missing or additional circuit/path results")
    populations, currents, pop_cov, cur_cov = [], [], [], []
    parity = np.where(np.arange(n-1)%2 == 0, 1., -1.)
    for index in range(paths):
        rows = grouped[index]
        if set(rows) != set(c["measurement_bases"]) or len({row[2] for row in rows.values()}) != 1:
            raise ValueError("missing bases or one path split between jobs")
        populations.append(rows["z"][0]); pop_cov.append(rows["z"][1])
        if "xy" in rows:
            factor = p["exchange_prefactor"]*p["J"]*parity
            currents.append(factor*(rows["yx"][0]-rows["xy"][0]))
            cur_cov.append(None if rows["xy"][1] is None else factor[:,None]*factor[None,:]*(rows["xy"][1]+rows["yx"][1]))
    measured = {"populations": np.asarray(populations)}
    covariance = {"populations": pop_cov}
    if currents:
        measured["excitation_flows"] = np.asarray(currents)
        covariance["excitation_flows"] = cur_cov
    conditional = {key: None if any(v is None for v in values) else
                   np.sqrt(np.maximum(np.diag(np.sum(values,axis=0)),0))/paths for key,values in covariance.items()}
    across = {key: value.std(axis=0,ddof=1)/np.sqrt(paths) for key,value in measured.items()}
    return measured, conditional, across, controls


def main(folder):
    folder, manifest, source = read_saved(Path(folder)/"manifest.json")
    if manifest["status"] != "collected":
        raise ValueError("analysis requires a fully collected experiment")
    measured, conditional, across, controls = reconstruct(manifest,source)
    c = manifest["configuration"]
    p = source["configuration"]["physics"]
    covariances = np.asarray(source["final_covariances_real_gauge"])
    ideal_paths = {"populations": np.diagonal(covariances,axis1=-2,axis2=-1),
                   "excitation_flows": 4*p["exchange_prefactor"]*p["J"]*np.diagonal(covariances,offset=1,axis1=-2,axis2=-1)}
    hardware = {key: values.mean(axis=0).tolist() for key,values in measured.items()}
    references = {"physical_stationary": {key:source["stationary_observables"][key] for key in hardware},
        "ideal_mean_channel": {key:source["mean_channel_observables"][key][0][-1] for key in hardware},
        "ideal_same_paths": {key:ideal_paths[key].mean(axis=0).tolist() for key in hardware}}
    errors = {key:norm_error(hardware,reference,p["J"]) for key,reference in references.items()}
    report = {"experiment": "Model2/Experiment5/hardware", "configuration": c,
        "job_ids": [batch["job_id"] for batch in manifest["batches"]],
        "observables": hardware, "references": references, "local_observable_errors": errors,
        "per_trajectory": {key:values.tolist() for key,values in measured.items()},
        "conditional_shot_standard_errors": {key:None if value is None else value.tolist() for key,value in conditional.items()},
        "across_path_standard_errors": {key:value.tolist() for key,value in across.items()},
        "per_job_preparation_controls": controls, "finite_path_assessment":source["finite_path_assessment"],
        "scope": "Raw output populations and currents when all bases measured. Preparation measurement bits ignored. No mitigation or clipping. Conditional shot SE excludes branch resampling; across-path SE includes path, preparation, shot and observed device variation. Do not add the two. SEs descriptive, no simultaneous or drift guarantee. Z-only data do not measure currents."}
    atomic_json(folder/"hardware_results.json",report)
    with (folder/"observables.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["observable","index_1based","hardware","ideal_same_paths","ideal_mean_channel","stationary","across_path_SE","conditional_shot_SE"])
        for key,values in hardware.items():
            for i,value in enumerate(values):
                writer.writerow([key,i+1,value,*[references[k][key][i] for k in ("ideal_same_paths","ideal_mean_channel","physical_stationary")],
                                 across[key][i],None if conditional[key] is None else conditional[key][i]])
    lines=["# Model2 Experiment5 hardware results","",
        f"44 preparation steps; {c['trajectories']} paths, {c['shots']} shots per basis, {c['total_jobs']} jobs.",
        f"Measured bases: {c['measurement_bases']}. Physical path: {c['physical_path']}.","",
        "| Reference | Maximum population/current-over-J discrepancy |","|---|---:|"]
    for key,value in errors.items():
        lines.append(f"| {key} | {value:.9g} |")
    lines += ["", "These are observed point estimates, not a hardware accuracy certificate. The default smaller path ensemble does not inherit the 16384-path validation. Compare the same-path ideal to separate device/measurement discrepancy from ideal path sampling and preparation bias.", "",
        "| Job | Zero max error | One max error | Half-mixture max error | Held-out mixture response residual |",
        "|---|---:|---:|---:|---:|"]
    for row in controls:
        lines.append(f"| {row['job_id']} | {row['maximum_zero_error']:.6g} | {row['maximum_one_error']:.6g} | {row['maximum_mixture_error']:.6g} | {row['maximum_held_out_residual']:.6g} |")
    lines += ["", "The held-out mixture is compared with (zero+one)/2 inside each job. These are preparation/readout diagnostics; no affine correction is applied to evolution. Same-job controls do not establish stable response throughout the job.","",
        "![Observed profiles](observables.png)","",report["scope"]]
    (folder/"Analysis.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(len(hardware),1,figsize=(10,4*len(hardware)),squeeze=False,constrained_layout=True)
    for ax,(key,values) in zip(axes.flat,hardware.items()):
        x=np.arange(1,len(values)+1)
        ax.errorbar(x,values,yerr=2*across[key],fmt="o",label="Hardware ±2 across-path SE")
        for ref,style in (("ideal_same_paths","--"),("ideal_mean_channel","-."),("physical_stationary",":")):
            ax.plot(x,references[ref][key],style,label=ref)
        ax.set(xlabel="Site / bond (one based)",ylabel=key)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(folder/"observables.png",dpi=160)
    plt.close(fig)
    print(f"Analysis: {folder/'Analysis.md'}; errors={errors}",flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    main(parser.parse_args().run_directory)
