"""All six DD-on profiles with streamed counts and historical DD-off context."""
from pathlib import Path
import json

import numpy as np

from library.reservoir_scaling import atomic_json
from Model2.Experiment6 import analyze_hardware as statistics
from Model2.Experiment6.progress import progress
from Model2.Experiment8 import run_hardware as run
from Model3.Experiment5 import run_mitigation as protocol
from Model3.Experiment5.analyze_hardware import serializable


def analyze(manifest_path, *, output=None):
    folder, m, source = run.read_saved(manifest_path)
    if m.get("submission_mode") != "hardware":
        raise ValueError("physical mitigation analysis requires hardware counts; dry-run results are randomized mocks")
    if m["status"] != "collected":
        raise ValueError("collect every job before full mitigation analysis")
    positions = {id(b):i for i,b in enumerate(m["batches"])}

    def observations(batch):
        saved = run.read_counts(folder, m, positions[id(batch)])
        if saved is None:
            raise ValueError("collected result file missing")
        return saved["observations"]

    with progress("Computing six DD-on profiles from one saved job at a time"):
        report = statistics.analyze_counts(m, source, validator=run.verify_manifest, observation_reader=observations)
    report.update(run_directory=str(folder), dd_condition="on", verification=dict(
        plan_sha256=m["plan_sha256"], source_integrity="exact_bytes", raw_counts_reconstructed=True,
        native_circuits_loaded=False, calibration="Independent job-local SPAM before trajectory-weighted averaging",
        local_dd_checks_preserve_original_operations=all(r["dd_check"]["original_operations_and_local_times_preserved"] for r in m["circuits"]),
        server_timing_scope="DD-on only. No paired compiler-returned DD-off schedule or causal DD-effect estimate."))
    reference = json.loads((folder/"historical_reference.json").read_text())
    old, c = reference["configuration"], m["configuration"]
    keys = ("backend", "trajectory_count", "seed", "seed_transpiler", "optimization_level", "shots", "control_shots",
            "randomizations", "fold_factors", "physics", "steps", "physical_path", "batch_size", "total_jobs", "bootstrap")
    differences = {key:dict(current=c[key],historical=old[key]) for key in keys if c[key] != old[key]}
    context = dict(scope=reference["scope"], matching_recorded_settings=not differences, differing_settings=differences,
        execution_dates_and_calibrations_are_unpaired=True, comparisons=[])
    for name,p in report["statistics"]["profiles"].items():
        q = reference["profiles"][name]
        context["comparisons"].append(dict(method=name, current_ideal_mae=p["ideal_mae"], historical_ideal_mae=q["ideal_mae"],
            current_ideal_maximum=p["ideal_maximum"], historical_ideal_maximum=q["ideal_maximum"],
            current_physical_maximum=p["physical_maximum"], historical_physical_maximum=q["physical_maximum"],
            physical_maximum_change=p["physical_maximum"]-q["physical_maximum"]))
    report["historical_dd_off_reference"] = context
    report["runtime_execution"] = dict(mode=c.get("execution_mode", "job"),
        current_batch=m.get("runtime_batch"), previous_batches=m.get("runtime_batch_history", []),
        job_runtime_batch_ids=[b.get("runtime_batch_id") for b in m["batches"]])
    target = np.asarray(report["physical_target"])
    report["population_target_assessment"] = {
        name:dict(maximum_point_error=float(np.max(abs(p["populations"]-target))),
                  point_estimate_within_0p02=bool(p["physical_maximum"] <= .02))
        for name,p in report["statistics"]["profiles"].items()}
    report["scope"] += " All measured arms use ratio-10 XX DD; no new DD-off hardware arm. Historical differences are descriptive only."
    output = Path(output) if output is not None else run.DIRECTORY/"analysis"/folder.name
    output = output.resolve()
    if output.is_relative_to(folder.resolve()):
        raise ValueError("analysis output must be outside the raw run directory")
    with progress("Writing DD-on analysis, plots, and historical comparison"):
        protocol.save_analysis(report, output)
        atomic_json(output/"analysis.json", serializable(report))
        run.timing.write_csv(output/"historical_comparison.csv", context["comparisons"])
        text = (output/"Analysis.md").read_text()
        text = text.replace("# First-order mitigation pilot", "# Model2 Experiment8: all profiles use DD")
        text += "\nAll newly measured profiles use XX DD with minimum idle ratio 10. No DD-off circuits were submitted.\n"
        text += "\nExperiment6 is a historical DD-off reference. Execution dates/calibration are unpaired; the differences do not isolate a DD effect.\n"
        text += f"\nRecorded physics/path/sampling settings match the historical run: **{not differences}**. See historical_comparison.csv and analysis.json.\n"
        text += "\nAll six intermediate profiles, half-mixture checks and uncertainty are retained. Pointwise/bootstrap uncertainty omits drift and calibration-model bias; no job exclusion, clipping, or target fitting. Currents were not measured.\n"
        (output/"Analysis.md").write_text(text, encoding="utf-8")
    atomic_json(folder/"analysis_location.json", dict(directory=str(output), plan_sha256=m["plan_sha256"]))
    print(f"Analysis: {output/'Analysis.md'}", flush=True)
    return report
