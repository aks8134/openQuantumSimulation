"""Inventory Model2 hardware runs, recover existing DONE results, and analyze.

Default is offline. --check-ibm checks saved, uncollected job IDs.
--collect-existing also saves their DONE results, with a manifest backup.
No Sampler, submission, cancellation, retry or compilation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from library.reservoir_scaling import atomic_json
from Model2.Experiment5 import input_data, qpy_storage, run_hardware as raw
from Model2.Experiment5 import analyze as raw_analysis
from Model2.Experiment6 import run_hardware as mitigation, analyze_hardware as mitigation_analysis
from Model3.Experiment5.analyze_hardware import serializable


def command_path(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def outcome_counts(observation, shots):
    counts = observation["outcome_counts"]
    if (not counts or any(type(value) is not int or value < 0 for value in counts.values())
            or any(not re.fullmatch("[01]{10}", label) for label in counts)
            or sum(counts.values()) != shots):
        raise ValueError("invalid ten-bit outcome counts or shot total")


def load_manifest(path):
    manifest = json.loads(path.read_text())
    kind = manifest.get("experiment")
    runner = raw if kind == raw.EXPERIMENT else mitigation if kind == mitigation.EXPERIMENT else None
    if runner is None or manifest.get("plan_sha256") != runner.plan_hash(manifest):
        raise ValueError("unknown experiment or manifest plan hash mismatch")
    if manifest["configuration"]["spec_sha256"] != input_data.SPEC_SHA256:
        raise ValueError("unexpected Model2 frozen specification")
    previous = 0
    for index, batch in enumerate(manifest["batches"]):
        if batch["start"] != previous or not previous < batch["stop"] <= len(manifest["circuits"]):
            raise ValueError("unordered or incomplete batch coverage")
        if batch["status"] == "collected" and (not batch.get("job_id") or batch.get("observations") is None):
            raise ValueError("collected batch has no job ID or results")
        if batch.get("observations") is not None:
            rows = manifest["circuits"][batch["start"]:batch["stop"]]
            if batch["status"] != "collected" or len(rows) != len(batch["observations"]):
                raise ValueError("incomplete saved batch results")
            for row, observation in zip(rows, batch["observations"]):
                if row["batch"] != index:
                    raise ValueError("circuit batch identity mismatch")
                outcome_counts(observation, row["shots"])
        previous = batch["stop"]
    if previous != len(manifest["circuits"]):
        raise ValueError("missing batch coverage")
    return runner, manifest


def load_source(folder, manifest):
    config = manifest["configuration"]
    if qpy_storage.file_sha256(folder/"source.json") != config["source_sha256"]:
        raise ValueError("saved source hash mismatch")
    spec = input_data.load_spec(folder/"spec.json")
    source = json.loads((folder/"source.json").read_text())
    size = config.get("trajectories", config.get("trajectory_count"))
    if (source["configuration"]["physics"] != spec["physics"] or config["physics"] != spec["physics"]
            or source["configuration"]["seed"] != config["seed"] or source["configuration"]["trajectories"] != size
            or source["preparation"] != spec["preparation"] or source["schedule"]["step_dts"] != spec["schedule"]["step_dts"]
            or source["configuration"]["order"] != "jump-first"):
        raise ValueError("saved source differs from the frozen physics, preparation, schedule or seed")
    return source


def raw_subset(manifest, source, output):
    """Construct an analysis view; preserve original IDs and ideal covariances."""
    batches = [b for b in manifest["batches"] if b["status"] == "collected"]
    ids = sorted({r["trajectory"] for b in batches for r in manifest["circuits"][b["start"]:b["stop"]]
                  if r["kind"] == "evolution"})
    if len(ids) < 2:
        raise ValueError("need at least two complete raw trajectories")
    mapping = {path: i for i, path in enumerate(ids)}
    view = dict(manifest, configuration={**manifest["configuration"], "trajectories": len(ids)}, circuits=[], batches=[])
    for index, batch in enumerate(batches):
        begin = len(view["circuits"])
        for row in manifest["circuits"][batch["start"]:batch["stop"]]:
            row = dict(row, batch=index, pub_index=len(view["circuits"]))
            if row["kind"] == "evolution":
                row["trajectory"] = mapping[row["trajectory"]]
            view["circuits"].append(row)
        view["batches"].append(dict(batch, start=begin, stop=len(view["circuits"])))
    measured, conditional, across, controls = raw_analysis.reconstruct(view, source)
    physics = source["configuration"]["physics"]
    covariances = np.asarray(source["final_covariances_real_gauge"])[ids]
    ideals = dict(populations=np.diagonal(covariances, axis1=-2, axis2=-1).mean(axis=0),
        excitation_flows=(4*physics["exchange_prefactor"]*physics["J"]*
                         np.diagonal(covariances, offset=1, axis1=-2, axis2=-1)).mean(axis=0))
    observations = {key: value.mean(axis=0) for key, value in measured.items()}
    report = dict(experiment=raw.EXPERIMENT, complete=len(batches) == len(manifest["batches"]),
        jobs=len(batches), planned_jobs=len(manifest["batches"]), trajectories=len(ids), trajectory_indices=ids,
        job_ids=[b["job_id"] for b in batches], observables=observations,
        ideal_same_paths={key: ideals[key] for key in observations},
        stationary={key: source["stationary_observables"][key] for key in observations},
        across_path_standard_errors=across, conditional_shot_standard_errors=conditional,
        per_job_preparation_controls=controls,
        verification="Manifest and exact source bytes verified. Analysis reconstructs outcome counts; QPY bytes are not audited by this command.")
    atomic_json(output/"analysis.json", serializable(report))
    rows = [dict(observable=key, index_1based=i+1, hardware=float(value),
            ideal_same_paths=float(ideals[key][i]), stationary=float(report["stationary"][key][i]),
            across_path_SE=float(across[key][i])) for key, values in observations.items() for i, value in enumerate(values)]
    with (output/"observables.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Model2 raw collected-job analysis", "", f"Complete: **{report['complete']}**. "
        f"{len(batches)}/{len(manifest['batches'])} jobs, {len(ids)} original trajectories.", "",
        "| Observable | Maximum discrepancy from same-path ideal | Maximum discrepancy from stationary target |",
        "| --- | ---: | ---: |"]
    for key, values in observations.items():
        lines.append(f"| {key} | {np.max(abs(values-ideals[key])):.9g} | {np.max(abs(values-report['stationary'][key])):.9g} |")
    lines += ["", "An incomplete cohort describes only the completed original paths and complete job-local controls. "
              "No paths are regenerated and no correction, clipping or final full-run accuracy claim is applied.", "", report["verification"]]
    (output/"Analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(complete=report["complete"], trajectories=len(ids), jobs=len(batches), report=str(output/"Analysis.md"))


def analyze_export(folder, output):
    """Analyze portable tables when original count-bearing files were not copied."""
    with (folder/"observables.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {}
    for observable in sorted({r["observable"] for r in rows}):
        selected = [r for r in rows if r["observable"] == observable]
        values = np.array([float(r["hardware"]) for r in selected])
        metrics[observable] = dict(minimum=float(values.min()), maximum=float(values.max()), mean=float(values.mean()),
            same_path_maximum=float(max(abs(float(r["hardware"])-float(r["ideal_same_paths"])) for r in selected)),
            stationary_maximum=float(max(abs(float(r["hardware"])-float(r["stationary"])) for r in selected)))
    text = (folder/"Analysis.md").read_text()
    controls = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) == 5 and re.fullmatch("[a-z0-9]{20,32}", parts[0]):
            controls.append(dict(job_id=parts[0], **dict(zip(
                ("zero_maximum", "one_maximum", "mixture_maximum", "heldout_residual"), map(float, parts[1:])))))
    output.mkdir(parents=True, exist_ok=True)
    report = dict(evidence="exported_analysis_and_csv_only", metrics=metrics, reported_jobs=len(controls),
        controls=controls, source_directory=str(folder.resolve()),
        limitation="Original manifest/counts/source are absent here; completion is reported by the exported analysis, not independently reconstructed from raw results.")
    atomic_json(output/"export_analysis.json", report)
    lines = ["# Model2 exported raw analysis", "", report["limitation"], "",
        f"The exported analysis lists **{len(controls)} jobs**.", "",
        "| Observable | Mean | Minimum | Maximum | Max vs same-path ideal | Max vs stationary |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, m in metrics.items():
        lines.append(f"| {key} | {m['mean']:.8f} | {m['minimum']:.8f} | {m['maximum']:.8f} | {m['same_path_maximum']:.8f} | {m['stationary_maximum']:.8f} |")
    if controls:
        worst = max(enumerate(controls, 1), key=lambda item: item[1]["one_maximum"])
        lines += ["", f"Largest all-one control discrepancy: **{worst[1]['one_maximum']:.6f}**, "
            f"job {worst[0]} (`{worst[1]['job_id']}`). Its held-out mixture residual is "
            f"**{worst[1]['heldout_residual']:.6f}**. These are per-job maxima over sites, "
            "not site-resolved calibration matrices; they cannot be used to apply an affine correction.", "",
            "![Per-job preparation/readout diagnostics](controls.png)"]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        for key in ("zero_maximum", "one_maximum", "mixture_maximum", "heldout_residual"):
            ax.plot(range(1, len(controls)+1), [r[key] for r in controls], label=key.replace("_", " "))
        ax.set(xlabel="Job index in exported report", ylabel="Maximum over ten sites")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.savefig(output/"controls.png", dpi=160); plt.close(fig)
    (output/"Analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def audit_run(folder, options, output, service=None):
    result = dict(run=folder.name, directory=command_path(folder), jobs=[])
    path = folder/"manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result.update(state="missing_manifest", limitation="Latest job IDs and counts cannot be determined from Plan.md.")
        if (folder/"Plan.md").exists():
            plan = (folder/"Plan.md").read_text(encoding="utf-8")
            match = re.search(r"([\d,]+) jobs", plan)
            if match:
                result["planned_jobs"] = int(match[1].replace(",", ""))
        if (folder/"Analysis.md").exists() and (folder/"observables.csv").exists():
            result["state"] = "completed_analysis_export"
            result["limitation"] = "Completion is reported by exported analysis; the original manifest, source and individual counts are absent here."
            if options.analyze:
                export = analyze_export(folder, output)
                result.update(reported_completed_jobs=export["reported_jobs"], export_metrics=export["metrics"],
                              analysis=str(output/"Analysis.md"))
        return result
    try:
        runner, manifest = load_manifest(path)
    except (ValueError, KeyError, OSError) as error:
        result.update(state="invalid_manifest", error=f"{type(error).__name__}: {error}")
        return result
    config = manifest["configuration"]
    result.update(experiment=manifest["experiment"], saved_status=manifest["status"], backend=config["backend"],
        planned_jobs=len(manifest["batches"]), planned_shots=config["total_shots"], plan_sha256=manifest["plan_sha256"])
    source = None
    backup = None

    def persist():
        nonlocal backup
        if backup is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            backup = folder/("manifest.before_audit_"+stamp+".json")
            shutil.copy2(path, backup)
        atomic_json(path, manifest)

    for index, batch in enumerate(manifest["batches"]):
        entry = dict(batch=index, job_id=batch.get("job_id"), saved_status=batch["status"],
                     circuits=batch["stop"]-batch["start"], collected=batch.get("observations") is not None)
        if batch.get("job_id") and not entry["collected"] and service is not None:
            try:
                job = service.job(batch["job_id"])
                status = job.status()
                entry["provider_status"] = str(getattr(status, "name", status)).upper()
                if entry["provider_status"] in ("ERROR", "CANCELLED"):
                    entry["error_message"] = str(job.error_message())
                if entry["provider_status"] == "DONE" and options.collect_existing:
                    source = source or load_source(folder, manifest)
                    response = job.result()
                    if len(response) != entry["circuits"]:
                        raise ValueError("provider result length differs from the saved job")
                    observations = []
                    for pub, row in zip(response, manifest["circuits"][batch["start"]:batch["stop"]]):
                        observation = dict(outcome_counts={str(k): int(v) for k, v in pub.data.outcome.get_counts().items()})
                        outcome_counts(observation, row["shots"])
                        observations.append(observation)
                    batch.update(observations=observations, status="collected", collected_at_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        batch["job_metrics"] = json.loads(json.dumps(job.metrics(), default=str))
                    except Exception:
                        pass
                    persist()
                    entry.update(collected=True, recovered=True)
            except Exception as error:
                entry["lookup_or_collection_error"] = f"{type(error).__name__}: {error}"
        if not batch.get("job_id") and batch["status"] == "submitting":
            entry["ambiguous_submission"] = True
        result["jobs"].append(entry)
    complete = all(b.get("observations") is not None for b in manifest["batches"])
    if complete and options.collect_existing and manifest["status"] != "collected":
        manifest["status"] = "collected"; persist()
    collected = sum(j["collected"] for j in result["jobs"])
    result.update(collected_jobs=collected, complete=complete,
                  saved_states=dict(Counter(b["status"] for b in manifest["batches"])))
    if backup:
        result["manifest_backup"] = command_path(backup)
    ambiguous = any(j.get("ambiguous_submission") for j in result["jobs"])
    failed = [j for j in result["jobs"] if j.get("provider_status") in ("ERROR", "CANCELLED")]
    script = "Model2/Experiment5/run_hardware.py" if runner is raw else "Model2/Experiment6/run_hardware.py"
    base = f'python {script} --resume "{command_path(path)}"'
    if ambiguous:
        result.update(state="ambiguous_submission", limitation="Reconcile the unknown job ID with IBM before any resume.")
    elif failed:
        result["state"] = "failed_saved_job"
        result["inspect_commands"] = [base+f" --inspect-job {j['job_id']}" for j in failed]
        result["retry_commands"] = [base+f" --execute --retry-failed-job {j['job_id']}" for j in failed
                                    if j["provider_status"] == "ERROR" and re.search(r"\b6056\b", j.get("error_message", ""))]
    elif complete:
        result["state"] = "complete"
    elif not any(j.get("job_id") for j in result["jobs"]) and manifest["status"] == "planned":
        result.update(state="unsubmitted_plan", limitation="No recorded submissions; do not restart an older plan to recover a different run.")
    else:
        result.update(state="incomplete", resume_command=base+" --execute")
    if options.analyze and collected:
        try:
            source = source or load_source(folder, manifest)
            if runner is raw:
                result["analysis_summary"] = raw_subset(manifest, source, output)
                result["analysis"] = str(output/"Analysis.md")
            else:
                report = mitigation_analysis.analyze_counts(manifest, source, allow_partial=True)
                report.update(run_directory=str(folder.resolve()), verification=dict(
                    source_integrity="exact_bytes", manifest_integrity="plan_hash",
                    raw_counts_reconstructed=True, compiled_archives="not read by this count analysis"))
                runner.protocol.save_analysis(report, output)
                title = "# Model2 mitigation: completed-job subset" if not report["analysis"]["complete"] else "# Model2 mitigation: complete run"
                with (output/"Analysis.md").open("r", encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
                lines[:1] = [title, "", f"**{report['analysis']['collected_jobs']}/{report['analysis']['planned_jobs']} jobs; "
                    f"{report['analysis']['trajectories']}/{report['analysis']['planned_trajectories']} original trajectories.**"]
                (output/"Analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
                result.update(analysis=str(output/"Analysis.md"), analysis_summary=report["analysis"])
        except (ValueError, KeyError, OSError) as error:
            result["analysis_error"] = f"{type(error).__name__}: {error}"
    return result


def main(arguments=None, *, service=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, action="append", help="audit only these manifest paths; repeatable")
    parser.add_argument("--check-ibm", action="store_true")
    parser.add_argument("--collect-existing", action="store_true", help="retrieve only known DONE jobs; backup and update saved counts")
    parser.add_argument("--analyze", action="store_true", help="analyze complete collected jobs, including labelled subsets")
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--output-directory", type=Path)
    options = parser.parse_args(arguments)
    output = options.output_directory or options.root/"Model2"/"hardware_audit"
    folders = ([path.resolve().parent for path in options.manifest] if options.manifest else
        sorted(path for number in (5, 6) for path in (options.root/f"Model2/Experiment{number}/hardware_runs").glob("*") if path.is_dir()))
    if any(output.resolve().is_relative_to(folder.resolve()) for folder in folders):
        raise ValueError("audit outputs must be outside the raw hardware run directories")
    account_error = None
    if (options.check_ibm or options.collect_existing) and service is None:
        try:
            service = raw.runtime.make_service(options)
        except Exception as error:
            account_error = type(error).__name__+": IBM account initialization failed; local evidence only."
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), account_error=account_error, runs=[])
    for folder in folders:
        print(f"Auditing {folder.name}", flush=True)
        entry = audit_run(folder, options, output/folder.name, service)
        report["runs"].append(entry)
        print(f"  {entry['state']}; collected {entry.get('collected_jobs', entry.get('reported_completed_jobs', '?'))}/{entry.get('planned_jobs', '?')}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"audit.json", report)
    lines = ["# Model2 hardware completion audit", "", f"Recorded at {report['created_at_utc']}.", "",
        "| Run | Evidence/state | Collected jobs | Planned jobs |", "| --- | --- | ---: | ---: |"]
    for entry in report["runs"]:
        lines.append(f"| {entry['run']} | {entry['state']} | {entry.get('collected_jobs', entry.get('reported_completed_jobs', '?'))} | {entry.get('planned_jobs', '?')} |")
    if account_error:
        lines += ["", account_error]
    for entry in report["runs"]:
        lines += ["", f"## {entry['run']}", ""]
        for key in ("limitation", "error", "analysis_error"):
            if entry.get(key): lines += [entry[key], ""]
        if entry.get("analysis"):
            lines += [f"[Collected-result analysis]({Path(entry['analysis']).resolve().relative_to(output.resolve()).as_posix()})", ""]
        if entry.get("jobs") and entry["state"] != "unsubmitted_plan":
            lines += ["| Job index | Saved ID | Results collected | Provider state if queried |", "| ---: | --- | --- | --- |"]
            for job in entry["jobs"]:
                lines.append(f"| {job['batch']+1} | {job['job_id'] or '(none)'} | {job['collected']} | {job.get('provider_status', 'not queried')} |")
            lines += [""]
            for job in entry["jobs"]:
                if job.get("lookup_or_collection_error"):
                    lines += [f"Job {job['batch']+1}: {job['lookup_or_collection_error']}", ""]
        commands = ([entry["resume_command"]] if entry.get("resume_command") else [])
        commands += entry.get("inspect_commands", [])+entry.get("retry_commands", [])
        for command in commands:
            lines += ["```powershell", command, "```", ""]
    lines += ["No jobs were submitted, cancelled or retried by this audit. Existing known IDs are preserved. "
        "A resume retrieves saved jobs and skips collected batches; terminal errors require explicit handling. "
        "Plan.md is a compilation snapshot and does not establish current provider state. "
        "Analysis of an incomplete run describes only the original paths in completed jobs.", ""]
    (output/"Audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit: {output/'Audit.md'}", flush=True)
    return report


if __name__ == "__main__":
    main()
