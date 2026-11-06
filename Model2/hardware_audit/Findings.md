# Model2 hardware findings — 6 November 2026

The available shared files contain a completed raw-results export and a
mitigation compilation plan. The latest count-bearing manifests were not
copied for either of those run directories. A read-only IBM account lookup in
this environment failed with `InvalidAccountError`, so current remote job
states could not be checked here.

| Directory suffix | Available evidence | Interpretation |
| --- | --- | --- |
| `cosine44_20261102T184547_409820Z` | Planned manifest, 0/7 saved submissions | Older unsubmitted pilot plan |
| `cosine44_20261102T194423_617988Z` | Planned manifest, 0/97 saved submissions | Older unsubmitted full-size plan |
| `cosine44_20261102T222704_381015Z` | Analysis, CSV and figure listing 97 jobs | Raw completion reported by the exported analysis |
| `mitigation44_20261103T145657_688162Z` | Compiled plan for 20 jobs; no latest manifest here | Completion unknown from shared files |

## Raw results

The raw export covers 16,384 paths, 32 shots per basis, all three bases and
97 jobs on Phoenix's path `[68,69,59,58,57,56,66,76,86,96,97,98]`.
The values below were independently recalculated from its exported CSV, not
reconstructed from missing individual outcome counts.

| Quantity | Value |
| --- | ---: |
| Mean observed population | 0.31281414 |
| Observed population range | 0.30650330–0.31839371 |
| Stationary target population | 0.25 |
| Maximum population discrepancy from same-path ideal | 0.05176509 |
| Maximum population discrepancy from stationary target | 0.06839371 |
| Maximum absolute observed current/J | 0.00401688 |
| Maximum current/J discrepancy from same-path ideal | 0.00221324 |

The populations are systematically above both the target and same-path ideal.
The raw run does not meet the 0.02 target across all observables. Currents are
much closer to zero. These are raw point estimates and descriptive uncertainty,
not a hardware accuracy certificate.

Preparation/readout controls change across the recorded job sequence. The
median per-job maximum all-one discrepancy is 0.01709 in jobs 1–15, 0.10596
in jobs 16–32, and 0.01514 in jobs 33–97. Job 31
(`dbkv43onceqs73ammpj0`) has the largest all-one discrepancy, 0.390137, and
a held-out half-mixture response residual of 0.189941. This shows a substantial
change in the diagnostics, but the exported maxima do not identify its physical
cause or provide the per-site responses needed for a correction. No jobs were
excluded or corrected in this summary.

See the [recalculated export analysis](cosine44_20261102T222704_381015Z/Analysis.md)
and its [control diagnostic figure](cosine44_20261102T222704_381015Z/controls.png).

## Check the interrupted mitigation run on the original Windows machine

From the original repository checkout, with the current manifests, source and
native QPY still present:

```powershell
python Model2/audit_hardware_runs.py --collect-existing --analyze
```

This checks known uncollected IDs, retrieves only finished results, saves a
manifest backup, and analyzes complete collected jobs. It submits nothing.
Read `Model2/hardware_audit/Audit.md` for exact counts, job states, analysis and
resume/inspection commands. Incomplete mitigation cohorts retain their
original path IDs, original same-path ideal covariances and job-local controls.

A normal continuation of the known mitigation directory is:

```powershell
python Model2/Experiment6/run_hardware.py --resume "Model2/Experiment6/hardware_runs/mitigation44_20261103T145657_688162Z/manifest.json" --execute
```

Run the audit first. A known failed job or an ambiguous submission needs the
specific handling printed there; do not reset job IDs or start a new run to
recover progress. The two older raw plans are not the completed raw run and
should not be started merely because their saved submission counts are zero.
