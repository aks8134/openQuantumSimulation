# Audit and resume interrupted Model2 hardware runs

Run the audit from the **original checkout containing the latest manifests
and counts**, such as the Windows machine that submitted the jobs:

```powershell
python Model2/audit_hardware_runs.py --collect-existing --analyze
```

The audit scans Experiment5 and Experiment6 hardware run directories. It
validates the immutable plan hash and saved count/shot structure, checks
uncollected saved job IDs with IBM, and retrieves results only for jobs
reporting `DONE`. It does not construct a Sampler, submit, retry, cancel or
compile. Active jobs are recorded without waiting for their results. Collected
IDs are skipped on later audits.

Before saving recovered counts, the audit verifies the frozen specification
and exact source bytes and creates `manifest.before_audit_<UTC>.json`. It then
atomically checkpoints the original manifest. Existing IDs and immutable plan
fields are retained; a run becomes `collected` only when every batch has
complete validated results. A process killed after submission but before its
ID was saved remains ambiguous and requires reconciliation with IBM.

Outputs are `Model2/hardware_audit/Audit.md`, `audit.json`, and a directory per
run containing its analysis. The report lists saved IDs, collected results,
queried provider states and appropriate resume/inspection commands. A normal
resume skips collected batches and retrieves known IDs before submitting the
remaining jobs. Terminal errors are reported separately. Only a confirmed
`ERROR/6056` receives the existing runner's explicit retry command; the audit
does not execute that command. Older unsubmitted plans are identified so they
are not mistaken for interrupted copies of a later completed experiment.

To inspect local evidence without an IBM connection or count updates:

```powershell
python Model2/audit_hardware_runs.py --analyze
```

Use `--check-ibm` to query states without retrieving counts. `--account-file`
selects a configured account file. `--manifest PATH` limits the audit to one
manifest and can be repeated; `--output-directory PATH` chooses a report
directory outside the raw run folders.

Completed-job analysis retains each path's original identity and covariance
from the saved source. An incomplete run is explicitly labelled as a subset.
All matched bases or mitigation scales and complete controls must be present
inside each analyzed job. Raw analysis reconstructs populations and currents;
mitigation analysis reports the six predefined profiles with independent
job-local SPAM calibration and descriptive uncertainty. No missing trajectories
are regenerated or substituted, and no partial cohort is certified as the
final full experiment. Count analysis verifies source/plan/count integrity;
it does not hash or load the large QPY archives. The execution runners retain
their own native-archive checks on resume.

If a shared checkout contains only `Plan.md` or exported CSV/Markdown, the
audit records that evidence without inferring current IBM state. Exported raw
tables can be summarized and their control diagnostics plotted, but independent
reconstruction requires the original count-bearing manifest and source.

For the known full mitigation directory, a normal resume on the original
machine is:

```powershell
python Model2/Experiment6/run_hardware.py --resume "Model2/Experiment6/hardware_runs/mitigation44_20261103T145657_688162Z/manifest.json" --execute
```

Use the audit's diagnosis first if it reports a failed or ambiguous job. Keep
the saved source, specification and QPY files with that manifest. Do not start
a new run to continue an old one.
