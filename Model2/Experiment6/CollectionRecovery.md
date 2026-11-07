# Resume Model2 mitigation collection after stopping the local process

The hardware job IDs are checkpointed immediately after submission. A local
Ctrl+C does not send a cancellation request to IBM. Stop the old process and
wait for its PowerShell prompt before updating code or starting another writer
against this run directory. Preserve the run directory, especially its manifest
and source. An interrupted result download can be repeated using the saved ID.

For the existing twenty-job run, use the following commands from the repository
root after updating the code:

```powershell
$mitManifest = "Model2/Experiment6/hardware_runs/mitigation44_20261103T145657_688162Z/manifest.json"

python Model2/Experiment6/run_hardware.py --resume "$mitManifest" --collect-only
```

Collection only uses recorded job IDs. It checks their status and retrieves
`DONE` results; queued, running, failed, cancelled, or unknown-ID jobs remain
unchanged. It does not construct a Sampler, query a backend calibration, read
QPY archives, compile circuits, submit jobs, perform analysis, or create a ZIP.
It needs an IBM account only when an unsaved job must be queried. Existing saved
counts and recoverable local journals do not require an account. The
`--allow-calibration-drift` submission option is unnecessary for collection.

The command prints the number of jobs saved. If it reports fewer than 20/20,
inspect its status messages and repeat the same command when the pending jobs
finish. An `ERROR` job needs inspection rather than an automatic replacement.

Once collection reports **20/20 jobs saved**, analyze locally:

```powershell
python Model2/Experiment6/run_hardware.py --resume "$mitManifest" --analyze-only
```

This step does not access an IBM account or submit jobs. It retains the original
paths, masks, shots, fold factors, six mitigation profiles, 2,000 bootstrap
replicates, seeds, and independent calibration inside each job. It prints its
current phase and a waiting heartbeat every twenty seconds during long stages.
A heartbeat signals elapsed time, rather than guaranteed progress or completion.

Outputs are saved under
`Model2/Experiment6/analysis/mitigation44_20261103T145657_688162Z/`.
The raw run directory also receives `hardware_results.json` and
`results_bundle.zip`. The default ZIP contains the manifest, source,
specification, available calibration snapshots, count journals, and analysis.
It excludes compiled QPY files, which can make packaging very slow and large.
The ZIP is replaced atomically after packaging succeeds.

## Memory and interruption handling

The earlier runner could retain the last submitted circuit list during result
collection. Analysis then deserialized every available QPY batch, and packaging
compressed every QPY archive. Jobs containing roughly ten thousand deep
circuits could consume several GB of local memory at these stages.

The updated execution path releases its circuit list before downloading results.
Count-only analysis reuses already loaded manifest/source objects and reduces
decoded counts to moments one job at a time. It still retains the saved manifest,
numeric trajectory arrays, and bootstrap workspace; this is a memory reduction,
not a guarantee of a particular peak RAM usage. SDK result decoding still needs
one job's response in memory.

The collector backs up the manifest before its first possible change. For each
downloaded job, it validates the number of PUBs, ten-bit final-register counts,
and shot totals. It then atomically saves `job_NNNN_counts.json` before updating
the larger manifest. If interrupted between those writes, another collection
loads that journal, checks the plan/job/range identity and counts, and finishes
the manifest checkpoint without another result request. Journals referenced by
the manifest also have a recorded SHA256 hash.

Optional metrics requests now follow the result checkpoint and are disabled by
default. To collect usage details for a particular saved ID, the existing
`--inspect-job JOB_ID` command remains available. `--include-job-metrics` opts
into metrics requests after newly downloaded results have been saved.

QPY hash/identity/resource checks still run on circuits used for new submissions.
For an explicit offline circuit re-audit during analysis, add
`--verify-circuits`. For a ZIP containing the large compiled archives, add
`--include-qpy`. These options restore the expensive work deliberately and are
not needed to recover or analyze already executed counts. Analysis records which
archives were checked, skipped, or unavailable.

The safe collection command does not reset job IDs, retry failed jobs, alter the
physical path, or bypass a submission guard. An interrupted submission whose
job ID was never saved still requires reconciliation with IBM.

## Software validation

Recovery tests use mocked jobs and forbid Sampler construction, calibration
selection, circuit loading, and analysis in collection-only mode. They exercise
interrupted manifest writes, local journal recovery, metrics interruptions,
active/failed/unknown-ID jobs, invalid counts, offline analysis, default QPY
exclusion, and preservation of the previous ZIP after interrupted packaging.
An offline comparison also reproduced the entire original analysis report
exactly from the same synthetic counts, including all six profiles, bootstrap
intervals, and independent job-local controls.

```bash
MPLCONFIGDIR=/tmp/model2_mit_mpl .venv/bin/python -m pytest -q \
  tests/test_model2_mitigation_collection.py \
  tests/test_model2_hardware_mitigation.py \
  tests/test_model2_hardware_audit.py
```
