# The reviewed 24-job production run in IBM Runtime Batch mode

Use the existing `.venv-dd` environment; no new dependencies are added. The reviewed configuration is **16,834 trajectories × 32 shots per raw arm/factor, 24 jobs, 219,202 circuits and 2,449,664 shots including controls**. Circuit cap 9,141 gives 701 or 702 complete matched trajectories per job. Physics, twirling, ZNE factors 1/1.25/1.5, job-local SPAM and ratio-10 XX DD are unchanged. All new treatments use DD.

From the repository root in PowerShell:

```powershell
python -u Model2/Experiment8/run_hardware.py plan --backend ibm_phoenix --trajectory-count 16834 --shots 32 --batch-size 9141 --instruction-fraction 0.8 --execution-mode batch

$ddManifest = "<manifest.json path printed by plan>"
python -u Model2/Experiment8/run_hardware.py prepare --resume "$ddManifest"
python -u Model2/Experiment8/run_hardware.py submit --manifest "$ddManifest" --hardware
```

The Python commands also work in bash; set `ddManifest="<manifest.json path>"` there. `plan` is offline. `prepare` constructs and validates every native circuit and actual job footprint before a Runtime Batch is opened. It retains the existing calibration screen and bounded QPY shards. Full preparation can take substantial local CPU time and disk space; numeric estimate caches do not contain executable QPY.

Only `submit --hardware` submits QPU jobs. It creates one `qiskit_ibm_runtime.Batch` and uses `SamplerV2(mode=runtime_batch)` for every job. It queues all 24 without waiting for `job.result()`. The Batch ID is saved before the first submission, and each job ID and association are saved immediately afterward. Only one job's circuits are loaded at a time. Larger individual jobs still need sufficient RAM for Sampler serialization.

`--batch-size` is the circuit cap per job; `--execution-mode batch` selects the IBM execution mode. These are 24 separate jobs in one Batch. IBM can parallelize classical processing and controls QPU scheduling; this does not promise simultaneous execution of 24 jobs on one QPU. See [IBM's Batch guide](https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-batch).

After all IDs are saved, `Batch.close()` stops new submissions while queued/running jobs continue. The code never calls `cancel()`. Collect separately, repeating until all jobs are collected, then analyze and package:

```powershell
python -u Model2/Experiment8/run_hardware.py collect --manifest "$ddManifest"
python -u Model2/Experiment8/run_hardware.py analyze --manifest "$ddManifest"
python -u Model2/Experiment8/run_hardware.py bundle --manifest "$ddManifest"
```

Collection never submits, waits on active jobs, loads QPY or launches analysis. The new cohort differs from Experiment6's 16,384 trajectories; that historical difference and all actual Batch IDs remain recorded.

## Batch lifetime and interrupted submission

Provider/account default lifetime applies unless overridden. The actual provider `max_time` is printed and saved. TTL begins when the first job starts; at expiry, running jobs may finish while queued jobs may fail. IBM currently documents defaults of eight hours for paid plans and ten minutes for Open. See [Batch lifetime](https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-batch#batch-length).

For an account permitting the requested lifetime, use `submit ... --hardware --batch-max-time "8h"` without recompiling. This option can also be set at planning time. An existing Batch's TTL cannot be changed; omit the override when resuming it.

After local interruption, rerun the original submit command. It reopens the saved Batch if accepting jobs and skips known IDs. It leaves an incomplete Batch open on error. A failed close after all IDs were recorded is retried without another submission; a provider-closed Batch is recognized without another close request.

If a Batch has stopped accepting jobs and IDs remain missing, the script stops without silently creating another Batch. Explicit recovery uses another Batch for missing jobs only:

```powershell
python -u Model2/Experiment8/run_hardware.py submit --manifest "$ddManifest" --hardware --new-batch-for-remaining
```

Old Batch IDs remain in history. This does not replace failed jobs or retry ambiguous submissions. A `submitting` record without an ID requires reconciliation by job tag. A provider Batch-ID mismatch is also retained and blocks further submission. Dry-run timing inspections use job mode with mock counts and never create a hardware Batch. Existing manifests without an execution-mode field retain job-mode behavior.

## Existing estimate caches

Experiment8 now accepts 16,834 trajectories; Experiment6 keeps its 16,384 limit. Completed numeric caches remain readable after transport/validation changes:

```powershell
python -u Model2/Experiment8/estimate_jobs.py --backend ibm_phoenix --trajectory-count 16834 --shots 32 --report-only
```

Report-only verifies sealed numeric context/units without unpickling a target, compiling or querying IBM. New native compilation still refuses changed compiler/code context, preventing a partial cache from mixing versions. The estimate describes its captured target; production preparation/submission check actual resources and timing. Batch mode does not relax those checks or guarantee server-compiler acceptance.
