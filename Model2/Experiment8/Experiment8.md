# Model2 Experiment8: twirling + ZNE + SPAM + DD

For the reviewed **16,834-path, 32-shot, 24-job** production cohort, see
[RuntimeBatch.md](RuntimeBatch.md) for `--batch-size 9141`,
`--execution-mode batch`, and resumable Batch submission.
Preparation supports `--workers 4` and per-group recovery receipts; see
[ParallelPreparation.md](ParallelPreparation.md) before stopping an older run.

Every new hardware circuit uses the XX DD policy. There is **no DD-off hardware
condition**. Experiment6 supplies historical DD-off results; different execution
dates and calibrations prevent a paired causal estimate of DD's contribution.
The raw and intermediate profiles in this experiment also have DD enabled.

The physical setup is unchanged: N=10, J=1, Γ/J=2, pL=pR=.25, exchange prefactor
1/4, Δ=0, half-mixed preparation, jump-first/even-odd evolution, and the frozen
44-step warped cosine. The original default uses 16,384 seed-29 paths;
the reviewed Batch production cohort uses 16,834. Both use
32 shots per raw arm/factor, four randomizations, factors 1/1.25/1.5, 2,048
control shots per preparation/treatment/job, and 2,000 bootstrap replicates.
Smaller explicit cohorts use the same first seed streams for inspection.

The initial mapping comes from the saved Experiment6 reference unless
`--physical-path` supplies another twelve-qubit chain. Preparation screens the
current backend calibration by default. `--calibration-policy record-only` is
available for an explicit fixed-chain study; both policies retain connectivity,
native-ISA, resource, timing and intentional-reset checks.

## Construction and measurement correction

1. Compile each native Model2 trajectory with transpiler seed 11 and level 3.
2. Apply native CZ Pauli twirling, selective CZ→CZ·CZ·CZ folding and balanced
   terminal X measurement masks. CZ counts are 880/1,100/1,320, and all 44 bath
   resets remain. The raw profile has the original CZ count and no twirl mask.
3. ALAP-schedule each completed variant, then insert XX in eligible system-only
   idle windows. Ratio 10 uses the real target's two-X duration: for 40 ns X
   pulses the cutoff is strictly greater than 0.8 µs. Spacings are
   [.25,.5,.25], with one cycle per eligible gap, reset-aware skipping and no
   protection after final readout. There is no further gate optimization.
4. Locally compare against the same variant's temporary scheduled reference:
   original operations and timestamps remain fixed, and every XX pair fits an
   allowed window with the target alignment. That temporary reference is not
   submitted. All submitted raw/twirled/control circuits carry the DD policy;
   short controls can have no eligible idle window and therefore no added X.
5. Decode measurement masks in software. Correct populations per qubit and per
   job using `(p-a)/(b-a)`, where a and b are that treatment/job's zero/one
   responses. The half-mixture is held out. Average corrected jobs with their
   trajectory weights and retain all six predefined intermediate/final profiles.

This is the same custom SPAM model as Experiment6, **not IBM Runtime TREX**.
Runtime automatic DD, gate twirling and measurement twirling stay disabled;
the corresponding transformations are explicit in the saved circuits. Raw and
twirled treatments each have their own controls in every job.

DD can change how the folded circuits' noise scales. Retain factor-resolved
populations, linear-fit residuals, DD pulse/window counts, held-out checks and
bootstrap uncertainty. A favorable intercept does not prove that linear
extrapolation to zero noise is accurate. No clipping, target fitting or job
exclusion is introduced.

## Commands from the repository root (PowerShell)

Use the DD environment with Runtime 0.51 and Plotly:

```powershell
python -m pip install -r Model2/Experiment8/requirements.txt
```

First create a small **DD-on-only** timing inspection plan. This step is offline:

```powershell
python Model2/Experiment8/run_hardware.py plan --trajectory-count 4 --capture-timing
```

Set the printed manifest path explicitly. Compile it against Phoenix without
submitting:

```powershell
$inspection = "<printed Experiment8 manifest.json path>"
python Model2/Experiment8/run_hardware.py prepare --resume "$inspection"
```

The default historical reference is the committed Experiment6 `results_bundle.zip`.
`--historical-reference` accepts another complete Experiment6 analysis JSON or
bundle when creating a new plan. The reference's configuration, job IDs, profile
summaries and input hash are copied to `historical_reference.json`; its counts
and old job IDs are never added to the new submission ledger.

Submit this small plan for cloud compiler inspection with randomized mock counts:

```powershell
python Model2/Experiment8/run_hardware.py submit --manifest "$inspection" --dry-run
python Model2/Experiment8/run_hardware.py collect --manifest "$inspection"
```

Repeat collection when pending jobs finish. Render path zero, variant zero, at
**all three factors**, using IBM's `draw_circuit_schedule_timing`:

```powershell
python Model2/Experiment8/run_hardware.py render --manifest "$inspection" --trajectory 0 --randomization 0 --view full
```

`--factor-index 0|1|2`, `--arm raw`, and `--view active` select other views.
Verified HTML views are cached. Plots always use the server-returned schedule;
there is no `timeline_drawer` or locally synthesized timing fallback. Since no
DD-off circuits were submitted, these are DD-on inspection views, not paired
DD-off/on service timing comparisons.

For a real hardware replay of the same small DD-on circuits, clone the completed
dry-run into a fresh ledger and use its printed manifest:

```powershell
python Model2/Experiment8/run_hardware.py clone-for-hardware --manifest "$inspection"
$run = "<printed new hardware manifest.json path>"
python Model2/Experiment8/run_hardware.py submit --manifest "$run" --hardware
python Model2/Experiment8/run_hardware.py collect --manifest "$run"
python Model2/Experiment8/run_hardware.py analyze --manifest "$run"
python Model2/Experiment8/run_hardware.py bundle --manifest "$run"
```

Only `submit` sends jobs. Existing IDs are skipped. `collect` retrieves only
known DONE results and leaves active, failed, cancelled or unknown-ID jobs
unchanged. It saves each job atomically before checkpointing the manifest and
can recover a result file left by an interrupted checkpoint without another
provider request. Optional `collect --job-details` fetches metrics/usage only
after counts are durable. There is no automatic failed-job replacement.

`inspect-job --manifest "$run" --job-id ID` reports an existing job's diagnostics.
An interrupted submission with status `submitting` but no saved ID requires
reconciliation by job tag before submission can continue.

## Full workload and memory limits

For a backend-dependent minimum-job analysis, use
[`estimate_jobs.py`](estimate_jobs.py) as described in
[JobEstimation.md](JobEstimation.md). It counts all requested paths by default,
includes the final DD pulses and delays, and reports counting lower bounds and
feasible partitions without imposing the default 512-circuit cap. It does not
submit jobs or change a hardware manifest.

Create a separate production plan for the full cohort. Production omits the
large server timing payload, while retaining local DD validation and summary
counts for every circuit:

```powershell
python Model2/Experiment8/run_hardware.py plan --backend ibm_phoenix --trajectory-count 16834 --shots 32 --batch-size 9141 --instruction-fraction 0.8 --execution-mode batch
$full = "<printed production manifest.json path>"
python Model2/Experiment8/run_hardware.py prepare --resume "$full"
python Model2/Experiment8/run_hardware.py submit --manifest "$full" --hardware
```

This explicitly selects the reviewed **24-job, 219,202-circuit, 2,449,664-shot**
balanced configuration and submits through one IBM Runtime Batch without
waiting for each job's result. See RuntimeBatch.md for lifetime and recovery.

The unchanged default circuit cap is **512**, including controls. As a
comparison, that cap gives 443 jobs for 16,834 paths, or **432 planned jobs,
219,472 circuits and 7,405,568 shots** for the original 16,384 paths. The four-path
inspection is **1 job, 67 circuits and 12,800 shots**. These totals include the
complete job-local controls; there is no DD-off arm. They are accounting totals
for the chosen grouping, not an estimate of QPU seconds or compiler acceptance.

The earlier Experiment6 twenty-job packing used a much larger circuit cap.
Do not assume it remains feasible after adding DD. `--batch-size` is a new-plan
option; actual DD-on footprints must fit the requested fraction of the published
per-qubit instruction allowance and execution limit before any submission.
Raising the cap also raises the local memory needed for Sampler serialization.
The defaults avoid materializing ten thousand deep circuits in one submission.

Compilation uses public QPY serialization in shards of at most one complete
13-circuit trajectory group or one five-circuit control group. Only one shard
is loaded for validation; submission retains at most one job. Preparation can
resume completed job checkpoints without recompiling them in the same target
timing/compiler environment. Count analysis never loads QPY and reads one job's
counts at a time. Full native compilation and scheduling can be substantial
local CPU work; this code does not promise a wall-clock duration or peak RAM.

Flat reset/measurement DAG ordering caches reachability while preserving the
SDK's operation order; control flow falls back to the SDK. Tests compare exact
node ordering on mixed reset/measurement circuits and a full folded variant.
The scheduler still uses Runtime's ALAP and DD passes. Third-party attribution
and the upstream license are under `third_party/`.

Timing capture is restricted to <=64 paths and jobs of at most 128 circuits to
bound inspection downloads. Full production uses the same DD policy without
requesting per-PUB compiler traces. No new full run is submitted by planning,
compilation, inspection rendering, collection, analysis or packaging.

## Analysis and transfer

`analyze` requires a completely collected **hardware** ledger. Dry-run counts
are refused. Outputs under `Model2/Experiment8/analysis/<run>/` include all six
profiles, population/noise-scaling figures, controls, descriptive uncertainty,
stationary-target errors, and `historical_comparison.csv`. Differences from
Experiment6 are labelled as unpaired historical context; a different cohort or
mapping is recorded explicitly. No paired DD-effect confidence interval is claimed.

`bundle` creates `results_bundle.zip` atomically with source, plan, counts,
available timing text, history, DD summaries and analysis. It excludes QPY and
HTML by default. `--include-qpy` opts into larger compiled-archive transfer.
The 0.02 comparison applies to physical stationary populations; agreement with
the finite 44-step same-path ideal is reported separately. Currents are not
measured and the SPAM model retains the limitations seen in Experiment6 controls.

The new experiment does not modify Experiment6 or Experiment7 datasets. Local
tests use a real CZ backend target and mocked IBM jobs, with no QPU submission.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_e8_mpl \
  .venv/bin/python -m pytest -q tests/test_model2_combined_dd.py
```
