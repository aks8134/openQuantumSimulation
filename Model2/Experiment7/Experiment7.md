# Model2 Experiment7: inspect IBM timing, then test dynamical decoupling

The hypothesis is that XX pulses in eligible system idle windows reduce device
error without changing the intended reservoir channel. This experiment first
retrieves IBM compiler timing and opens it with `draw_circuit_schedule_timing`.
The viewer includes dynamic/controller operations; `timeline_drawer` is not used.
No DD is inserted until a collected scout and an explicit inspection policy exist.

Physics stays at N=10, J=1, Γ/J=2, pL=pR=0.25, half-mixed preparation, and the
validated 44-step warped cosine from Experiment5. Default seeds are trajectory
29 and transpiler 11. Both baths, all 44 resets, measurement registers and
exchange operations are preserved. All three bases (z, xy, yx) support population
and current comparisons.

The default **4-path scout is for schedule inspection**, not a replacement for
the previous 16,384-path accuracy study. Choose the path count explicitly when
scaling up. A suggested next statistical pilot is 64 paths × 256 shots per
basis/arm. Its adequacy must be evaluated against its own ideal paths and
uncertainty before a full 16,384-path run. The initial implementation tests XX;
XY4 and combinations with twirling/ZNE remain separate experiments.

## Environment and access

Cloud dry-run needs Runtime >=0.51 and a Flex, Premium or On-Prem plan. Open and
Pay-As-You-Go plans do not have dry-run access. Hardware timing collection also
works on older Runtime versions with scheduler timing support. The requirements
below target the new dry-run workflow. Keep the environment for previous
Experiment5/6 resumptions intact. From the repository root in PowerShell:

```powershell
py -m venv .venv-dd
.\.venv-dd\Scripts\Activate.ps1
python -m pip install -r Model2/Experiment7/requirements.txt
```

Account discovery follows existing runners: optional `--account-file PATH`, then
the repository's ignored `IBMRuntime/apikey.json`, then saved Runtime credentials.
If using a new environment with saved credentials, configure the same IBM account
there. No credential values are copied into experiment output. Selecting Phoenix
uses `use_fractional_gates=False`.

## 1. Compile the DD-off timing scout

```powershell
python Model2/Experiment7/run_hardware.py prepare --backend ibm_phoenix --trajectory-count 4 --shots 32
$scout = (Get-ChildItem Model2/Experiment7/hardware_runs/dd44_scout_*/manifest.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

This contacts IBM for the target and calibration, compiles locally, and submits
no jobs. It saves the source/specification, fixed physical path, calibration,
timing model, native circuits, scheduled circuits, hashes and job packing.
Auto-selection uses the earlier quality thresholds (T1 >=80 μs, T2 >=60 μs,
CZ error <=0.005, readout error <=0.02). An explicit path is supported:

```powershell
python Model2/Experiment7/run_hardware.py prepare --backend ibm_phoenix --trajectory-count 4 --shots 32 --physical-path 106 116 115 114 113 112 111 101 91 81 80 90
```

The historical path is only an example and is screened against current values.
For a deliberately fixed path, `--calibration-policy record-only` records the
calibration and checks native connectivity without applying quality thresholds.

Default scout workload is **1 job, 15 circuits, 6,528 shots including controls**,
provided the compiled footprint fits. A dry-run uses no QPU shots. The saved
Plan.md reports the actual footprint-dependent packing and intended shot count.

## 2. Request server timing and collect it

```powershell
python Model2/Experiment7/run_hardware.py submit --manifest "$scout" --dry-run
python Model2/Experiment7/run_hardware.py collect --manifest "$scout"
```

Submission saves each ID before continuing. Collection fetches completed known
jobs only; it does not wait for queued/running jobs or submit anything. Repeat
`collect` when they finish. Unsupported dry-run fails without falling back to a
real QPU run. If the account has no dry-run access, the explicit alternative is
`submit --manifest "$scout" --hardware`; that executes the stated shots and
consumes allocation. Never select both. A manifest cannot mix the two modes.

Open the generated index:

```powershell
Invoke-Item (Join-Path (Split-Path "$scout") "timing/index.html")
```

Each path/basis has a full interactive view with controller/readout channels, a
view of the twelve active drive channels, raw timing text, instruction CSV and
drive-gap CSV. Plotly is embedded for portable HTML. Use the range slider/zoom
to inspect resets and exchange layers within the full view. Rendering can be
repeated without IBM access:

```powershell
python Model2/Experiment7/run_hardware.py render --manifest "$scout"
```

After `Job ... DONE`, collection still downloads/decodes the result, saves the
PUBs, fetches optional metrics/usage, and builds the HTML views. Progress messages
identify each stage, with a waiting message every 20 seconds for long calls.
`DONE` alone is not confirmation that the local files or plots are finished.

For a faster first inspection, separate download from rendering:

```powershell
python Model2/Experiment7/run_hardware.py collect --manifest "$scout" --download-only
python Model2/Experiment7/run_hardware.py render --manifest "$scout" --trajectory 0 --basis z --view full
Invoke-Item (Join-Path (Split-Path "$scout") "timing/path_00000_z_baseline.html")
```

Download-only skips optional metrics/usage requests and HTML generation. If
results are already saved, it checks/reuses them without contacting IBM. A
later ordinary `collect` can fetch the optional details into a separate
`job_0000_details.json`; the timing/count archive and inspected policy evidence
stay unchanged. Raw results and the collected manifest checkpoint are written
**before** optional job-detail calls or Plotly rendering.

Rendering uses saved server metadata and submits no jobs. `--view active`
selects the twelve drive channels; `--view both` produces both views (the
default). Each completed HTML is written atomically and recorded with input and
output hashes. Interrupted rendering reuses verified completed views, and the
index is updated after each view. `--overwrite` explicitly rebuilds them.
Existing HTML produced before render caching was added can still be opened
directly, but is rebuilt once when first used by the updated renderer.

Drive-channel gaps are **candidates for inspection**, not certified free pulse
slots. The CSV keeps branch identity and overlapping controller/readout channels.
It excludes leading/trailing intervals and never merges alternative branches.
The cloud timing metadata is the only source of these plots and tables. Missing
timing or Plotly causes an actionable error; raw job results remain saved.
Compiler timing is not billing time or a per-shot timing trace.

## 3. Choose the DD rule after inspection

Record the logical system sites (1..10), minimum idle ratio and reason. For
example, **only if inspection supports these choices**:

```powershell
python Model2/Experiment7/run_hardware.py policy --manifest "$scout" --system-sites 1 2 3 4 5 6 7 8 9 10 --min-idle-ratio 2 --note "Inspected all scout paths and bases; test XX on fitting system idle windows while leaving reservoir/controller operations intact." --output Model2/Experiment7/dd_policy.json
python Model2/Experiment7/run_hardware.py apply-policy --manifest "$scout" --policy Model2/Experiment7/dd_policy.json --shots 256
$paired = (Get-ChildItem Model2/Experiment7/hardware_runs/dd44_paired_*/manifest.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

The policy references exact scout/result hashes and maps system sites to physical
qubits. Bath indices cannot be selected. Apply-policy creates a **new** experiment;
it does not alter the scout or any prior Model2 run. It inherits the exact scout
trajectories, native circuits, path and target durations. If durations/constraints
changed, prepare and inspect a fresh scout. No seed/path selection on measured
outcomes is permitted.

Dynamic ALAP scheduling is used for both arms. The IBM dynamic DD pass replaces
fitting local idle windows with a single symmetric XX cycle: slack fractions
1/4, 1/2, 1/4. An eligible gap must exceed twice the total pulse duration by
default; hardware pulse alignment also applies. Initialized/reset qubits and
waits following final system readout are skipped. This protects existing slots;
it does not synthesize delays to fill opaque inter-block/controller latency.

Construction checks compare every original gate, measurement and reset with its
local block/start/duration. Each added pair must fit an original idle window on
an allowed system qubit. Native support and all 44 resets are checked again.
There is no optimization after DD insertion. A small adapter recomputes the
DD pass's pulse-length cache from the unchanged target in dt, avoiding the
seconds/dt mismatch found in Runtime 0.49 and 0.51. It also skips terminal readout
waits. The adapter is tested because dynamic scheduling is experimental. Local checks are not used to claim
global wall-clock equality; IBM timing is retrieved for both arms next.

## 4. Check paired compiler timing, then run hardware

```powershell
python Model2/Experiment7/run_hardware.py submit --manifest "$paired" --dry-run
python Model2/Experiment7/run_hardware.py collect --manifest "$paired"
Invoke-Item (Join-Path (Split-Path "$paired") "timing/index.html")
```

`timing/server_comparison.json` compares non-X operation and controller timing.
It excludes X because compiler pulse names alone cannot identify reset X gates
versus DD pulses. Local checks verify the inserted identities and placements.
The compiler comparison deliberately includes implicit FINI/readout/controller
operations. It also checks the added drive-channel X count per physical qubit
against construction, so a missing/renamed/optimized-away sequence needs
inspection. A mismatch needs investigation before claiming DD alone changed the
outcomes. Cloud dry-run counts are randomized mock data and physical analysis
refuses them.

After a successful paired timing check, clone the **identical QPY files** into a
fresh hardware job ledger, with no mock counts or job IDs reused:

```powershell
python Model2/Experiment7/run_hardware.py clone-for-hardware --manifest "$paired"
$hardware = (Get-ChildItem Model2/Experiment7/hardware_runs/dd44_hardware_*/manifest.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python Model2/Experiment7/run_hardware.py submit --manifest "$hardware" --hardware
python Model2/Experiment7/run_hardware.py collect --manifest "$hardware"
python Model2/Experiment7/run_hardware.py analyze --manifest "$hardware"
```

Clone refuses changed compared anchors, total end times or added X counts. Submission
checks target timing, current native support and active backend status again.
If dry-run is unavailable, the alternative is an explicit small real paired
timing/performance pilot via `submit --manifest "$paired" --hardware`, after
inspecting the real scout. Such a pilot has not had a separate cloud timing
preflight; inspect its returned schedules before drawing conclusions.

Every real job contains matched trajectory/basis/arm groups with alternating
requested order and fresh zero/one/half-mixture controls per arm. Actual dispatch
order is controlled by IBM. Short controls diagnose preparation/readout; they do
not validate full-depth DD survival. Paired means, populations, currents,
same-path ideal errors, shot SE and descriptive across-path SE are reported in
`analysis/Analysis.md`, `analysis/analysis.json` and `analysis/observables.csv`.
No SPAM inversion, twirling, ZNE or post-hoc job exclusion is applied.

Four paired paths at 256 shots imply **1 job, 30 circuits, 18,432 shots**, if the
native footprints fit. A 64-path paired pilot would have 390 circuits and
110,592 shots in one job if it fits. To test 64 paths, start with a **64-path
scout**, inspect it, then apply its policy. A smaller scout's policy cannot
silently expand to unseen trajectories. Full 16,384 paths remain an explicitly
requested later stage; packing is computed from all actual compiled circuits,
including DD, at 80% of IBM's instruction heuristic and the execution limit.

## Recovery and limits

Re-running `submit` in the same mode keeps every saved job ID and submits only
unsubmitted batches. It never automatically retries failed or ambiguous jobs.
An interruption after submission but before ID persistence leaves `submitting`;
reconcile the uniquely tagged job in IBM before proceeding. `collect` does not
compile or run QPU jobs. Source, QPY and immutable plan hashes are checked on
every command. Generated results and HTML remain local.

No Phoenix jobs were submitted while implementing this experiment. Offline fake
backend tests validate construction and bookkeeping, not DD effectiveness. An
idle/dephasing noise study remains necessary before scaling a hardware pilot;
a gate-only unscheduled FakeKingston model cannot establish a DD benefit.

References: [IBM timing viewer](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-runtime-circuit-timing),
[cloud dry-run and plan access](https://quantum.cloud.ibm.com/docs/en/guides/dry-run-jobs),
[dynamic DD insertion](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-scheduling-pad-dynamical-decoupling),
[job limit heuristic](https://quantum.cloud.ibm.com/docs/en/guides/job-limits).
