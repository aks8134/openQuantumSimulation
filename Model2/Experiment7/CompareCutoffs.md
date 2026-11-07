# Compare XX cutoffs of 0.16 and 0.8 microseconds

This creates one matched experiment with three conditions: scheduled DD-off,
XX with minimum idle ratio 2, and XX with minimum idle ratio 10. On the inspected
Phoenix target an X pulse is 40 ns, so these ratios select local idle windows
strictly longer than 0.16 and 0.8 us, respectively. Ratios and the actual
target-dependent cutoffs are recorded in the generated Plan.md. Both DD arms
use the same ten system sites, one symmetric XX cycle per eligible gap, and
no bath DD. No SPAM correction, twirling or ZNE is added.

All arms use the exact same saved trajectory circuits, source, physical path,
measurement bases, shots and target timing. The native source circuits are
copied from the scout rather than transpiled again. Original local operation
times and all 44 resets are checked independently for each DD arm. All arms
and bases of a path stay in the same job. Six permutations of the three arm
orders are cycled over path/basis groups to balance requested positions;
actual device dispatch order is controlled by IBM. Each job repeats zero, one
and half-mixture controls for each of its three arms.

In the existing `.venv-dd` PowerShell environment, from the repository root:

```powershell
$scout = "Model2/Experiment7/hardware_runs/dd44_scout_20261107T163922_266664Z/manifest.json"

python Model2/Experiment7/run_hardware.py policy --manifest "$scout" --system-sites 1 2 3 4 5 6 7 8 9 10 --min-idle-ratios 2 10 --note "Inspected all four Phoenix paths and three bases; compare broad (>0.16 us) and selective (>0.8 us) XX against the same DD-off baseline." --output Model2/Experiment7/dd_policy_016_vs_080.json

python Model2/Experiment7/run_hardware.py apply-policy --manifest "$scout" --policy Model2/Experiment7/dd_policy_016_vs_080.json --shots 256 --control-shots 2048

$comparison = (Get-ChildItem "Model2/Experiment7/hardware_runs/dd44_paired_*/manifest.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

These commands write a policy and compile a new experiment. They do not submit
jobs. Use the full original scout folder on your computer, including its QPY
files; the compact ZIP shared for inspection omits those files. If target
durations/constraints changed since inspection, the compiler refuses to reuse
the stale scout. No previous source, manifest, job ID or result is modified.

For the four inspected trajectories at 256 shots per basis/arm, the expected
packing is **1 job, 45 circuits, 27,648 intended shots**, subject to the actual
compiled instruction checks:

| Component | Circuits | Intended shots |
| --- | ---: | ---: |
| 4 paths x 3 bases x 3 conditions x 256 shots | 36 | 9,216 |
| 3 preparations x 3 conditions x 2,048 shots | 9 | 18,432 |
| Total | 45 | 27,648 |

Request the three-arm compiler timing first:

```powershell
python Model2/Experiment7/run_hardware.py submit --manifest "$comparison" --dry-run
```

Once that job is DONE:

```powershell
python Model2/Experiment7/run_hardware.py collect --manifest "$comparison" --download-only
python Model2/Experiment7/run_hardware.py render --manifest "$comparison" --trajectory 0 --basis z --view full
Invoke-Item (Join-Path (Split-Path "$comparison") "timing/index.html")
```

This renders three full-channel views for a first check instead of all 72 full
and active HTML views. Download-only saves all 45 PUBs regardless of the render
selection. Dry-run measurement outputs are randomized and cannot decide which
policy performs better. Rendering or compilation alone cannot establish a DD
benefit.

After inspecting the returned timing, prepare a separate hardware ledger:

```powershell
python Model2/Experiment7/run_hardware.py clone-for-hardware --manifest "$comparison"
$hardware = (Get-ChildItem "Model2/Experiment7/hardware_runs/dd44_hardware_*/manifest.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

Clone checks **both DD arms, all four paths and all three bases**, not just the
rendered preview. It requires unchanged compared non-X/controller timing,
unchanged end times, and the expected added X counts per physical qubit for
each arm. It copies identical compiled QPY into a fresh job ledger without
mock results or reused job IDs. The following commands are the explicit real
QPU stage; do not use them merely to retrieve compiler timing:

```powershell
python Model2/Experiment7/run_hardware.py submit --manifest "$hardware" --hardware
python Model2/Experiment7/run_hardware.py collect --manifest "$hardware" --download-only
python Model2/Experiment7/run_hardware.py analyze --manifest "$hardware"
```

Hardware analysis reports population/current profiles and same-path ideal
errors for all three arms. It includes three direct contrasts: each DD arm
minus baseline, and ratio-10 DD minus ratio-2 DD. Shot uncertainty for the last
contrast uses only those two DD arms; baseline noise is not added to it. The
combined maximum error and RMSE are recorded in `method_errors.csv`;
`error_comparisons.csv` gives treatment-minus-reference error differences.
Negative differences favor the treatment in the observed sample. These error
rankings are descriptive, with no confidence interval for the maximum-error
difference; the per-observable uncertainty vectors should also be inspected.

The four-path stage remains an inspected-cohort pilot. It is not certification
of the stationary-state tolerance or of the best cutoff for all 16,384 paths.
A small apparent improvement needs a larger matched confirmation before
choosing a production policy. To expand the cohort, prepare and inspect a
scout with the desired trajectory count and then create its comparison policy.

The earlier single-policy commands and saved two-arm manifests remain
supported. This comparison has been implemented but not run on IBM by the
assistant.
