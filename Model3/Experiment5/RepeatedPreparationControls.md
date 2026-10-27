# Repeated preparation controls

The mitigation pilot's initial half-mixture was too low on physical qubit 48: 0.4150 with raw readout and 0.4375 with terminal-twirled readout. Zero/one controls predicted 0.4929 and 0.4961. However, those calibration controls and the mixture ran in different jobs, whose start times were about 97 minutes apart. The previous data cannot distinguish a preparation/context problem from a change between jobs.

[run_repeated_preparation_controls.py](run_repeated_preparation_controls.py) repeats all three preparations under both readout treatments **within every round of every job**. The physical mapping and source are inherited from the mitigation pilot. Default optimization settings, seeds, measurement masks and Runtime options match that pilot. It uses the same preparation construction and native compilation/transform verification as that runner. No Trotter evolution, CZ gates, bath collisions, or reservoir resets are included.

## Default workload

| Setting | Value |
|---|---:|
| Jobs | 2 |
| Complete control rounds per job | 2 |
| Preparations per round | all-zero, all-one, half-mixture |
| Readout treatments | raw and terminal-twirled |
| Total shots per preparation/treatment/round | 2048 |
| Terminal-twirled variants | 4, with 512 shots each |
| Circuits per round | 15 |
| Circuits per job | 30 |
| Total circuits | **60** |
| Total shots | **49,152** |

Each raw preparation gets 2048 shots in one circuit. Each twirled preparation gets four complementary-mask circuits, totalling 2048 shots. The half-mixture uses the same rotation and ignored intermediate Z-measurement preparation as the evolution experiment. Only final outcome bits are unmasked; preparation bits remain ignored. The controls here have no CZs, so “twirled” means terminal-measurement twirling alone.

The fixed reference path is:

`49 – 39 – 38 – 48 – 58 – 57 – 56 – 66 – 76 – 86 – 85 – 84`.

The system sites are the ten interior vertices; physical qubit 48 is site 3. The two baths are allocated as in the reference but unused. Runtime gate/measurement twirling and dynamical decoupling remain disabled; the explicit terminal masks are decoded locally. Initialization remains enabled.

Control circuits are shuffled within each requested round, with no round split across jobs. A requested round is an analysis group, not a verified time interval: hardware dispatch order and timing are not guaranteed. The two jobs run sequentially without a deliberately imposed waiting period. A calibration snapshot is saved before each new job submission. Known job IDs are recovered on resume; an ambiguous submission with no recorded ID blocks automatic retry.

The path is kept fixed for this diagnostic, including its weak qubits. Connectivity and native instructions are checked, and calibration is recorded; the runner does not reject a qubit merely because its calibration has deteriorated. It does not choose a replacement mapping or launch an evolution run.

## Run on Windows PowerShell

From the repository root, this **submits the two controls jobs**:

```powershell
python Model3/Experiment5/run_repeated_preparation_controls.py --execute
```

It reads `Model3/Experiment5/hardware_runs/ibm_phoenix_mitigation_20261026T184843_413808Z` by default. Use `--reference-run <folder>` to use another compatible mitigation pilot. The backend and physical mapping come from that reference; no shell-specific path escaping is necessary in the default command.

With no action flag, or with `--plan-only`, the script only writes an offline plan and does not access an account/backend. `--compile-only` queries the backend and saves circuits without submitting. To recover an interrupted run, use the actual manifest printed by the runner:

```powershell
python Model3/Experiment5/run_repeated_preparation_controls.py --resume "Model3/Experiment5/hardware_runs/ibm_phoenix_repeated_preparation_<timestamp>/manifest.json"
```

Resume can submit remaining jobs; a fully collected resume only regenerates offline analysis. Do not use `--execute` to recover a previous run, since it creates a fresh run.

Outputs are saved under `hardware_runs/ibm_phoenix_repeated_preparation_<timestamp>/`. Transfer the final `results_bundle.zip` or the updated run folder. The bundle contains raw counts, source/provenance, calibration snapshots, job metrics and analysis; native QPY binaries stay local and are excluded from the transfer archive. Retain them locally. Analysis is under `analysis/hardware/<run>/repeated_preparation/`, including `Analysis.md`, all-site `controls.csv`, and a three-site mixture figure showing sites 1, 3 and 9 for N=10.

For local analysis of transferred results, with no account/backend access:

```bash
.venv/bin/python Model3/Experiment5/run_repeated_preparation_controls.py --analyze-only Model3/Experiment5/hardware_runs/ibm_phoenix_repeated_preparation_<timestamp>
```

## Evaluation

For each site, treatment and round, let $a$, $b$ and $m$ be the observed zero, one and mixture populations. The local affine-response prediction and mixture residual are

$$
m_{\mathrm{pred}}=\frac{a+b}{2},
\qquad r=m-\frac{a+b}{2}.
$$

The exploratory corrected mixture is

$$
\widehat n_{\mathrm{mix}}=\frac{m-a}{b-a},
$$

with expected value $1/2$ under the stable local response assumption. Shot covariances are calculated from each PUB's joint bitstring counts. Delta-method uncertainty includes zero/one calibration uncertainty. Corrections with contrast $b-a<0.5$ are marked unavailable rather than inverted or clipped. Each job is also summarized by pooling its rounds; calibration is never borrowed from a different job. No ideal evolution target is used to fit a correction.

Approximate normal residual tests receive a Holm adjustment over all sites, treatments, round-level checks and pooled-job checks. Default N=10 gives 120 comparisons. These are diagnostic tests under independent-shot and stable-response assumptions, not certified bounds or a claim that unflagged controls are accurate. Raw zero/one populations, contrast and both mixture estimates remain visible even when no mixture discrepancy is flagged.

- A repeated mixture discrepancy within the same round weakens the explanation based solely on calibration being borrowed across jobs. It still does not isolate preparation, measurement context, or dynamic-circuit latency.
- Different round/job behavior motivates closer attention to temporal/context dependence. It does not prove drift, since grouping and actual dispatch timing are imperfect.
- Healthy controls support a later matched mitigation confirmation; they do not establish that 960-CZ evolution or ZNE will meet 0.02 accuracy.

This diagnostic does not change the first-order cosine schedule or run a larger mitigated experiment automatically.
