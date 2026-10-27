# First-order Model3 mitigation

The 48-step first-order schedule, selected jump branches, Hamiltonian, reservoir rates and initial half-mixture stay unchanged. The default chain is the alternative mapping from the paired evaluation:

`49 – 39 – 38 – 48 – 58 – 57 – 56 – 66 – 76 – 86 – 85 – 84`.

There are two independent entry points:

- [analyze_readout_mitigation.py](analyze_readout_mitigation.py): existing-data exploratory marginal SPAM correction, entirely offline.
- [run_mitigation.py](run_mitigation.py): matched raw/twirled/selectively folded pilot. **Default is an offline plan.** Only explicit `--execute` or `--resume` submits jobs; `--compile-only` queries the backend and compiles without submission.

The follow-up [repeated preparation controls](RepeatedPreparationControls.md) test the pilot's mixture discrepancy with complete raw/twirled zero, one and mixture controls repeated inside each job. Its default workload is 60 circuits and 49,152 shots across two jobs, without Trotter evolution.

[Pure circuit/statistical helpers](mitigation.py), [tests](../../tests/test_model3_mitigation.py), [reviewable default workload](inputs/mitigation_pilot_summary.json).

## Existing-data correction

For each site, let $a$ be the measured population after an all-zero control, and $b$ after an all-one control. Under a stable local affine-response model,

$$
n_{\rm observed}=a+(b-a)n_{\rm true},\qquad
\widehat n_{\rm corrected}=\frac{n_{\rm observed}-a}{b-a}.
$$

The all-zero/all-one circuits include initialization and preparation errors. Consequently this is an **exploratory SPAM correction**, not demonstrated readout-only inversion. It does not separate correlated or state-dependent readout, drift, preparation error or evolution noise. The half-mixture is held out of calibration. Calibration contrast below 0.5 is rejected rather than inverted unstably. Estimates are never clipped into [0,1], and neither the physical target nor ideal trajectories are used to fit the correction.

From the repository root, the existing-data analysis is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model3_boundary_analysis/mpl .venv/bin/python Model3/Experiment5/analyze_readout_mitigation.py
```

It defaults to the latest paired mapping run and writes separate files under `analysis/hardware/<run>/readout_spam_exploratory/`. `--run-directory`, `--bootstrap`, `--seed` and `--output-directory` are configurable. Original raw/analysis files are preserved.

The existing-data result is already available in [Analysis.md](analysis/hardware/ibm_phoenix_mapping_comparison_20261026T050356_310922Z/readout_spam_exploratory/Analysis.md). On the alternative chain, ideal MAE changes from **0.017388 to 0.018939** and the ideal maximum discrepancy from **0.054126 to 0.058146**. The descriptive calibration/path bootstrap supports worsening in this example. On the current chain, MAE moves from **0.023005 to 0.021896**, with an interval including no change. The correction is retained as a diagnostic result rather than assumed beneficial or used automatically to select hardware parameters.

## Pilot arms and randomization

Every selected saved trajectory appears in all four treatment/noise arms:

1. Raw, untwirled at the original CZ count.
2. CZ-twirled and terminal-measurement-twirled, CZ factor 1.
3. The same treatment with selective CZ factor 1.25.
4. The same treatment with selective CZ factor 1.5.

The twirled arms use four explicit seeded circuit variants. Frames and nested fold subsets match across factors. Terminal measurement masks occur in complementary pairs, so each measured site is flipped in exactly half the variants. They are fixed across trajectories/noise levels and matched by the calibration controls. Initial preparation measurements and bath resets are never measurement-twirled. The outcome register is decoded by XOR with the recorded terminal mask; preparation bits remain ignored.

All variants/factors of one trajectory stay in the same batch and are shuffled within that group. Trajectory groups are also shuffled. The requested order is recorded; actual hardware dispatch order is not asserted. Each raw trajectory gets 256 shots; each of the four twirl variants gets 64 shots per factor, preserving 256 total per trajectory per factor.

### CZ twirling and folding

For a Pauli frame $P$, the post-frame is obtained by CZ conjugation, giving

$$
Q\,\mathrm{CZ}\,P=e^{i\phi}\mathrm{CZ}.
$$

This equivalence is tested for all 16 two-qubit Pauli frames. A selected CZ is replaced by three CZs, each individually twirled:

$$
\mathrm{CZ}^3=\mathrm{CZ}.
$$

Only native CZs are folded. This includes CZs used to compile bath exchange unitaries, whose ideal joint action remains unchanged. Reset channels, initial preparation, reservoir probabilities, physical simulation times and gamma are never folded or rescaled. Fold subsets are selected by a seed and nested across noise levels. Rounding is recorded through the **actual** CZ factors; distinct actual factors are required.

Folding/twirling occurs **after** ordinary target compilation. A restricted `Optimize1qGatesDecomposition` pass absorbs Pauli frames into neighboring one-qubit gates; no general transpiler or two-qubit optimization is run afterward. Pairwise barrier directives protect the explicit fold copies. Terminal-flip barriers prevent cancellation with preparation gates in the calibration circuits. Supported native instructions, gate counts, final measurement mapping and hashes are checked. New QPY loads also regenerate each variant from its raw template and compare circuit contents. Full small-system density-matrix tests verify output distributions through preparation, bath collisions, resets and final-mask decoding.

Pauli twirls add one-qubit pulses. Their cost is included in saved native metrics; they are not assumed to improve accuracy or leave duration unchanged. Runtime's own gate/measurement twirling is disabled to avoid applying a second, unrecorded randomization. DD is disabled. This is a client-side SamplerV2 workflow, not an Estimator `resilience_level` setting. [IBM's feature-compatibility documentation](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options) describes the dynamic-circuit restrictions.

The [offline compilation preview](inputs/mitigation_compilation_preview.json), using saved trajectory 0 and the saved backend target, gives the following unweighted depths across four variants:

| Circuit | CZ count | Native depth |
|---|---:|---:|
| Raw | 960 | 758 |
| Twirled, factor 1 | 960 | 795–807 |
| Twirled, factor 1.25 | 1200 | 987–1041 |
| Twirled, factor 1.5 | 1440 | 1170–1219 |

This is one offline representative, not measured execution duration or a reproduction of original submitted QPY. The restricted one-qubit pass avoids the much larger overhead of literal unmerged frames, while retaining every planned folded CZ and reset. [Preview script](../analysis/offline_audit/compile_mitigation_preview.py).

## Default workload and limits

| Quantity | Default |
|---|---:|
| Saved branch trajectories | 64 |
| Shots per trajectory per arm/factor | 256 |
| Explicit variants per twirled factor | 4 |
| CZ counts per twirled circuit | 960, 1200, 1440 |
| Bath resets per evolution circuit | 48 |
| Evolution circuits | 832 |
| Calibration/control circuits | 15 |
| Total circuits | **847** |
| Total shots | **77,824** |
| Sequential batches | **2**, sizes 504 and 343 |

The controls are all-zero, all-one and initial mixture under both raw and terminal-twirled readout treatments. Each preparation/treatment uses 2048 total shots: one raw PUB or four twirled PUBs of 512 shots each. The same twirled calibration is shared across noise factors.

The 64-path ideal profile itself has maximum physical-target population error **0.023795**. Therefore **this pilot cannot certify the physical 0.02 goal even with perfect hardware**. Its purpose is to compare methods against the exact ideal of the same fixed paths and to assess noise-scaling behavior. The configuration records both references and this limitation. A later physical steady-state evaluation would need an adequately sized ensemble and its own uncertainty budget; the pilot is not advertised as that evaluation.

## Predeclared analysis

The analysis retains raw, exploratory SPAM-corrected, twirled, corrected-twirled, uncorrected-ZNE and corrected-ZNE population estimates. All ten populations are analyzed. Figures show the spatial profiles and noise scaling for three preset sites (1, 5 and 9 at N=10).

ZNE uses one fixed **unweighted linear intercept**, not a data-selected fit. For default factors,

$$
\widehat n_{\rm ZNE}
=\frac{17}{6}\widehat n(1)
+\frac13\widehat n(1.25)
-\frac{13}{6}\widehat n(1.5).
$$

With equally precise independent scale estimates, this increases measurement SE by approximately **3.58 times** relative to one scale at that same shot count. This sampling cost is recorded, not hidden. The analysis saves all scale populations, fit residuals and intercept weights; a favorable intercept alone does not validate extrapolation. No monotonicity or linearity is assumed proven by three points.

The paired empirical-trajectory bootstrap is stratified by observed batches. Calibration PUBs are independently resampled at the joint bitstring level; the same calibration draw is shared across factors, preserving calibration correlations. Evolution shots are not independently resampled again on top of the empirical path bootstrap, which would double-count their variation. Fixed-path shot/calibration SEs are reported separately using the delta method. Intervals are descriptive, not simultaneous guarantees, and omit drift and calibration-model error. Small bootstrap groups can be poorly resolved.

CZ-focused ZNE does **not** extrapolate every hardware process uniformly. Reset errors remain unscaled, and elapsed-time/relaxation effects need not follow the CZ factor. Intentional reservoir dissipation and finite-Trotter bias must remain. Report both same-path ideal and physical-target errors, including unfavorable outcomes and estimates outside [0,1]. [IBM's mitigation discussion](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-mitigation) explains the noise-scaling assumptions and statistical overhead.

## Commands and lifecycle

An offline preview, with no account/backend access:

```bash
.venv/bin/python Model3/Experiment5/run_mitigation.py --plan-only
```

On the Windows hardware-execution machine, compile against the current backend without submitting:

```powershell
python Model3/Experiment5/run_mitigation.py --compile-only
```

To execute the compiled manifest, substitute the actual folder printed by that command:

```powershell
python Model3/Experiment5/run_mitigation.py --resume "Model3/Experiment5/hardware_runs/ibm_phoenix_mitigation_<timestamp>/manifest.json"
```

Alternatively, `python Model3/Experiment5/run_mitigation.py --execute` explicitly compiles and submits a fresh pilot. **These execution commands have not been run during development.** Parameters such as trajectory count, randomizations, shots, control shots and fold factors are configurable; shots must divide evenly among variants, with at least two shots per PUB. A trajectory's full treatment group must fit in one batch.

Both input hashes and deterministic plans are checked on resume. Job IDs are saved before collection. A known job is recovered rather than resubmitted; a submission interrupted before its ID was saved stops for reconciliation rather than automatically retrying. Fully collected resumes are local analysis only. Initial calibration snapshots are preserved; later snapshots are appended separately.

The runner saves per-batch native QPY, source/manifest hashes, software versions, explicit randomization seeds, terminal masks, actual noise factors, native metrics, calibration snapshots, job IDs and raw outcome-register counts. A small `results_bundle.zip` excludes QPY; retain those binaries locally and do not commit them to Git. Offline `--analyze-only <run-directory>` reconstructs the counts and writes separate population plots, CSV and JSON, explicitly recording unavailable native QPY.

## Validation

All **194 Model3 tests passed** after implementation and one-qubit frame optimization. The [validation record](inputs/mitigation_validation.json) includes the command, output and tested code hashes.

Tests use local targets, exact density matrices and mocked Runtime. They cover all Pauli frames, full native output equivalence with ignored preparation measurements and resets, terminal-mask decoding, nested fold subsets, actual CZ/reset counts, shared calibration uncertainty, fixed intercept weights, absence of reference fitting, offline planning, matched budgets, QPY tampering, portable analysis, interrupted collection, snapshot preservation and rejection of ambiguous submission retries. No IBM service was accessed during development.
