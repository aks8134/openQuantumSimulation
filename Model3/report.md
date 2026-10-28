# Preparing Model3's nonequilibrium steady state with cosine-annealed Trotter steps

*Research report — October 28, 2026*

Preparing a dissipative steady state on a quantum processor creates a tension: the system must evolve long enough to forget its initial condition, but every additional Trotter step adds gates and resets that can damage the state. Model3 makes this tension measurable. Its boundary-driven XX chain has a known current-carrying stationary state, and its ideal dynamics can be checked independently. The question guiding this work is therefore **how much circuit evolution is actually needed to prepare the stationary observables, and how much of that ideal preparation survives on hardware?**

The evidence supports a clear progression. Coarse evolution followed by refinement provided a practical way to reduce preparation cost. A reference-assisted cosine schedule turns that observation into a reproducible design: a 72-step preparation passed the saved classical accuracy checks, and a warped cosine reduced the cost to **48 steps, 960 CZ gates and 48 resets**, with the same total tolerance of 0.02 under a revised allocation. Hardware measurements then showed that circuit cost and observed accuracy are separate optimization problems. Better mapping improved average population agreement, and twirling followed by zero-noise extrapolation reduced the pilot's largest discrepancy from its matching ideal prediction from **0.0739 to 0.0245**. Repeated preparation controls also improved overall and motivated calibration within each evolution job.

The central result is a resource-efficient first-order preparation with encouraging measured mitigation gains. The remaining test is whether those gains persist when evolution and its preparation controls are evaluated together in each job. All numerical evidence below comes from saved results; the appendices identify the supporting records.

## 1. A steady state with a known physical target

Model3 follows Section 4.9 of the modified [writeup](../writeup.pdf#page=11): an XX chain connected to reservoirs that favor different excitation populations. Using the writeup's jump-operator convention,

$$
\frac{d\rho}{dt}=-i[H,\rho]
+\sum_\alpha\left(V_\alpha\rho V_\alpha^\dagger
-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right),
\qquad
H=\frac J2\sum_{\ell=1}^{N-1}(X_\ell X_{\ell+1}+Y_\ell Y_{\ell+1}).
$$

Only the two endpoints dissipate:

$$
\begin{aligned}
V_{L,+}&=\sqrt{\Gamma p_L}\,\sigma_1^+,&
V_{L,-}&=\sqrt{\Gamma(1-p_L)}\,\sigma_1^-,\\
V_{R,+}&=\sqrt{\Gamma p_R}\,\sigma_N^+,&
V_{R,-}&=\sqrt{\Gamma(1-p_R)}\,\sigma_N^-.
\end{aligned}
$$

Here $\sigma^+=|1\rangle\langle0|$ pumps an excitation, $\sigma^-=|0\rangle\langle1|$ removes one, and the CLI parameter `gamma` is the total rate $\Gamma$ at **each** boundary. The preferred populations are $p_L=0.9$ and $p_R=0.1$. Their imbalance sustains transport even when the local populations stop changing.

We measure population and excitation current through

$$
n_\ell=\frac{I-Z_\ell}{2},\qquad
I_\ell=\frac J2(Y_\ell X_{\ell+1}-X_\ell Y_{\ell+1}).
$$

Positive excitation current flows left to right. The writeup's magnetization current is $j_\ell=-2I_\ell$; most saved numerical files use $I_\ell$.

For equal boundary rates, stationary continuity and the XX correlation equations give

$$
I_* = \frac{2J^2\Gamma}{\Gamma^2+4J^2}(p_L-p_R),\qquad
n_{1,*}=p_L-\frac{I_*}{\Gamma},\qquad
n_{N,*}=p_R+\frac{I_*}{\Gamma},
$$

with interior populations $(p_L+p_R)/2$. Two parameter sets appear in the work:

| Physical instance | Stationary populations | Excitation current on every bond |
|---|---|---:|
| Writeup benchmark: $N=3$, $J=1$, $\Gamma=2$ | $(0.7,0.5,0.3)$ | $0.4$ |
| Main preparation problem: $N=10$, $J=\Gamma=1$ | $(0.58,0.5,\ldots,0.5,0.42)$ | $0.32$ |

The different endpoint values follow from the rate choice. Increasing the length of this homogeneous XX chain preserves a flat interior profile; a gradual measured bulk slope cannot automatically be interpreted as the target transport behavior. The stationary state also carries coherences, so its local populations alone do not specify a product state.

Full-spin stationary solves and independent covariance calculations reproduce the benchmark. This gives the later optimization a fixed physical target: the target is not adjusted to accommodate a schedule or a hardware output. The main preparation begins from

$$
\rho_0=\left(\frac I2\right)^{\otimes N},\qquad C(0)=\frac I2.
$$

On hardware, H followed by an unconditioned Z measurement prepares each local half-mixture; the preparation outcomes are ignored. The physical stationary target is independent of the initial preparation for the cases studied, but the finite preparation schedule is not. The chosen half-mixture is therefore part of the problem specification, not an interchangeable initialization.

## 2. Turn the accuracy question into a circuit-cost question

The first task was to decide what “reaching the steady state” should mean operationally. A curve that appears flat may have reached the stationary state of an inaccurate finite-step channel. Conversely, insisting on accurate reproduction of every transient point can require a much deeper circuit than preparing one useful final state.

Our objective is an output whose populations and currents are close to their stationary values. The classical metric is

$$
d(C,C_*)=\max\left\{\max_\ell|n_\ell-n_{\ell,*}|,
\max_\ell\frac{|I_\ell-I_{\ell,*}|}{|J|}\right\}.
$$

It checks **all ten sites and all nine bonds**. The tolerance 0.02 is an absolute local-observable tolerance. It is not a many-body state-distance claim. Figures display only three representative sites or bonds to keep the observable values legible.

The XX covariance reference allows physical relaxation and numerical bias to be separated. For comparison, a saved exact half-mixture run reached a conservative future local-error bound below 0.02 at physical time **45.5**, with actual local error **0.00279**. That is a bound crossing on its observation grid, not a minimum settling time. An optimized finite-step preparation can approach the desired output without closely following the exact transient along the way.

The circuit makes the cost of that choice concrete. Each step applies two disjoint Hamiltonian bond layers and one randomly selected boundary collision. Rate-weighted selection uses

$$
w_\alpha=\frac{\lambda_\alpha}{2\Gamma},\qquad
\boldsymbol w=(0.45,0.05,0.05,0.45),
$$

and probability compensation gives the bath interaction angle

$$
\theta_{\alpha,k}=\sqrt{\frac{\lambda_\alpha\Delta t_k}{w_\alpha}}
=\sqrt{2\Gamma\Delta t_k}.
$$

The native logical factors are a system `XXPlusYYGate` with parameter $2J\Delta t_k$ and a selected bath exchange with parameter $2\theta_{\alpha,k}$. An empty bath produces loss; X on the reset bath before exchange produces pumping. Separate left/right baths keep endpoint interactions local, while only one bath is active in a given step. The [circuit derivation](circuit_derivation.md) proves the equivalence to the writeup's empty-ancilla pump/loss formulation.

For ten sites, nine system exchanges and one bath exchange compile to **20 CZ gates per step**, followed by one bath reset. Thus reducing preparation steps directly reduces its largest repeated circuit costs. Increasing trajectory count or shots changes total workload, but does not shorten an individual circuit.

The finite-step collision is a genuine approximation: its transfer probability is $\sin^2\sqrt{2\Gamma\Delta t}$, rather than an exact reservoir exponential, and its coherence factor retains the signed cosine. Very coarse steps can therefore be valid quantum channels while having substantial physical bias. This explains why reducing the late timestep matters.

The manual data demonstrate the principle without requiring an interpretation based only on visual flatness. Three schedules share exactly the same coarse history, step 2 through physical time 30, and differ afterward:

| Step after time 30 | Total steps to time 60 | Final population bias vs exact transient | Final current bias / J |
|---:|---:|---:|---:|
| 2 | 30 | 0.119655 | 0.461831 |
| 1 | 45 | 0.147608 | 0.214004 |
| 0.1 | 315 | 0.028613 | 0.019418 |

Refinement greatly improves the final state, but also raises circuit cost. The nonmonotone population response to the intermediate choice shows why “use a larger early step” is not a complete scheduling rule. The state reached before cleanup matters as well. [Source and actual-observable figures](Experiment3/analysis/ExistingRunsAnalysis.md).

This led to an explicit budget: choose the schedule using the deterministic mean channel, then test independent finite ensembles. Most of the later work allocates 0.016 to the mean output/forward check and 0.004 to sampling, totaling 0.02. An early controller that repeatedly halved the step established convergence but required **3,530 steps** at a tighter 0.01 target; its 64 trajectories still failed the sampling allowance. Refinement could fix bias, but could not fix insufficient sampling.

The useful change was to optimize the **preparation length** and move the stability check outside the executable preparation. After the proposed output, the mean channel is continued classically for at least $10/J$ at the same final timestep. Every checked node must meet the mean allowance. Sampling is checked at the prepared output. These continuation nodes add no hardware gates. This protocol found a 117-step piecewise preparation with forward mean error **0.0159223** and passing three-seed replays. Earlier 490-step results used a sampled trailing window, so their reduction to 117 combines a changed acceptance rule with better schedules. They are not equal-protocol depth benchmarks.

The question was now precise enough for automatic optimization: **find a short schedule that prepares an acceptable output and remains acceptable under the declared continuation, then spend a separate budget on sampling.**

## 3. Cosine annealing makes the preparation short enough to test

A useful schedule needs coarse evolution for economical preparation and refinement before the output. Searching a small family captures that structure more efficiently than making a locally attractive timestep decision at every stage. A state-feedback controller achieved a passing 273-step preparation, while a jointly optimized two-stage schedule needed 73 steps. Cosine annealing provided a smooth alternative with only a few global parameters.

We write both cosine variants as

$$
\Delta t_k=\Delta t_{\min}
+\frac{\Delta t_{\max}-\Delta t_{\min}}2
\left[1+\cos\left(\pi\left(\frac{k}{M-1}\right)^q\right)\right],
\qquad k=0,\ldots,M-1.
$$

The ordinary cosine has $q=1$. The planner selects the length $M$ and timestep parameters using the deterministic mean covariance, with reference-assisted error checks. Sampling seeds are applied after selecting the schedule. The resemblance to learning-rate annealing is the decreasing step profile; this calculation does not use stochastic gradients or noisy hardware feedback to choose its steps.

The first selected cosine used **72 steps**. Allowing the exponent $q$ to vary later produced a **48-step warped cosine**:

| Schedule parameter | Ordinary cosine | Warped cosine |
|---|---:|---:|
| Preparation steps $M$ | 72 | 48 |
| $\Delta t_{\max}$ | 0.5762677230 | 1.2288353376 |
| $\Delta t_{\min}$ | 0.05733068986 | 0.02282076280 |
| Exponent $q$ | 1 | 0.3469734992 |
| Total simulated preparation time | 22.80954286 | 13.35252757 |

With $q<1$, the decrease is front-loaded: the warped schedule spends fewer steps near its large initial value and reaches refinement earlier. A starting step larger than 1 was therefore feasible in the saved preparation study. It is accepted because the complete output and continuation pass the chosen tests, not because such a step accurately reproduces the early continuous-time trajectory.

![Observable evolution under the ordinary cosine](analysis/variable_schedule/cosine_observables.png)

*Figure 1. Three actual populations and three excitation currents for the 72-step cosine, $R=8192$, seed 29. The sampled mean is blue, the deterministic mean dotted orange, exact continuous evolution dashed black, and the physical stationary target green. The plot uses zero-based indices. The endpoint becomes accurate even though the coarse portion visibly changes the transient.*

The 48-step schedule required an explicit reallocation of the same total tolerance: **0.0165 mean + 0.0035 sampling**, instead of 0.016 + 0.004. Its mean error slightly exceeds the original allowance. That change is part of the result and is accounted for by a larger classical sampling ensemble.

| Validation quantity | 72-step cosine | 48-step warped cosine |
|---|---:|---:|
| Mean output error | 0.01592539 | 0.01627869 |
| Worst forward mean error | 0.01592539 | 0.01635805 |
| Stationary bias of the final-step channel | 0.00909086 | 0.00363862 |
| Mean / sampling allowance | 0.016 / 0.004 | 0.0165 / 0.0035 |
| Classical trajectories per seed | 8192 | 16384 |
| Sampled output errors, seeds 29–31 | 0.017110, 0.017428, 0.016858 | 0.017537, 0.016828, 0.017674 |

All three replays of each schedule passed their configured checks. The initial mixture is averaged analytically in those calculations, and measurement/device noise is excluded. The larger $R$ improves ensemble precision; it does not increase per-circuit depth. Three seeds and descriptive two-SE checks also do not constitute simultaneous confidence guarantees. The bounded search did not prove a global minimum: its best tested 40-step candidate failed, but untested schedules remain possible.

The important outcome is the circuit saving under the stated acceptance rules:

| Preparation | CZ gates | Bath resets | Two-qubit depth | Full native depth |
|---|---:|---:|---:|---:|
| 117-step piecewise schedule, FakeKingston | 2340 | 117 | 468 | 1882–1887 |
| 72-step cosine, FakeKingston | 1440 | 72 | 288 | 1125–1131 |
| 48-step warped cosine, saved-target representative | 960 | 48 | 192 | 758 |

The 72-step cosine reduces step/CZ count by **38.5%** relative to the 117-step preparation under the same forward-validation framework. The warped cosine removes another **one-third** of the steps, CZs and resets, with the revised allocation disclosed above. Different compilation targets affect exact depths, so the strongest common resource comparison is the explicit gate/reset count.

![Schedule compression, actual final observables and cost](analysis/hardware_cost_redesign/comparison.png)

*Figure 2. The warped cosine front-loads refinement while retaining mean-channel populations and currents near their physical targets. The saved-target representative changes from depth 1126 to 758 and modeled duration 115.040 to 76.228 microseconds. These durations exclude unknown dynamic-block latency and are not measured hardware elapsed times.*

The result answers the original depth concern: steady-state preparation does not require thousands of uniformly accurate transient steps for this initial state. Cosine scheduling supplies a short, independently checked candidate. The next question is whether making that candidate cheaper also makes its measured output better. [72-step verification](Experiment7/runs/comparison_d3fd93cb452e/VerifiedComparison.md), [48-step redesign and validation](analysis/hardware_cost_redesign/Report.md).

## 4. A cheaper ideal circuit exposes a hardware problem

A hardware output has to be compared with the ideal circuit actually sampled. Let $O_{\mathrm{ideal},R}$ be the ideal observable for the same saved branch paths, and $O_{\mathrm{mean}}$ the deterministic branch average. Then

$$
O_{\mathrm{hw}}-O_*
=(O_{\mathrm{hw}}-O_{\mathrm{ideal},R})
+(O_{\mathrm{ideal},R}-O_{\mathrm{mean}})
+(O_{\mathrm{mean}}-O_*).
$$

This decomposition separates device/measurement discrepancy, finite-path sampling and the ideal preparation error. Each term is an observable difference; their maximum norms cannot simply be added as measured scalar contributions.

The distinction mattered immediately. Four trajectories with many shots gave a precise estimate of an unrepresentative path ensemble. Increasing to 8192 trajectories with one shot per measurement setting made the ideal ensemble much closer to the target, while a large hardware gap remained:

| 72-step hardware evaluation | Four paths × 8192 shots | 8192 paths × one shot |
|---|---:|---:|
| Same-path ideal maximum population error to target | 0.050738 | 0.017110 |
| Hardware maximum population error to target | 0.062225 | 0.082520 |
| Spatial mean same-path ideal current | 0.278798 | 0.309521 |
| Spatial mean hardware current | 0.097446 | 0.104045 |

The population discrepancy is important in its own right; the hardware conclusion is not based only on suppressed currents. More trajectories fixed a weakness in the ideal reference, but did not remove hardware bias. These were different execution occasions, so the table does not show that using one shot caused poorer fidelity. Saved-count subsampling likewise showed that more shots concentrate estimates around the existing hardware population profile, rather than moving their expectation toward ideal. [Population and current analysis](Experiment5/analysis/hardware/ibm_phoenix_20261021T102532_376654Z/Analysis.md), [shot-subsampling evidence](Experiment5/analysis/hardware/ibm_phoenix_20261020T190507_427277Z/population_subsampling/Analysis.md).

Before changing the physical model, the numerical and circuit pipeline was audited. Independent full-spin checks, actual-input noiseless Aer density calculations, path regeneration, bit decoding, current parity, source hashes and raw-count coverage found no model or logical-circuit defect explaining the mismatch. Actual N=10 density predictions agreed with covariance values at approximately $10^{-15}$. This supports treating the remaining difference as an implementation/noise problem within the limits of the available compiled artifacts. It does not provide a calibrated hardware noise model. [Offline audit](analysis/offline_audit/Report.md).

Three pieces of hardware evidence then clarified the direction of improvement.

**Reducing depth saves resources, but is not sufficient by itself.** A paired 1024-path comparison on the same chain reduced CZs/resets by one-third. The maximum physical population error improved from **0.054199 to 0.042603**, while mean discrepancy from each schedule's own ideal changed from **0.020436 to 0.023989**. Paired aggregate intervals included zero. The result therefore establishes the cost saving and a mixed accuracy outcome, rather than a universal fidelity gain from shortening the schedule. [Paired schedule comparison](Experiment5/analysis/hardware/ibm_phoenix_comparison_20261026T005025_553744Z/schedule_comparison/Findings.md).

**The physical mapping changes the measured state materially.** An initial chain contained q7 with short coherence times and poor neighboring CZ calibration. Moving to a calibration-selected chain improved the 128-path population MAE to the physical target from **0.053165 to 0.023502**, although execution date also changed. A subsequent paired comparison isolated mapping more directly by fixing the 48-step schedule, all 1024 branch paths, and 32 shots per path:

| Paired population comparison | Previous mapping | Alternative mapping |
|---|---:|---:|
| MAE to the common same-path ideal | 0.023005 | **0.017388** |
| Maximum same-path discrepancy | 0.047356 | 0.054126 |
| MAE to physical target | 0.019232 | **0.015747** |
| Maximum physical-target error | 0.046143 | **0.036316** |

Average agreement with the ideal improved **24.4%**, with a descriptive paired 95% interval for the MAE change of **[-0.007754, -0.002846]**. Some sites worsened, and the maximum ideal discrepancy was not improved. Mapping is thus a demonstrated contributor, while logical role and circuit context still matter. The alternative chain was retained for mitigation.

![Mapping comparison at identical ideal branch paths](Experiment5/analysis/hardware/ibm_phoenix_mapping_comparison_20261026T050356_310922Z/mapping_comparison/paired_mapping_populations.png)

*Figure 3. The common ideal is held fixed while the physical mapping changes. Several measured populations improve markedly; others move away from the ideal. The data support the average-agreement gain without identifying a single qubit as the complete explanation. [Paired mapping evidence](Experiment5/analysis/hardware/ibm_phoenix_mapping_comparison_20261026T050356_310922Z/mapping_comparison/Findings.md).*

**Preparation and elapsed hardware time contribute before and alongside exchange-gate errors.** Prefix measurements showed population offsets even at zero steps, with average same-prefix discrepancy growing in longer prefixes. Isolated boundary tests also showed losses with no CZ gates. Adding idle controls clarified that much of the apparent reset-associated loss could occur during waiting alone: after 31.68 microseconds of modeled wait, initially-one populations fell from **0.9824 to 0.8521** on q49 and **0.9946 to 0.8691** on q95. Reset sequences gave comparable losses, and none of 24 reset-minus-idle tests survived the stated multiplicity correction. This does not prove resets harmless; it prevents attributing all loss to reset infidelity. Model-time matching also omits unknown dynamic reset latency. [Prefix evidence](Experiment5/analysis/hardware/ibm_phoenix_prefix_20261024T162033_596311Z/Findings.md), [waiting/reset comparison](Experiment5/analysis/hardware/ibm_phoenix_reset_survival_20261025T224328_184810Z/reset_survival/Findings.md).

Taken together, these results explain why schedule optimization and mapping help but do not close the gap. The ideal preparation is already accurate, and the chain accumulates additional device and measurement effects. The appropriate next change is to improve or compensate the implementation while holding the first-order physical schedule fixed.

## 5. Mitigation improves the measured state

The mitigation pilot preserved the 48-step warped cosine, initial mixture and saved branch choices. This isolates a different question from timestep search: **can the same ideal state be estimated more accurately by changing how its native circuit is randomized and measured?**

The pilot evaluated 64 identical branch paths in every treatment, using 256 shots per path per arm/noise factor. One arm retained the raw circuit. Three further arms used explicit CZ Pauli twirling and terminal-measurement twirling at CZ factors **1, 1.25 and 1.5**, with four randomizations per factor. Selective folding replaces a CZ by three CZs, preserving its ideal action. The corresponding circuits contain **960, 1200 and 1440 CZs**, while retaining **48 bath resets**. The extra one-qubit twirl pulses and larger depths are included in the resource records.

The physical bath evolution is kept intact. In particular, reset channels are not folded and reservoir rates are not rescaled. This makes the mitigation compatible with the intended open-system model, but means the extrapolation does not scale every hardware process uniformly.

For these noise factors, the fixed unweighted linear zero-noise intercept is

$$
\widehat n_{\mathrm{ZNE}}
=\frac{17}{6}\widehat n(1)
+\frac13\widehat n(1.25)
-\frac{13}{6}\widehat n(1.5).
$$

No ideal or physical target is used to fit the result. Because the coefficients amplify uncertainty, success has to be judged from both observable values and precision. The 847-circuit, 77,824-shot pilot gave:

| Population estimator | MAE to same-path ideal | Maximum same-path discrepancy | Maximum physical-target error |
|---|---:|---:|---:|
| Raw | 0.021425 | 0.073905 | 0.050110 |
| Twirled | 0.020160 | 0.042411 | 0.039324 |
| ZNE | **0.013613** | **0.024462** | **0.025543** |
| ZNE with exploratory SPAM correction | **0.013053** | **0.023918** | **0.024581** |

The largest same-path discrepancy fell by approximately **43% with twirling** and **67% with ZNE** relative to raw. This is meaningful progress. It also appears directly in the populations: at site 9 the measured/extrapolated values move from **0.4499 → 0.4814 → 0.5093**, for raw, twirled and ZNE, respectively, toward the same-path ideal **0.5238** and physical target **0.5**.

![Population values before and after mitigation](Experiment5/analysis/hardware/ibm_phoenix_mitigation_20261026T184843_413808Z/mitigation_review/selected_populations.png)

*Figure 4. Mitigation changes the actual population estimates. The green band shows ±1 sample standard deviation across the 64 ideal trajectories; it describes their spread, not uncertainty of the ideal mean. The orange dash-dot line is the deterministic infinite-trajectory mean-channel output. Blue bars show ±1.96 conditional shot SE for the fixed paths. A [companion plot](Experiment5/analysis/hardware/ibm_phoenix_mitigation_20261026T184843_413808Z/mitigation_review/selected_populations_mean_uncertainty.png) shows the ideal mean with ±1.96 SE instead, using SE = SD/√64. [All-site ideal uncertainty values](Experiment5/analysis/hardware/ibm_phoenix_mitigation_20261026T184843_413808Z/mitigation_review/ideal_path_uncertainty.csv) distinguish the two quantities. The numerical accuracy comparisons still use all ten sites and matched ideal paths.*

The strongest statistical evidence is for twirling's reduction of the worst discrepancy. ZNE improves the point estimates further, but its incremental gain over twirling has paired intervals that include no improvement. With equally precise independent scale estimates, its shot-SE amplification is approximately **3.58**, giving per-site SE around **0.0135–0.0140** in this pilot. The favorable intercept therefore motivates confirmation with calibration and uncertainty tracked together.

Two distinctions keep the result interpretable. First, this is a population-only measurement: later Z-only records do not contain the X/Y correlations required for excitation currents. Second, the ideal 64-path subset already differs from the physical target by **0.023795** in its worst population. It is a useful fixed-path mitigation test but cannot inherit the large classical ensemble's 0.02 validation automatically. The measured mitigation gain and a full steady-state accuracy claim are different conclusions. [All site values, paired uncertainty and full correction table](Experiment5/analysis/hardware/ibm_phoenix_mitigation_20261026T184843_413808Z/mitigation_review/Findings.md).

## 6. Preparation controls make that improvement interpretable

The remaining calibration question follows directly from the mitigation result. A zero-noise estimate can look closer to the ideal while a preparation/readout offset remains. To test that offset, the workflow measures all-zero and all-one controls and holds the half-mixture out as a consistency check.

If the local zero/one responses are $a$ and $b$, a stable affine-response model predicts a half-mixture measurement of $(a+b)/2$ and suggests

$$
\widehat n_{\mathrm{corr}}=\frac{n_{\mathrm{observed}}-a}{b-a}.
$$

These controls include preparation and initialization errors, so this is an exploratory SPAM correction. The mixture tests its assumption rather than being used to fit it. Poor contrast is rejected and estimates are not clipped. Earlier SPAM-only postprocessing did not consistently improve the data, which is why raw and corrected estimates are retained side by side.

In the mitigation pilot, physical q48 had mixture populations **0.4150 raw** and **0.4375 twirled**, substantially below its zero/one predictions. But the zero/one controls and the mixture were in different jobs whose start times were about 97 minutes apart. That left an avoidable ambiguity: the calibration might not describe the conditions of the measured output.

Repeating complete zero/one/mixture controls twice inside each of two jobs addressed that ambiguity. The overall result was **better preparation/readout agreement**, not a general failure of the controls:

| Corrected-mixture error relative to 0.5 | Earlier pilot controls | New average across both jobs | Reduction |
|---|---:|---:|---:|
| Mean absolute error, raw readout | 0.017328 | **0.009543** | **44.9%** |
| Mean absolute error, terminal-twirled readout | 0.012540 | **0.009574** | **23.7%** |

These descriptive comparisons cover all ten sites. They span different execution occasions and therefore do not isolate a single causal improvement. They do show that the earlier large q48 deficit was not persistent:

| Physical qubit | Job | Corrected raw mixture | Corrected twirled mixture |
|---|---:|---:|---:|
| 48 | 1 | 0.4991 | 0.4880 |
| 48 | 2 | 0.4970 | 0.4995 |
| 76 | 1 | 0.4854 | 0.4899 |
| 76 | 2 | 0.4778 | **0.4422** |

The specific remaining problem is q76 under twirled readout in job 2. Both rounds and all eight mixture variants were low, including variants with and without a terminal X on q76. Its pooled corrected estimate, **0.442197**, had conditional interval **[0.426461, 0.457934]**; the discrepancy survived the stated adjustment over all 120 checks. The two rounds and pooled flag describe the same repeated issue, not three independent events.

![Preparation consistency across jobs and rounds](Experiment5/analysis/hardware/ibm_phoenix_repeated_preparation_20261027T043444_434996Z/preparation_review/selected_mixture_controls.png)

*Figure 5. Qubit 48's earlier large deficit disappears, while q76 shows a larger deficit in the second job. These short controls contain no CZ gates or bath resets. They demonstrate why preparation/readout consistency should be evaluated alongside the evolved state, even after overall control accuracy improves.*

Q76's saved T1 also changed from **203.4 to 52.3 microseconds** between the pilot and repeated controls. This identifies a concrete coherence weakness in the retained mapping. Identical q76 calibration in the two new snapshots cannot, by itself, explain their different outcomes. Preparation context, intermediate-measurement timing, other flipped qubits and temporal variation remain possible contributors.

The consequence is a change in **measurement design**. The improved controls support proceeding with the evolved state, while the remaining issue motivates complete controls in every evolution job and separate calibration before combining job estimates. This allows the next comparison to evaluate mitigation and preparation consistency together. [Repeated-control evidence](Experiment5/analysis/hardware/ibm_phoenix_repeated_preparation_20261027T043444_434996Z/preparation_review/Findings.md).

## 7. The result so far and the next test

The work has reduced two different obstacles. Reference-assisted cosine scheduling addresses the cost of the **ideal preparation**: the current candidate uses 48 first-order steps and one-third fewer CZs/resets than the validated 72-step cosine. Mapping and mitigation address the **measured implementation**: a paired layout change improved average agreement, and the mitigation pilot moved the population profile substantially closer to the ideal. Preparation controls improved as well, and revealed how to compare results without relying on calibration borrowed across jobs.

These gains should be assessed together. Shortening the schedule does not guarantee a fidelity gain; better average fidelity does not guarantee every site passes; and a favorable extrapolated point estimate has to be weighed against its uncertainty. The available evidence supports progress on each of these fronts, while a hardware demonstration of 0.02 accuracy over every population and current remains open.

The agreed next evaluation holds the physical model, half-mixture, first-order 48-step schedule, mapping and branch paths fixed. It compares raw, twirled and ZNE populations with complete preparation controls **inside each evolution job**. The implemented plan uses 64 matched paths, 256 shots per path per arm/noise factor, and four randomizations in each twirled arm. Two balanced jobs each contain 32 paths and 15 control circuits: **862 circuits and 90,112 shots total**.

Each job's exploratory SPAM inversion is applied using its own zero/one measurements before trajectory-weighted averaging. The half-mixture remains held out. Uncorrected estimates, per-job controls and calibration uncertainty remain visible, so a correction cannot conceal a failed assumption. The explicit `record-only` calibration policy retains the current chain for this paired comparison, including q76, and records its available calibration rather than rejecting it at the previous T1 cutoff. Connectivity and native-circuit checks remain active.

**The implementation and offline plan are complete, but hardware results for this combined evaluation are not present in the checkout as of October 28.** The [execution protocol](Experiment5/Mitigation.md) contains the command and resumption instructions. The central question is whether the pilot's mitigation gains remain visible in the evolved populations under the new same-job control design. Another standalone control campaign is not needed to pose that test.

## Appendix A. What the ideal reference and validation establish

For the noninteracting XX model, the one-body covariance obeys

$$
\dot C=AC+CA^\dagger+P,\qquad A=-ih-\frac G2,
$$

where $h_{\ell,\ell+1}=J$, $G=\operatorname{diag}(\Gamma,0,\ldots,0,\Gamma)$ and $P=\operatorname{diag}(\Gamma p_L,0,\ldots,0,\Gamma p_R)$. Thus

$$
AC_*+C_*A^\dagger+P=0,\qquad
C(t)=C_*+e^{tA}(C(0)-C_*)e^{tA^\dagger}.
$$

This is an exact reference for these XX observables, not a reconstruction of the full many-body density matrix. The efficient covariance studies do not silently apply the same closure to interacting XXZ dynamics. Full-state small-system calculations provide an independent check of the reduction and current conventions.

The primary schedule checks use an output test plus direct mean continuation at the same final step for at least $10/J$. Continuation work is classical and adds no preparation gates. A later audit extended the mean check and bounded its remaining constant-step tail. This strengthens the discrete-node ideal mean result, without establishing between-node, noisy-hardware or full-state accuracy. The final-step stationary biases are already included in the schedule analysis.

Different initial states retain the same physical stationary target but need different preparation schedules. The later audit found that substituting all-zero/all-one preparations into the fixed cosine lists produced output errors around 0.1122 for 72 steps and 0.1636 for 48 steps. This is why the half-mixture specification must accompany the advertised step count.

Historical validation milestones include 183 Model3 tests in the consolidated audit, 194 after initial mitigation implementation, and 18 targeted mitigation/repeated-control tests after adding job-local calibration. Some transferred runs omit native QPY by design; recorded execution-machine verification is distinguished from fresh binary inspection. Proven Windows CRLF/Linux LF changes are distinguished from corruption using original archived hashes. No new simulations, transpilation or hardware jobs were launched to rewrite this report.

## Appendix B. Evidence and implementation map

The linked records retain full settings, arrays, uncertainty definitions and intermediate searches for the evidence used above.

| Question or supporting work | Evidence and scope |
|---|---|
| Does the physical target agree with the writeup? | [Benchmark and controls](Experiment1/Experiment1.md); full Lindblad/covariance agreement, reservoir reversal and equilibrium checks |
| What steps reproduce strict transient populations? | [Population-step limits](Experiment1/analysis/PopulationStepLimits.md); earlier shared-ancilla protocol, a different criterion from output preparation |
| How do size and rate affect workload? | [Scaling study](Experiment2/Experiment2.md); the two [saved](Experiment2/study/scaling_20cba10326e7/Study.md) [checkpoints](Experiment2/study/scaling_78179558adfd/Study.md) remain marked `running`, so the full planned sweep is not a completed result |
| Does manual refinement help? | [Saved manual comparisons](Experiment3/analysis/ExistingRunsAnalysis.md); [observable-value figure](analysis/variable_schedule/manual_observables_left_excitation.png) |
| How were the output and forward-check rules developed? | [490-step result](Experiment4/analysis/Tolerance002Run.md); [117-step forward study](Experiment4/analysis/ForwardWindowStudy.md); acceptance rules are explicitly distinguished |
| Can mean-state feedback choose steps? | [273-step controller](Experiment6/runs/adaptive_6ea9f15a44d8/Study.md); an earlier configuration reached its cap, demonstrating the limits of local refinement |
| Why choose the cosine family? | [Verified two-stage/cosine comparison](Experiment7/runs/comparison_d3fd93cb452e/VerifiedComparison.md); 73 versus 72 steps, with independent saved replays |
| How was 48 obtained? | [Bounded redesign search and validation](analysis/hardware_cost_redesign/Report.md); warped cosine and revised mean/sampling allocation |
| What is implemented in the circuit? | [Mathematical derivation](circuit_derivation.md); [logical/native construction](Experiment5/circuit_depth.py); [shared covariance channel](../library/reservoir_scaling.py) |
| Are ideal predictions and measurement decoding consistent? | [Offline audit](analysis/offline_audit/Report.md); [independent noiseless Aer check](Experiment5/analysis/hardware/ibm_phoenix_20261020T190507_427277Z/aer_population_check/Analysis.md) |
| What motivated preparation/timing controls? | [Fresh preparation findings](Experiment5/analysis/hardware/ibm_phoenix_20261025T150619_925833Z/preparation/Findings.md); [isolated boundary findings](Experiment5/analysis/hardware/ibm_phoenix_boundary_20261025T215447_981524Z/boundary/Findings.md); [idle/reset comparison](Experiment5/analysis/hardware/ibm_phoenix_reset_survival_20261025T224328_184810Z/reset_survival/Findings.md) |
| What should be executed and analyzed next? | [Mitigation protocol with same-job controls](Experiment5/Mitigation.md); [standalone-control protocol](Experiment5/RepeatedPreparationControls.md) documents the preceding completed diagnostic |

The physical convention throughout the main preparation work is the exchange prefactor $1/2$. Model2's documented default is $1/4$, so comparisons require that normalization to be matched. The randomized Model3 covariance method also does not assume Model1's single-excitation restriction; boundary pumping populates multiple excitation sectors.

## Appendix C. Saved hardware records


All rows below are marked `collected` in their saved manifests. Folder timestamps are UTC run identifiers, not a claim about local calendar dates or per-circuit durations. Shot totals include controls when present. These are the **13 available run folders**, not an inventory of unpulled or external jobs.

| Run-folder suffix after `ibm_phoenix_` | Purpose | Jobs | Circuits | Observed shots |
|---|---|---:|---:|---:|
| `20261020T190507_427277Z` | Four-path, 72-step populations and currents | 1 | 12 | 98,304 |
| `20261021T102532_376654Z` | 8192 paths, one shot, three settings | 48 | 24,576 | 24,576 |
| `20261023T135506_304864Z` | Original-chain populations plus controls | 1 | 131 | 9,728 |
| `20261023T192723_098429Z` | Selected-chain preparation controls | 1 | 3 | 6,144 |
| `20261024T042722_287349Z` | Selected-chain 128-path populations | 1 | 128 | 8,192 |
| `prefix_20261024T162033_596311Z` | Prefix progression | 1 | 160 | 20,480 |
| `20261025T150619_925833Z` | Fresh preparation controls | 1 | 3 | 12,288 |
| `boundary_20261025T215447_981524Z` | Isolated collisions/reset controls | 1 | 20 | 40,960 |
| `reset_survival_20261025T224328_184810Z` | Idle/reset duration comparison | 1 | 64 | 131,072 |
| `comparison_20261026T005025_553744Z` | Paired 72/48-step schedules | 5 | 2,051 | 22,528 |
| `mapping_comparison_20261026T050356_310922Z` | Paired mappings, same 48-step inputs | 5 | 2,054 | 77,824 |
| `mitigation_20261026T184843_413808Z` | Raw/twirled/ZNE pilot | 2 | 847 | 77,824 |
| `repeated_preparation_20261027T043444_434996Z` | Two-job repeated controls | 2 | 60 | 49,152 |
| **Total in these folders** | | **70** | **30,109** | **579,072** |

Raw manifests/counts and transfer bundles are under [Experiment5/hardware_runs](Experiment5/hardware_runs/). Derived figures, tables and reviews are under [Experiment5/analysis/hardware](Experiment5/analysis/hardware/). Source-specific caveats, uncertainty definitions and native-binary availability remain recorded in the linked detailed reports.

