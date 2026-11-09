# Model2 mitigation + DD: minimum-job analysis

Backend `ibm_phoenix`; physical path `[106, 116, 115, 114, 113, 112, 111, 101, 91, 81, 80, 90]`; target captured 2026-11-09T00:16:11.268133+00:00.

**16,834 trajectories × 32 shots per arm/factor; 16,834 actual native DD-on paths counted. Projection: False.**

**published_limit: lower bound 19, best feasible 19 jobs; minimum certified for the full-cohort counting model.**
**selected_fraction: lower bound 24, best feasible 24 jobs; minimum certified for the full-cohort counting model.**

All thirteen matched variants stay together. Each job includes all fifteen zero/one/half-mixture control circuits. Gate twirling and folding precede the actual Experiment8 ALAP + ratio-10 XX passes. Final explicit delays, DD X gates, resets, measurements and implicit initialization are counted. Shots are summed separately.

| Packing | Feasible jobs | Lower bound | Full-cohort minimum certified | Circuits | Shots incl. controls | Peak instructions/qubit | Within counted limits |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| published_limit | 19 | 19 | True | 219,127 | 2,388,224 | 26,796,152 | True |
| balanced_runner_published_limit | 19 | 19 | True | 219,127 | 2,388,224 | 26,465,318 | True |
| selected_fraction | 24 | 24 | True | 219,202 | 2,449,664 | 21,439,915 | True |
| balanced_runner_selected_fraction | 24 | 24 | True | 219,202 | 2,449,664 | 20,974,163 | True |
| current_circuit_cap | 443 | 443 | True | 225,487 | 7,598,336 | 1,147,603 | True |

The explicitly checked balanced grouping corresponds to runner `--batch-size 9141` under the same target and cohort. This is a planning result. Actual production preparation still recompiles/validates every DD circuit; larger Sampler submissions also require more host memory.

The lower bound is the maximum of each physical qubit's total path footprint divided by its per-job allowance after reserving controls, and the analogous execution bound. A feasible partition attaining that integer lower bound proves the minimum for this counting model. If the two numbers differ, this script reports an interval, not a proven minimum. The 80% allowance is a separate planning choice; the configured 512-circuit comparison cap is not an IBM minimum.

A projection cannot establish the full-cohort minimum. CSV path_start/path_stop index the saved path_order, and trajectory_ids gives actual IDs. No existing hardware manifests, circuits, job IDs or controls are changed.

The cache freezes the target and compiler environment for reproducibility; resuming does not silently mix calibrations. Fetch a new target into a new output directory if required. Counts are not server-compiler acceptance or measured instruction-memory usage.

The estimator accepts 16,834 for accounting if requested, but the current hardware runner still caps new cohorts at 16,384; this script does not change that protocol limit.

[IBM job limits](https://quantum.cloud.ibm.com/docs/en/guides/job-limits). Machine-readable details: [job_estimate.json](job_estimate.json).
