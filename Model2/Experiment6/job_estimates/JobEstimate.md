# Model2 full mitigation job estimate

**16,384 trajectories × 32 shots per arm/factor.** Four variants per twirled factor, three factors (1, 1.25, 1.5), paired raw arm. Controls: 2,048 shots per preparation/treatment/job.

Source: `saved_fake_kingston`, backend `fake_kingston`, **32 counted paths**. Footprint expansion: **True**. The bundled sample is FakeKingston; it is not a full Phoenix archive audit.

Instruction counts use [IBM's published heuristic](https://quantum.cloud.ibm.com/docs/en/guides/job-limits). Shots are summed separately; complete controls repeat in every job. All matched variants for a path stay together, with at least two paths per job.

| Scenario | Jobs | Counting lower bound | Circuits | Shots including controls | Peak instructions/qubit | Within counted limits |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No circuit cap, published instruction allowance | 16 | 16 | 213,232 | 2,293,760 | 26,798,982 | True |
| No circuit cap, 80% instruction allowance | 20 | 20 | 213,292 | 2,342,912 | 21,439,658 | True |
| Observed maximum at every position, 80% | 21 | 21 | 213,307 | 2,355,200 | 21,413,790 | True |
| Current runner cap: 512 circuits/job | 432 | 432 | 219,472 | 7,405,568 | 986,457 | True |

The first two scenarios construct a contiguous greedy partition in the saved-seed path order. Equality with the lower bound establishes the minimum only for the supplied footprint/counting model. The observed-maximum scenario is a sensitivity calculation, not a bound on unseen Phoenix circuits. CSV path_start/path_stop are positions in the documented permutation, not physical trajectory IDs.

The circuit-cap row reproduces the runner's balanced path grouping. If its limits check is false, the current runner would refuse submission; the number of jobs in that row is not a feasible hardware plan.

**Planning only:** memory-packed partitions are not installed in the hardware runner. Neither the estimator nor these CSV files change its default 64-path pilot, 512-circuit cap, job IDs, source, QPY or manifest. A smaller projected job count does not imply that the full run is ready to submit.

This does not estimate QPU seconds, dollars, compiler acceptance or statistical accuracy. All input archives remain read-only. Source hashes, weights, per-job shot/control totals and assumptions are in job_estimate.json.
