# Review of full Phoenix mitigation + DD job estimate

**The full-count report supports 19 jobs at the published instruction allowance, or 24 jobs with the selected 80% allowance.** All 16,834 requested trajectories were counted according to the saved estimate; footprint projection is false.

| Plan | Jobs | Circuits | Shots including controls | Peak instructions/qubit | Headroom to hard allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced published allowance | 19 | 219,127 | 2,388,224 | 26,465,318 | 1.249% |
| Balanced 80% allowance (recommended) | 24 | 219,202 | 2,449,664 | 20,974,163 | 21.738% |
| Existing 512-circuit cap | 443 | 225,487 | 7,598,336 | 1,147,603 | 95.718% |

The balanced 24-job partition has 701 or 702 paths per job, at most 9,141 circuits/job and 102,144 shots/job. Its peak is 20,974,163 instructions/qubit: 465,837 below the selected 21,440,000 allowance and 5,825,837 below the hard 26,800,000 allowance. The corresponding runner setting is `--batch-size 9141`, conditional on the same target and cohort and successful native checks during production preparation.

The 19-job balanced partition has 886 paths/job and uses `--batch-size 11533`. Its peak is 26,465,318, leaving 334,682 instructions (1.249%) below the hard allowance. This is considerably tighter than the 24-job partition; the additional five jobs cost 61,440 control shots.

The bottleneck is physical qubit 112. Its path footprint totals 502,248,664 instructions, with 998 additional control instructions per job. Reserving these controls gives lower bounds of 19 jobs at 26,800,000 and 24 at 21,440,000. The saved feasible partitions attain both lower bounds, certifying minima for the respective published-counting models on this frozen target. This does not establish server-compiler acceptance or QPU execution time.

The report contains 16,834 distinct per-path hashes and a complete seed-29 permutation. All five CSV partitions cover every trajectory exactly once. Their trajectory IDs, circuit totals, shot totals, lower bounds and reported instruction maxima agree with the JSON report. Where every job shares one bottleneck qubit, sums of per-job maxima also reproduce that qubit's aggregate path/control footprint. The context seal matches. At initial review, before the subsequent production transport/validation update, all ten code hashes matched this checkout after LF/CRLF normalization.

Mean DD insertion is 7936.874 X pulses per path across its thirteen variants. The aggregate DD-versus-scheduled-reference instruction increase agrees with the expected weighted XX/delay contribution.

The 24-job plan saves 5,148,672 shots (67.76%) against the 443-job circuit-cap configuration by repeating controls fewer times. Evolution shots are unchanged at 2,154,752; controls contribute 294,912.

**Subsequent production update:** Experiment8 now accepts this 16,834-path cohort and supports queueing the 24 jobs in one Runtime Batch. Experiment6 retains its original cohort limit. See [RuntimeBatch.md](../../RuntimeBatch.md) for the explicit cap-9,141/Batch-mode commands. The estimate remains a record of its captured target; actual production timing/resource checks, calibration screening and sufficient host memory for larger jobs still apply.

Review scope: report, context and CSV consistency. Per-path numeric caches, source and target pickle were intentionally excluded from the transferred files; native circuits were not independently recompiled or recounted here. No IBM account access, simulation or QPU submission occurred during this review.

Target captured: 2026-11-09T00:16:11.268133+00:00; estimate generated: 2026-11-09T13:45:26.167543+00:00.
Detailed checks: [verification.json](verification.json).
