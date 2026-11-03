# Model2 cosine44: preliminary Phoenix budget

This estimate applies to the **raw Experiment5 run**, not the FakeKingston
mitigation pilot. It is a planning baseline, not a provider quote or a measured
successful Model2 hardware run. The first reported job failed with IBM compiler
error 6056 while the backend reported maintenance.

| Saved workload | Count |
| --- | ---: |
| Trajectories / seed | 16,384 / 29 |
| Shots per trajectory per basis | 32 |
| Bases | Z, XY, YX |
| Evolution circuits / shots | 49,152 / 1,572,864 |
| Controls | 3 per job, 2,048 shots each |
| Runtime jobs | 97 |
| Control shots | 595,968 |
| Total circuits / shots | 49,443 / 2,168,832 |

IBM's [usage estimation guide](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time)
gives a generic quick approximation of `2 + 0.00035 * executions` seconds,
assuming typical circuit lengths and repetition delay. The overhead is really
per **server sub-job**, so one Runtime job may incur it more than once.

Assuming one sub-job per Runtime job:

```
97 * 2 + 2,168,832 * 0.00035 = 953.0912 seconds
                                   ≈ 15.9 QPU minutes
```

At IBM's currently advertised Pay-As-You-Go **starting** price of
[$96 USD per minute](https://www.ibm.com/quantum/products), that baseline would
be approximately **$1,525 USD**. An institutional allocation or negotiated plan
can have different pricing or consume allocated QPU time instead of generating
that cash charge.

This is not a Phoenix-specific scheduled timing estimate: the live repetition
delay, initialization, mid-circuit reset/measurement timing, additional server
sub-jobs, and account pricing have not been measured for a successful Model2
job. They can change the estimate in either direction. Queue wait and local
compilation are separate from job-mode QPU usage. Aer simulation wall time is
not IBM QPU usage.

The first full job contains 510 circuits and 22,368 shots including controls.
Once it completes successfully, use its actual reported usage to refine the
remaining budget, accounting for other job/sub-job overhead and the smaller
last job. The read-only diagnostic command saves provider metrics:

```powershell
python Model2/Experiment5/run_hardware.py --resume $runManifest --inspect-job JOB_ID
```

For an internal system failure, IBM's usage guide says job/batch-mode usage is
zero when the provider classifies it as a system error. Check the failed job's
actual metrics and instance billing rather than assuming a refund or charge.

The Model2 mitigation comparison remains an offline experiment in
`Model2/Experiment6/fake_kingston_zne.py`; its 28,672-shot pilot and ZNE sampling
cost must not be substituted for this raw 97-job budget.
