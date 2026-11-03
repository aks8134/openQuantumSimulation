# Model2 control-instruction audit

**Footprint projection:** 32 saved FakeKingston paths are repeated to model 16,384 native path groups. These are not counts from the complete Phoenix archive.

This is an offline application of [IBM's published validation heuristic](https://quantum.cloud.ibm.com/docs/en/guides/job-limits). It is not a measurement of controller memory or a guarantee that the IBM compiler accepts a payload.

The limit is 26,800,000 weighted instructions per qubit per job, plus 10,000,000 shot executions per job. Weights: RZ/delay 1, SX/X 2, CZ/CX/ECR 5, measure 10, reset 17, init 50. Two-qubit operations are charged to both operands. We conservatively include implicit init on every allocated qubit for every circuit. Memory counts are program-storage counts, not multiplied by shots; executions are counted separately. Manually inserted twirls/folds are already present in the QPY and are counted once. Unknown operations/control flow fail rather than being omitted.

A path's matched bases/mitigation variants stay together. Proposed new jobs each receive a complete control set. These proposed partitions are calculations only; the saved run, job IDs and QPY are unchanged.

| Workload | Paths | Single-job peak instructions/qubit | Lower-bound jobs | Feasible jobs | Jobs at 80% memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 16,384 | 85,835,972 | 4 | 4 | 5 |
| mitigation | 16,384 | 423,167,968 | 16 | 16 | 20 |

## raw

Input classification: **replicated_saved_FakeKingston_footprint_projection**.

Compilation backend: **FakeKingston, compact logical layout**.

Projection repeats observed native footprint rows, not quantum trajectories. It is not an audit of all 16,384 paths or of the Windows Phoenix archive.

Bottleneck physical index/logical position: **2**. A one-job configuration would use **1,579,008 executions** with one control set. The calculated partition uses **1,597,440 total executions** including repeated controls.

The lower bound and constructed partition agree: **True**. When they agree, this is the minimum for the supplied footprint vectors and counting model, not an unconditional hardware limit.


## mitigation

Input classification: **replicated_saved_FakeKingston_footprint_projection**.

Compilation backend: **FakeKingston, compact logical layout**.

Projection repeats observed native footprint rows, not quantum trajectories. It is not an audit of all 16,384 paths or of the Windows Phoenix archive.

Bottleneck physical index/logical position: **1**. A one-job configuration would use **2,109,440 executions** with one control set. The calculated partition uses **2,293,760 total executions** including repeated controls.

The lower bound and constructed partition agree: **True**. When they agree, this is the minimum for the supplied footprint vectors and counting model, not an unconditional hardware limit.


## Exact Phoenix archive check on Windows

From the repository root, after syncing the audit script:

```powershell
python Model2/Experiment5/instruction_audit.py --manifest $runManifest
```

This can take time because it hashes and reads every compiled program. It uses the bounded QPY loader, does not recompile or access an IBM account, and does not submit or merge jobs. It writes an audit report beside the saved run. Parameter or layout changes, missing bases, invalid QPY, and unsupported instructions are not silently estimated.

The current 6056 failure is not diagnosed by this count. Payload serialization size, controller details not captured by the public heuristic, execution duration, and provider compiler faults can still restrict a job.
