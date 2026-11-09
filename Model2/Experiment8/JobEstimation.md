# Backend-dependent minimum-job analysis for mitigation + DD

This estimator fetches the requested backend target and locally constructs the same frozen Experiment8 circuits: 44 cosine steps, half mixture, seed 29, transpiler seed 11/level 3, CZ/measurement twirling, factors 1/1.25/1.5, four variants per factor, then ALAP and ratio-10 selective XX DD. It counts the final native programs, including explicit delays, X pulses, resets, readout and implicit initialization. Circuit instruction counts are not multiplied by shots; executions are counted separately.

**The default counts every requested trajectory.** At 16,384 paths that is 212,992 evolution variants plus 15 controls. Native compilation and schedule validation can take substantial local CPU time. It does not run Aer, create a Sampler, submit cloud dry-runs/QPU jobs, or mutate hardware plans. It saves numeric footprints, not large circuit archives.

Use your DD environment on the machine with the IBM account:

```bash
python -m pip install -r Model2/Experiment8/requirements.txt
python -u Model2/Experiment8/estimate_jobs.py --backend ibm_phoenix --trajectory-count 16384 --shots 32 --workers 4
```

The Python command also works in PowerShell. If you literally want 16,834 paths, use:

```bash
python -u Model2/Experiment8/estimate_jobs.py --backend ibm_phoenix --trajectory-count 16834 --shots 32 --workers 4
```

Both counts are supported for estimation. The hardware runner's existing 16,384-path limit is unchanged; a larger accounting cohort is not a statement that the production runner accepts that count or that its accuracy has been validated.

The default physical chain is read from the complete Experiment6 reference bundle. Supply `--physical-path` with twelve physical qubits in left-bath/system/right-bath order to count another chain. The script checks native connectivity, durations and scheduled circuit construction; it does not apply the hardware runner's coherence/readout quality screen. `--account-file PATH` uses another account configuration. Only initial target retrieval uses the account.

Default output: `Model2/Experiment8/job_estimates/ibm_phoenix_16384/` (or the corresponding requested count). Use `--output-directory PATH` to choose another directory. Results include:

- `JobEstimate.md`: lower bounds, feasible jobs, certification status, circuits, shots including controls, instruction peaks, and balanced-runner compatibility.
- `job_estimate.json`: counting assumptions, cached-target date, path permutation, unit hashes, DD overhead and every numeric partition.
- `*_jobs.csv`: complete matched trajectory groups with actual trajectory IDs and per-job circuit/shot/instruction totals.
- `context.json`, `source.json`, `target.pkl`, `controls.json`, `path_*.json`: hashed resume cache; only the target object is serialized, never the service/account.

Each of the 13 matched variants for a path stays in one job. All fifteen zero/one/held-out half-mixture control circuits repeat in every job. With 32 shots, that is 128 evolution shots per path and 12,288 control shots per job. The raw and all intermediate/final treatments have DD enabled; no DD-off hardware arm is added.

The report compares:

- Published instruction allowance without the arbitrary 512-circuit cap.
- Balanced grouping at that published allowance, reproducible with a runner circuit cap when feasible.
- The selected instruction fraction (80% by default), with greedy and balanced packing.
- The current 512-circuit runner cap as a comparison only.

`--instruction-fraction 0.8`, `--circuit-cap 512`, `--shots` and `--control-shots` affect packing/reporting; changing them does not require recompiling an otherwise identical cached target/cohort. The script accounts for [IBM's published limits](https://quantum.cloud.ibm.com/docs/en/guides/job-limits), including the per-circuit checks. These are published validator heuristics, not measured controller memory or compiler acceptance.

The lower bound reserves complete controls in every job and sums every counted path's instruction vector on each physical qubit. It also applies the execution bound. **A full-cohort minimum is certified only when all requested paths were counted and a feasible partition attains that lower bound.** Otherwise the report gives the lower bound and a feasible upper bound; it does not call the upper bound the minimum. Different path orders can potentially improve an uncertified packing.

To resume, run exactly the same command. Each completed path is saved atomically, hashed and verified before reuse. Only a bounded number of worker tasks are active, so interruption cannot leave an unbounded submitted work queue. The frozen target and compiler versions are reused, avoiding a mixture of calibrations across a long count. A changed compiler environment or changed scientific code is refused; use a new output directory for a fresh target/count. Worker count can change on resume.

After counting, regenerate the report offline without querying IBM or compiling:

```bash
python -u Model2/Experiment8/estimate_jobs.py --backend ibm_phoenix --trajectory-count 16384 --shots 32 --report-only
```

An optional `--sample-trajectories 32` explicitly requests a footprint projection. It is not the default and never certifies the full-cohort minimum. Omit it on a subsequent invocation to extend the same cache to exhaustive counting. The quantum trajectories themselves are never duplicated to create samples.

The CSV partitions are planning output, not installed hardware manifests. A checked balanced grouping reports its compatible `--batch-size`, but production `prepare` must still compile/validate the actual run against its backend. Larger jobs require more local memory for Sampler serialization. No QPU-time, dollar, noise performance or 0.02-accuracy estimate is made here.
