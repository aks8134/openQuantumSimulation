# Experiment8 software validation

## Calibration diagnostics

The diagnostic update passed **six tests** in 109.93 seconds under Runtime
0.51/Qiskit 2.6.2:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/model2-dd-runtime051 MPLCONFIGDIR=/tmp/model2-e8-calibration-mpl \
  .venv/bin/python -m pytest -q \
  tests/test_model2_dd_calibration.py \
  tests/test_model2_combined_dd.py::test_real_folded_native_circuits_preserve_resets_frames_measurements_and_dd \
  tests/test_model2_combined_dd.py::test_preparation_resume_preserves_shards_without_recompilation
```

The new tests verify exact T1/T2/readout/CZ failure reporting, unchanged
manifest bytes, and rejection before compilation. Passing screens and explicitly
planned record-only policies still reach compilation. Existing native DD and
shard-resume checks also pass. Targets and services were local or mocked;
no IBM account, live query or hardware submission was used.

## Runtime Batch and 16,834-path update

The following Runtime 0.51/Qiskit 2.6.2 regression run passed **38 tests plus
five subtests** in 372.33 seconds:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2-e8-batch-mpl \
  .venv/bin/python -m pytest -q \
  tests/test_model2_runtime_batch.py \
  tests/test_model2_combined_dd.py \
  tests/test_model2_dd_job_estimate.py \
  tests/test_model2_mitigation_collection.py
```

The new offline cohort test creates the 16,834-path plan and verifies exactly
24 jobs, 219,202 circuits and 2,449,664 shots at cap 9,141. Experiment6's default
validator still rejects the larger cohort. Runtime Batch tests use mocked
providers and verify that all 24 jobs bind to one Batch, IDs are durable before
further uploads, results are never awaited during submission, and closure does
not cancel jobs. They cover interrupted upload/resume, ambiguous submission,
expired-Batch explicit replacement, association mismatch, failed/idempotent
closure, TTL overrides and job-mode dry-run isolation. Native DD construction,
spawned count workers, sealed cache reuse and legacy collection tests also pass.
Completed older count caches can be reported without target deserialization or
new compilation; partial native caches still enforce compiler consistency.

No IBM account, live backend query, cloud dry-run or QPU submission was used.

## Original mitigation + DD implementation

Validation used local backend targets and mocked IBM jobs. No live account,
cloud dry-run, QPU submission, or Aer experiment was used.

The Runtime 0.51/Qiskit 2.6.2 environment passed 39 tests plus seven subtests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model2_e8_mpl \
  .venv/bin/python -m pytest -q \
  tests/test_model2_combined_dd.py \
  tests/test_model2_mitigation_collection.py \
  tests/test_model2_hardware_audit.py \
  tests/test_model2_dd_experiment.py
```

Two additional Experiment8 tests passed after that run: compiled-plan resume
without recompilation, and a short open-channel equivalence check combining
CZ folding, Pauli/measurement twirling, reset and ratio-10 XX DD. The channel
check uses three-qubit SuperOp matrices and undoes the terminal measurement mask.

A further existing Experiment6 execution/analysis test passed under Qiskit
2.5.2/Runtime 0.49.0, exercising the default shared validators, legacy QPY,
job-local calibration and all six profiles. Total: **42 tests passed**, plus
the seven reported subtests.

The real local compilation fixture contains two complete Model2 trajectories
with all 44 steps and resets on a twelve-qubit GenericBackendV2 CZ chain. It
checks 880/1,100/1,320 CZ counts, frame/mask provenance, fixed physical readout
mapping, allowed DD qubits, original scheduled operation preservation, native
ISA and instruction budgets, and bounded public QPY round trips. Independent
measurements may be reordered by scheduling; validation compares physical
qubit/classical-bit pairs and still rejects changed, missing or duplicate mappings.

The scheduling-order optimization matches the installed SDK exactly on 200
mixed reset/measurement DAGs, a control-flow fallback, and a full 1,320-CZ
folded variant. Runtime ALAP/DD passes themselves are reused.

Mocked service tests cover DD-on-only publications, offline defaults, fixed
budgets, known IDs, ambiguous submission refusal, active-job handling, atomic
count-file recovery without another provider call, optional metadata separation,
render caching, dry-run/hardware ledger separation, mock-count analysis refusal,
fresh hardware cloning, source/QPY tampering, streamed six-profile analysis,
historical comparison labels and ZIPs without QPY.

These checks validate construction and recovery behavior. They do not establish
hardware accuracy, noise-scale linearity, an optimal DD cutoff, the 0.02 target,
or a full-run compilation duration/peak memory. Production native instruction
checks use the requested backend target before any submission.
