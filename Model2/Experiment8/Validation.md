# Experiment8 software validation

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
