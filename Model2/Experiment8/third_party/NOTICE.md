The flat reset/measurement scheduling order in `../ordering.py` is derived from
the behavior and algorithm of Qiskit IBM Runtime's
`qiskit_ibm_runtime/transpiler/passes/scheduling/utils.py`, copyright IBM
2022–2026, licensed under Apache License 2.0. The original license is included
as `QISKIT_IBM_RUNTIME_LICENSE.txt` in this directory.

Modifications replace repeated graph searches with precomputed bitset
reachability while preserving the SDK's emitted operation order on flat DAGs.
Circuits containing control flow use the installed SDK implementation.
