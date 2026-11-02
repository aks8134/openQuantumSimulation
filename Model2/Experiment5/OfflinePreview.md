# Offline compilation preview

Two fixed seed-29 trajectories of the frozen 44-step Model2 schedule were compiled against the locally installed FakeKingston snapshot. No account, live backend query, job submission or hardware execution was used.

Physical path: `[95, 94, 93, 79, 73, 74, 75, 59, 55, 54, 53, 52]`.

| Path | Basis | CZ | Bath resets | Two-qubit depth | Full native depth |
|---|---|---:|---:|---:|---:|
| 0 | z | 880 | 44 | 178 | 720 |
| 0 | xy | 880 | 44 | 178 | 722 |
| 0 | yx | 880 | 44 | 178 | 723 |
| 1 | z | 880 | 44 | 178 | 724 |
| 1 | xy | 880 | 44 | 178 | 726 |
| 1 | yx | 880 | 44 | 178 | 727 |

There are 12 active qubits for evolution; the backend-width allocation is larger. Full depth includes initial mixture preparation, measurements, bath resets and final readout rotations. These are dependency depths, not scheduled execution durations.

The snapshot did not provide a 12-qubit chain under the live pilot defaults. For this offline compilation only, the explicitly supplied screens were T1/T2 >=1 microsecond, CZ error <=0.05 and readout error <=0.05. These values are not the live command defaults. Compilation verifies circuit compatibility and costs; it does not forecast hardware accuracy.

Versions: `{'qiskit': '2.5.2', 'qiskit-ibm-runtime': '0.49.0', 'numpy': '2.5.2'}`.

Local native QPY SHA-256: `e7b834d6df8256238eaa0fee5592b3104f622979b2529f509a044e949c36517f`.

The two-path subset is a compilation sample. It is not sampling-validated. Every submitted job ID remains empty in this local manifest.

Local artifacts: `Model2/Experiment5/runs/offline_preview/cosine44_20261102T185136_537472Z`. Generated JSON/QPY remain local according to the repository artifact policy.
