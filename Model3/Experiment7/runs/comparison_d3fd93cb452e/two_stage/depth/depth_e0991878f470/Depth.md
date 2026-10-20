# Experiment5: FakeKingston circuit depth

Offline target compilation only. No backend execution or noise simulation. Selected saved trajectories only.

Source: `/storage/home/aks8134/OpenQuantumSystem/Model3/Experiment7/runs/comparison_d3fd93cb452e/two_stage/replay_seed29_R8192.json` (status `passed`).
Ancillas: boundary; physical chain: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`; optimization level 3; seed 11.

Preparation is a diagonal product mixture made by rotations and mid-circuit Z measurements; preparation results are ignored, with no system reset. All selected jump resets are included.

| Trajectory | Steps | Logical body depth | Native body depth | Full Z depth | Full XY/YX depth | Two-qubit depth | CZ gates | Resets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 73 | 171 | 1177 | 1178 | 1180/1181 | 292 | 1460 | 73 |
| 1 | 73 | 166 | 1176 | 1177 | 1179/1180 | 292 | 1460 | 73 |
| 2 | 73 | 168 | 1177 | 1178 | 1180/1181 | 292 | 1460 | 73 |
| 3 | 73 | 164 | 1175 | 1176 | 1178/1179 | 292 | 1460 | 73 |

Qiskit unweighted dependency layers, including reset and measurement; not scheduled duration.

Optimize body once, then append fixed native end rotations and measurement at final physical layout; no post-readout reoptimization.

Depths describe the compiled subset, not all saved R trajectories or a minimum-depth proof. Three final measurement settings cover all populations and currents. R repetitions and measurement shots increase total work, not depth of one trajectory.

Native QPY files contain four circuits in order: body, Z readout, XY readout, YX readout. Logical QPY files contain the body. depth.json records all gate counts and final layouts.

The previous ideal covariance tolerance does not establish noisy hardware accuracy. No shots, Aer sampling, hardware jobs, timing schedule, or duration prediction were performed.
