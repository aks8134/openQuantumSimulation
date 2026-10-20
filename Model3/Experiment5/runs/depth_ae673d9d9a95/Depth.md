# Experiment5: FakeKingston circuit depth

Offline target compilation only. No backend execution or noise simulation. Selected saved trajectories only.

Source: `/storage/home/aks8134/OpenQuantumSystem/Model3/Experiment4/runs/budget_e8ec77f803de/replay_R8192.json` (status `passed`).
Ancillas: boundary; physical chain: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`; optimization level 3; seed 11.

Preparation is a diagonal product mixture made by rotations and mid-circuit Z measurements; preparation results are ignored, with no system reset. All selected jump resets are included.

| Trajectory | Steps | Logical body depth | Native body depth | Full Z depth | Full XY/YX depth | Two-qubit depth | CZ gates | Resets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 49 | 330 | 331 | 333/334 | 80 | 400 | 20 |
| 0 | 40 | 94 | 651 | 652 | 654/655 | 160 | 800 | 40 |
| 0 | 490 | 1101 | 7849 | 7850 | 7852/7853 | 1960 | 9800 | 490 |
| 1 | 20 | 46 | 331 | 332 | 334/335 | 80 | 400 | 20 |
| 1 | 40 | 90 | 651 | 652 | 654/655 | 160 | 800 | 40 |
| 1 | 490 | 1095 | 7851 | 7852 | 7854/7855 | 1960 | 9800 | 490 |
| 2 | 20 | 51 | 331 | 332 | 334/335 | 80 | 400 | 20 |
| 2 | 40 | 95 | 652 | 653 | 655/656 | 160 | 800 | 40 |
| 2 | 490 | 1097 | 7851 | 7852 | 7854/7855 | 1960 | 9800 | 490 |
| 3 | 20 | 45 | 331 | 332 | 334/335 | 80 | 400 | 20 |
| 3 | 40 | 92 | 652 | 653 | 655/656 | 160 | 800 | 40 |
| 3 | 490 | 1109 | 7850 | 7851 | 7853/7854 | 1960 | 9800 | 490 |

Qiskit unweighted dependency layers, including reset and measurement; not scheduled duration.

Optimize body once, then append fixed native end rotations and measurement at final physical layout; no post-readout reoptimization.

Depths describe the compiled subset, not all saved R trajectories or a minimum-depth proof. Three final measurement settings cover all populations and currents. R repetitions and measurement shots increase total work, not depth of one trajectory.

Native QPY files contain four circuits in order: body, Z readout, XY readout, YX readout. Logical QPY files contain the body. depth.json records all gate counts and final layouts.

The previous ideal covariance tolerance does not establish noisy hardware accuracy. No shots, Aer sampling, hardware jobs, timing schedule, or duration prediction were performed.
