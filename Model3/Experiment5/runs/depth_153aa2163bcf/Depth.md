# Experiment5: FakeKingston circuit depth

Offline target compilation only. No backend execution or noise simulation. Selected saved trajectories only.

Source: `/storage/home/aks8134/OpenQuantumSystem/Model3/Experiment4/runs/budget_419029a6f9bc/replay_R8192.json` (status `passed`).
Ancillas: boundary; physical chain: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`; optimization level 3; seed 11.

Preparation is a diagonal product mixture made by rotations and mid-circuit Z measurements; preparation results are ignored, with no system reset. All selected jump resets are included.

| Trajectory | Steps | Logical body depth | Native body depth | Full Z depth | Full XY/YX depth | Two-qubit depth | CZ gates | Resets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 23 | 139 | 140 | 142/143 | 32 | 160 | 8 |
| 0 | 64 | 150 | 1034 | 1035 | 1038/1037 | 256 | 1280 | 64 |
| 0 | 117 | 269 | 1881 | 1882 | 1884/1885 | 468 | 2340 | 117 |
| 1 | 8 | 19 | 138 | 139 | 141/142 | 32 | 160 | 8 |
| 1 | 64 | 147 | 1034 | 1035 | 1038/1037 | 256 | 1280 | 64 |
| 1 | 117 | 272 | 1883 | 1884 | 1886/1887 | 468 | 2340 | 117 |
| 2 | 8 | 21 | 138 | 139 | 141/142 | 32 | 160 | 8 |
| 2 | 64 | 148 | 1035 | 1036 | 1038/1039 | 256 | 1280 | 64 |
| 2 | 117 | 272 | 1883 | 1884 | 1886/1887 | 468 | 2340 | 117 |
| 3 | 8 | 20 | 138 | 139 | 142/142 | 32 | 160 | 8 |
| 3 | 64 | 145 | 1033 | 1034 | 1036/1037 | 256 | 1280 | 64 |
| 3 | 117 | 268 | 1881 | 1882 | 1884/1885 | 468 | 2340 | 117 |

Qiskit unweighted dependency layers, including reset and measurement; not scheduled duration.

Optimize body once, then append fixed native end rotations and measurement at final physical layout; no post-readout reoptimization.

Depths describe the compiled subset, not all saved R trajectories or a minimum-depth proof. Three final measurement settings cover all populations and currents. R repetitions and measurement shots increase total work, not depth of one trajectory.

Native QPY files contain four circuits in order: body, Z readout, XY readout, YX readout. Logical QPY files contain the body. depth.json records all gate counts and final layouts.

The previous ideal covariance tolerance does not establish noisy hardware accuracy. No shots, Aer sampling, hardware jobs, timing schedule, or duration prediction were performed.
