# FakeKingston depth and gate counts: 117-step forward preparation

The saved 117-step output circuit was transpiled for four trajectories (0,1,2,3) from the successful seed-29, R=8192 replay. Optimization level 3, transpiler seed 11, the same 12-qubit physical chain, and the same native target versions were used for both comparisons.

| Per-trajectory metric, across Z/XY/YX readouts | Old 490-step preparation | New 117-step preparation |
|---|---:|---:|
| Native circuit depth | 7,850–7,855 | 1,882–1,887 |
| Two-qubit depth | 1,960 | 468 |
| CZ gates | 9,800 | 2,340 |
| Unitary gates including RZ | 65,712–65,766 | 14,715–14,761 |
| Unitary gates excluding RZ | 38,985–39,004 | 9,159–9,194 |
| Ancilla resets | 490 | 117 |
| Measurements (10 preparation + 10 final) | 20 | 20 |
| All operations including resets and measurements | 66,222–66,276 | 14,852–14,898 |

Mean native depth falls by 76.00%; CZ count falls by 76.12%.

Unitary gate counts sum SX, RZ, X, CZ (and ID if present), excluding resets and measurements. RZ operations are listed separately because they can be virtual frame updates. Depth is an unweighted Qiskit dependency-layer count including preparation, resets, and readout; it is not scheduled execution time. There are 12 active qubits (10 system + 2 endpoint baths), with the target circuit allocated at the 156-qubit backend width.

The preparation is 8 steps at dt=0.75, 56 at dt=0.25, then 53 at dt=0.05, ending at physical time 22.65. The additional 200 mean steps used for the forward stability check are classical validation and are not in these circuits. Sampling was validated at the output only. The old result used a trailing window, so the resource reduction includes the changed stopping rule.

No hardware or noisy simulation was executed. These ranges cover the four compiled paths and three final readout settings, not all 8192 trajectories or a globally optimized layout. Three settings cover all populations and currents; repetitions and shots multiply total workload without changing an individual circuit depth.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/model3_exp5_forward117_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/circuit_depth.py \
  --source Model3/Experiment4/runs/budget_419029a6f9bc/replay_R8192.json \
  --trajectory-indices 0 1 2 3 --prefix-steps 8 64 117
```

## Files

- [Full per-path report](../runs/depth_153aa2163bcf/Depth.md).
- [Complete metrics](../runs/depth_153aa2163bcf/depth.json).
- [Depth/gate-count plot](../runs/depth_153aa2163bcf/figures/depth.png).
- [One-step circuit with both ancillas visible](../runs/depth_153aa2163bcf/figures/one_step_all_ancillas.png).
- [Path 0 native circuits](../runs/depth_153aa2163bcf/circuits/trajectory_0_steps_117_native.qpy).
- [Machine-readable comparison](Forward117DepthComparison.json).

Each native QPY file contains four circuits: body, Z, XY, YX. All four saved final-path QPY files were reloaded, and their depths and gate counts matched the recorded metrics. Source SHA-256 checks passed and the compared settings match.
