# Historical initial pilot

This pilot preceded the agreed common normalization. The full study now
uses J=gamma=1 and H=(J/2) sum(XX+YY) in every model. Do not combine these
historical data with normalized full-study fits.

| Historical model | J | gamma | c in H=cJ sum(XX+YY) |
|---|---:|---:|---:|
| Model1 | 1 | 0.2 | 0.5 |
| Model2 | 1 | 2 | 0.25 |
| Model3 | 1 | 1 | 0.5 |

The [historical reference table](runs/scaling_f86137e3d449/Exact.md) contains
39 cases. The N=2 pipeline pilot used K<=128, four optimizer generations,
population multiplier 4 and one optimizer seed.

| Historical model | Fixed steps | Constant steps | Warped-family steps |
|---|---:|---:|---:|
| Model1 | No pass within 128 | 31 | 31 |
| Model2 | 29 | 6 | 6 |
| Model3 | 78 | 78 | 25 |

The [pilot schedules](runs/scaling_76a6a452f1c8/Schedules.md) and
[pilot compilation](runs/scaling_76a6a452f1c8/Resources.md) remain available.
Their whole-circuit optimization differs from the full study's declared
block-preserving compiler. These are pipeline checks, not minimum-step or
scaling claims. Local pilot checkpoints retain their historical configs;
the current default intentionally differs.
