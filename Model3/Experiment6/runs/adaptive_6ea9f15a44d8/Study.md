# Experiment6: automatic budget-driven schedule

Controller status: **mean_target_met**.
Preparation: half_mixed; steps: 273/500; output time: 15.397712.

Total tolerance 0.02; mean 0.016; sampling 0.004.
Worst forward mean error: 0.0140855002. Classical continuation: 100000 steps.

The controller uses continuous bounded timestep optimization and current mean-state feedback. No prior schedules or timestep grid were loaded. Forecast scores are heuristics; actual mean validation checks every node. Sampling is checked at output only.

| Steps | dt |
|---:|---:|
| 4 | 0.260561788 |
| 4 | 0.176318343 |
| 8 | 0.356741739 |
| 4 | 0.13100295 |
| 4 | 0.0902984074 |
| 4 | 0.0608417703 |
| 4 | 0.0466924579 |
| 4 | 0.0355730068 |
| 4 | 0.0103427211 |
| 4 | 0.00684849293 |
| 4 | 0.00459982686 |
| 4 | 0.00300212982 |
| 4 | 0.00200139368 |
| 4 | 0.00134149871 |
| 4 | 0.106585674 |
| 4 | 0.12322929 |
| 8 | 0.182504305 |
| 8 | 0.0956673961 |
| 4 | 0.0726778988 |
| 4 | 0.047839527 |
| 4 | 0.0404253519 |
| 4 | 0.0739468361 |
| 4 | 0.0553587461 |
| 4 | 0.0447918417 |
| 4 | 0.015335955 |
| 4 | 0.00957924934 |
| 4 | 0.00628008849 |
| 4 | 0.00406319908 |
| 4 | 0.00262823809 |
| 4 | 0.0483096552 |
| 4 | 0.0244868736 |
| 4 | 0.0244814826 |
| 4 | 0.0395526945 |
| 4 | 0.0333546525 |
| 4 | 0.0296575967 |
| 4 | 0.0266471677 |
| 4 | 0.0386067315 |
| 4 | 0.0226064232 |
| 4 | 0.0136083543 |
| 4 | 0.0086816923 |
| 4 | 0.00554817988 |
| 4 | 0.00364012026 |
| 4 | 0.0631383539 |
| 4 | 0.0254479106 |
| 4 | 0.0362202534 |
| 4 | 0.0443144641 |
| 4 | 0.0428056845 |
| 4 | 0.0399422713 |
| 4 | 0.037241081 |
| 4 | 0.0356820284 |
| 4 | 0.035427334 |
| 4 | 0.0378519916 |
| 4 | 0.042541715 |
| 4 | 0.0471391771 |
| 8 | 0.0554948744 |
| 4 | 0.0403013123 |
| 4 | 0.0361205969 |
| 4 | 0.0320168656 |
| 4 | 0.0279118461 |
| 4 | 0.0239514354 |
| 4 | 0.0206825077 |
| 4 | 0.0178995759 |
| 8 | 0.0153008697 |
| 1 | 0.0001 |

| Replay | Status | Sampled output error | Sampling distance | Two SE |
|---|---|---:|---:|---:|
| seed29_R8192 | passed | 0.0130768879 | 0.00198044348 | 0.00139440477 |
| seed30_R8192 | passed | 0.0130159567 | 0.000916216259 | 0.00138721323 |
| seed31_R8192 | passed | 0.0123963926 | 0.00131704391 | 0.00139758627 |

Reference-assisted covariance model; no hardware feedback, shots, hardware noise, or compiled-depth calculation. The controller does not guarantee a globally shortest schedule or beat an offline optimized schedule. Validation covers a finite future mean window, not permanent settling.

The numerical optimizer is local; see [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html).
