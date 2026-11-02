# Model2 Experiment5 hardware plan

Frozen 44-step warped cosine, N=10, J=1, Gamma=2, p_L=p_R=.25, half-mixed preparation, J/4 exchange, jump-first even/odd layers.
Backend: `ibm_phoenix`. Status: `planned`.
1024 paths x 8 shots x 3 bases; three 2048-shot Z controls inside each job.
**7 jobs, 3093 circuits, 67584 total shots.**

Ideal finite-path target error: 0.0201363873; sampled-minus-mean distance: 0.00680954294; maximum descriptive two SE: 0.0111872496.
This finite ensemble passes the classical empirical checks: **False**.
The default is a hardware pilot. The passing full 16384-path classical replay does not certify a smaller subset, measurement precision or device fidelity. Compare hardware to its own saved paths as well as the physical stationary target. No seed/path selection on hardware outcomes.

Every path's requested bases stay in one job. Zero, one and held-out half-mixture controls share that job and physical mapping. Requested order is recorded; actual dispatch timing is not guaranteed. No automatic SPAM correction, twirling, ZNE or DD.

Physical path: None. Twelve active qubits when evolving: left bath, ten sites, right bath. Preparation records are ignored. Only final `outcome` counts are analyzed.
