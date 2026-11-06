# Model2 Experiment5 hardware plan

Frozen 44-step warped cosine, N=10, J=1, Gamma=2, p_L=p_R=.25, half-mixed preparation, J/4 exchange, jump-first even/odd layers.
Backend: `ibm_phoenix`. Status: `compiled`.
16384 paths x 32 shots x 3 bases; three 2048-shot Z controls inside each job.
**97 jobs, 49443 circuits, 2168832 total shots.**

Ideal finite-path target error: 0.0174127367; sampled-minus-mean distance: 0.00195097497; maximum descriptive two SE: 0.00273147427.
This finite ensemble passes the classical empirical checks: **True**.
The default is a hardware pilot. The passing full 16384-path classical replay does not certify a smaller subset, measurement precision or device fidelity. Compare hardware to its own saved paths as well as the physical stationary target. No seed/path selection on hardware outcomes.

Every path's requested bases stay in one job. Zero, one and held-out half-mixture controls share that job and physical mapping. Requested order is recorded; actual dispatch timing is not guaranteed. No automatic SPAM correction, twirling, ZNE or DD.

Physical path: [68, 69, 59, 58, 57, 56, 66, 76, 86, 96, 97, 98]. Twelve active qubits when evolving: left bath, ten sites, right bath. Preparation records are ignored. Only final `outcome` counts are analyzed.
