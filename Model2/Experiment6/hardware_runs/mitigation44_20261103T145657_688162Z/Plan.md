# Model2 cosine44 hardware mitigation plan

Backend: `ibm_phoenix`. Status: `compiled`.

**16384 paths × 32 shots per arm/factor; 20 jobs, 213,292 circuits, 2,342,912 shots including controls.**

N=10, J=1, Γ/J=2, pL=pR=0.25, exchange prefactor 1/4, Δ=0; half-mixed preparation. The frozen 44-step warped cosine, jump-first order and even/odd exchange layers are unchanged.

Raw plus twirled factors [1.0, 1.25, 1.5]; 4 randomizations split each factor into 8 shots per variant. Zero, one and held-out half-mixture controls each receive 2048 raw and 2048 twirled shots inside every job.

Native CZ→CZ·CZ·CZ folding and Pauli frames preserve the intentional bath channel. All 44 resets remain. Balanced terminal X masks are decoded from final outcome counts; preparation bits are ignored. Only one-qubit frame optimization follows folding. Runtime twirling and DD are disabled; initialization is enabled.

Selected path: `[106, 116, 115, 114, 113, 112, 111, 101, 91, 81, 80, 90]`. Requested/reference path: `[106, 116, 115, 114, 113, 112, 111, 101, 91, 81, 80, 90]`. Calibration policy: `screen`.

Exact finite-path assessment: `{'ideal_subset_target_error': 0.017412736685081864, 'sampling_distance': 0.001950974974025299, 'maximum_two_standard_errors': 0.002731474265061107, 'empirical_classical_checks_passed': True, 'scope': 'This exact finite path ensemble, analytic initial mixture, no measurement shots or device noise; two SE descriptive. Full 16384-path validation is separate.'}`

This is a population-only mitigation comparison. It does not measure currents or certify the 0.02 target. All six predefined profiles and descriptive uncertainty are reported without clipping or target fitting. SPAM corrections use each job's own calibration before weighted averaging; the mixture is held out.

Each compiled job is checked against 80% of the published 26.8-million-instructions-per-qubit limit and the 10-million-execution limit. These are validation heuristics, not proof of provider compiler acceptance. All jobs are compiled before the first submission.

Default invocation and plain --resume are offline. --compile-only queries the backend without submitting. Only --execute submits; known job IDs are retrieved, and ambiguous submissions are never automatically repeated.
