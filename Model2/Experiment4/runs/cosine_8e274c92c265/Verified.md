# Independent saved-schedule verification

Every preparation and forward node was reconstructed as an explicit weighted sum of four affine branch matrices, independently of the search/shared channel implementation. This checks ideal numerical means; hardware and global minimality are outside its scope.

| Family | Steps | Output covariance agreement | Forward error agreement | Permanent order check | Passed |
|---|---:|---:|---:|---|---|
| constant | 80 | 7.98e-17 | 1.11e-16 | True | True |
| cosine | 80 | 1.67e-16 | 1.67e-16 | True | True |

The additional permanent order check tests a signed covariance deviation B >= 0 and B-M(B) >= 0 with eigenvalue margins above 1e-10. Positivity then implies 0 <= M^(k+1)(B) <= M^k(B). For Model2's prefactor 1/4, every population/current error is bounded by the largest diagonal of B, so passing establishes the mean allowance at all later constant-final-step nodes. This can succeed when the coarser spectral-norm bound in Study.md does not. It is a floating-point matrix-order check, not an interval-arithmetic certificate, finite-sampling guarantee or full-state bound.
