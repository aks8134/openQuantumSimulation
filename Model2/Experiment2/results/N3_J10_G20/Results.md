# Randomized single-jump Ding–Li–Lin dilation

N=3, J=10, Gamma=20, p=0.25, Delta=0, exchange prefactor=0.25, T=12. This is a numerical-method experiment on the same equal-reservoir problem as Experiment 1.

The implementation follows the randomized Lindbladian framework and single-ancilla first-order implementation in Appendix B of [Chen, Li, Lu and Ying](https://arxiv.org/html/2407.06594v5). Its general convergence results distinguish the mean channel from one sampled realization. We benchmark observables here; we do not certify diamond-norm accuracy or apply the paper's Gibbs-sampler guarantees.

## Construction

Write $\mathcal L=-i[H,\cdot]+\sum_{a=1}^4\mathcal D[V_a]$. Draw $a$ independently with probability $\mu_a$ and use $\mathcal L_a=-i[H,\cdot]+\mathcal D[V_a/\sqrt{\mu_a}]$. Then $\sum_a\mu_a\mathcal L_a=\mathcal L$; the Hamiltonian is applied once per step, with its original strength.

The sampled interaction is $W_a=\exp[-i\sqrt{\tau}(|1\rangle\langle0|\otimes V_a/\sqrt{\mu_a}+|0\rangle\langle1|\otimes V_a^\dagger/\sqrt{\mu_a})]$. Prepare the ancilla in zero, apply this interaction and the separate $U_H=\exp(-i\tau H)$ factor, trace/reset the ancilla. Both factor orders are tested. No exact boundary-reservoir exponential is used.

Uniform sampling uses mu=(1/4,1/4,1/4,1/4). Rate-weighted sampling uses mu=(p/2,(1-p)/2,p/2,(1-p)/2), in left-pump, left-loss, right-pump, right-loss order.

For jump rate $r_a$, the boundary rotation angle is $\theta_a=\sqrt{r_a\tau/\mu_a}$, and its transition probability is $\sin^2\theta_a$, not $1-e^{-r_a\tau/\mu_a}$. Rate weighting equalizes all four angles. In the mean channel, this makes the pumping/loss transition ratio exactly p/(1-p), so the product target is fixed for either factor order. Individual sequences do not fix that target. Uniform sampling instead selects $p_{\rm eff}=\mu_+\sin^2\theta_+/(\mu_+\sin^2\theta_++\mu_-\sin^2\theta_-)$.

## Measured accuracy

Maximum errors below cover all three initial preparations, all sites/bonds, every step through t=1, and observations approximately 0.1 apart afterwards. Late observations are rounded to actual step boundaries; no fractional steps are inserted. These are sampled-time errors, not uniform-time bounds. The previous Lie baseline checks every step through t=0.1 and a dense observation grid afterwards; its known maximum is in that early window.

| Sampling | Order | Step | Population error | Current error / abs(J) | Steps | Logical factors |
|---|---|---:|---:|---:|---:|---:|
| rate-weighted | hamiltonian-first | 0.005 | 0.010895 | 0.00730139 | 2400 | 4800 |
| rate-weighted | hamiltonian-first | 0.002 | 0.00431599 | 0.00289805 | 6000 | 12000 |
| rate-weighted | hamiltonian-first | 0.001 | 0.00215123 | 0.0014451 | 12000 | 24000 |
| rate-weighted | hamiltonian-first | 0.00075 | 0.00161216 | 0.0010831 | 16000 | 32000 |
| rate-weighted | hamiltonian-first | 0.000625 | 0.00134295 | 0.000902279 | 19200 | 38400 |
| rate-weighted | hamiltonian-first | 0.0005 | 0.00107394 | 0.000721579 | 24000 | 48000 |
| rate-weighted | hamiltonian-first | 0.0004 | 0.000858882 | 0.000577105 | 30000 | 60000 |
| rate-weighted | hamiltonian-first | 0.000375 | 0.000805139 | 0.000541 | 32000 | 64000 |
| rate-weighted | hamiltonian-first | 0.0003125 | 0.000670819 | 0.000450757 | 38400 | 76800 |
| rate-weighted | hamiltonian-first | 0.00025 | 0.000536551 | 0.000360544 | 48000 | 96000 |
| rate-weighted | hamiltonian-first | 0.000125 | 0.000268172 | 0.000180211 | 96000 | 192000 |
| rate-weighted | jump-first | 0.005 | 0.00693102 | 0.0048131 | 2400 | 4800 |
| rate-weighted | jump-first | 0.002 | 0.00265711 | 0.00196296 | 6000 | 12000 |
| rate-weighted | jump-first | 0.001 | 0.00131002 | 0.000987511 | 12000 | 24000 |
| rate-weighted | jump-first | 0.00075 | 0.000979089 | 0.000741748 | 16000 | 32000 |
| rate-weighted | jump-first | 0.000625 | 0.000814484 | 0.000618591 | 19200 | 38400 |
| rate-weighted | jump-first | 0.0005 | 0.00065045 | 0.00049524 | 24000 | 48000 |
| rate-weighted | jump-first | 0.0004 | 0.000519635 | 0.00039643 | 30000 | 60000 |
| rate-weighted | jump-first | 0.000375 | 0.000486988 | 0.000371708 | 32000 | 64000 |
| rate-weighted | jump-first | 0.0003125 | 0.00040547 | 0.000309872 | 38400 | 76800 |
| rate-weighted | jump-first | 0.00025 | 0.000324094 | 0.00024799 | 48000 | 96000 |
| rate-weighted | jump-first | 0.000125 | 0.000161765 | 0.000124087 | 96000 | 192000 |
| uniform | hamiltonian-first | 0.005 | 0.0203814 | 0.0100799 | 2400 | 4800 |
| uniform | hamiltonian-first | 0.002 | 0.00803928 | 0.00401758 | 6000 | 12000 |
| uniform | hamiltonian-first | 0.001 | 0.00400109 | 0.00200626 | 12000 | 24000 |
| uniform | hamiltonian-first | 0.00075 | 0.00299736 | 0.0015042 | 16000 | 32000 |
| uniform | hamiltonian-first | 0.000625 | 0.00249636 | 0.0012533 | 19200 | 38400 |
| uniform | hamiltonian-first | 0.0005 | 0.00199594 | 0.00100247 | 24000 | 48000 |
| uniform | hamiltonian-first | 0.0004 | 0.00159602 | 0.000801871 | 30000 | 60000 |
| uniform | hamiltonian-first | 0.000375 | 0.0014961 | 0.000751729 | 32000 | 64000 |
| uniform | hamiltonian-first | 0.0003125 | 0.00124639 | 0.000626388 | 38400 | 76800 |
| uniform | hamiltonian-first | 0.00025 | 0.000996824 | 0.000501069 | 48000 | 96000 |
| uniform | hamiltonian-first | 0.000125 | 0.000498126 | 0.000250493 | 96000 | 192000 |
| uniform | jump-first | 0.005 | 0.0208468 | 0.00475804 | 2400 | 4800 |
| uniform | jump-first | 0.002 | 0.00814819 | 0.00182507 | 6000 | 12000 |
| uniform | jump-first | 0.001 | 0.00404274 | 0.000899878 | 12000 | 24000 |
| uniform | jump-first | 0.00075 | 0.00302618 | 0.00067256 | 16000 | 32000 |
| uniform | jump-first | 0.000625 | 0.00251937 | 0.000559492 | 19200 | 38400 |
| uniform | jump-first | 0.0005 | 0.00201354 | 0.000446814 | 24000 | 48000 |
| uniform | jump-first | 0.0004 | 0.00160958 | 0.000356955 | 30000 | 60000 |
| uniform | jump-first | 0.000375 | 0.00150869 | 0.000334528 | 32000 | 64000 |
| uniform | jump-first | 0.0003125 | 0.00125663 | 0.000278531 | 38400 | 76800 |
| uniform | jump-first | 0.00025 | 0.00100482 | 0.000222631 | 48000 | 96000 |
| uniform | jump-first | 0.000125 | 0.00050192 | 0.000111122 | 96000 | 192000 |

Selection criteria: population error <=0.001 and current error/abs(J) <=0.001. The recommendation additionally requires population error <=0.0005.

```json
{
  "uniform/hamiltonian-first": {
    "largest_passing_tested_step": 0.00025,
    "recommended_tested_step": 0.000125
  },
  "uniform/jump-first": {
    "largest_passing_tested_step": 0.000125,
    "recommended_tested_step": null
  },
  "rate-weighted/hamiltonian-first": {
    "largest_passing_tested_step": 0.0004,
    "recommended_tested_step": 0.000125
  },
  "rate-weighted/jump-first": {
    "largest_passing_tested_step": 0.00075,
    "recommended_tested_step": 0.000375
  }
}
```

![Mean error and cost](comparison.png)

## Cost and sequence fluctuations

A sampled step uses one Hamiltonian and one jump-interaction factor, one ancilla qubit, and one reset cycle. The previous shared-ancilla Lie step uses one Hamiltonian and four jump-interaction factors, three binary ancilla qubits encoding five levels, and one reset cycle. Counts are logical factors per sequence, not compiled gate counts, circuit depth, total measurement cost, or a hardware speedup. The full Hamiltonian factor is treated as an exact oracle in both methods.

For rate-weighted jump-first sampling, the largest passing tested step is 0.00075; the largest tested step with the additional population margin is 0.000375.

The 64-sequence Monte Carlo mean has maximum population error 0.01293, while the largest measured sequence standard deviation is 0.04804. That finite collection of random sequences does not reach the 0.001 accuracy target, even though the analytically evaluated ensemble does. This is a material distinction for the cost comparison.

Random-sequence diagnostics (conditional channel expectations, no shot noise):

```json
{
  "seed": 20261011,
  "realizations": 64,
  "realized_sampling_probabilities": [
    0.12473095703125,
    0.375703125,
    0.12482958984375,
    0.374736328125
  ],
  "dt": 0.000375,
  "policy": "rate-weighted",
  "order": "jump-first",
  "maximum_population_sequence_standard_deviation": 0.04803535172594711,
  "maximum_population_standard_error": 0.006004418965743389,
  "maximum_normalized_current_sequence_standard_deviation": 0.012124439847995503,
  "maximum_sample_mean_population_error_vs_ensemble": 0.012929882621017463,
  "maximum_sample_mean_population_error_vs_exact": 0.012929882621017463,
  "final_population_standard_deviations": [
    [
      0.03593528582710383,
      0.013105743002962612,
      0.033849481754311614
    ],
    [
      0.03593528582710383,
      0.013105743002962612,
      0.033849481754311614
    ],
    [
      0.03593528582710383,
      0.013105743002962612,
      0.033849481754311614
    ]
  ],
  "pointwise_seed_count_estimate_for_population_standard_error_0p001": 2308
}
```

![Random sequences](random_sequences.png)

Reusing one random sequence across many shots measures that sequence's channel. Resample the entire sequence per measurement shot to estimate the average channel directly. For population measurements, the marginal outcome remains Bernoulli with the averaged population, so the observed sequence variance is not automatically an extra multiplier in the fundamental shot cost. Averaging conditional channel expectations over a fixed, small collection of seeds is a different estimator and can have substantial sampling error. Standard errors here are pointwise, not simultaneous confidence bounds over every observation.

## Numerical validation and scope

XX covariance evolution exactly closes for these boundary channels, including non-Gaussian ensemble states. We use powers of its n^2-dimensional affine map to evaluate the exact sequence average to roundoff; this is a classical evaluation shortcut, not an extra quantum implementation. We do not reconstruct full density matrices or certify trace distances. The fast runner requires Delta=0 and equal reservoirs. Full two-level ancilla Kraus construction is also available in randomized.sampled_kraus.

Tests compare all four branches with full density-matrix dilation on a coherent mixed three-spin state, enumerate all 16 two-step sequences, check the complete product target, and compare continuous reference covariances with the full Liouvillian. Eigenchannel powers are checked against 100 direct steps and their decomposition residual. Residual bounds concern exact-arithmetic propagation of the measured residual; they do not rigorously bound all floating-point operations.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/model2_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python Model2/Experiment2/qdrift.py --n-qubits 3 \
  --J 10 --gamma 20 --p 0.25 --t-final 12 \
  --realizations 64
```
