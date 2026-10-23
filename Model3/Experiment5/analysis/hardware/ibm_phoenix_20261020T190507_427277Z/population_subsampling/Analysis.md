# Population shot-subsampling analysis

Source: `ibm_phoenix_20261020T190507_427277Z`. Fixed trajectory indices: [0, 1, 2, 3].
10,000 replicates per shot level; seed 20261022. Only saved Z-basis counts were used.

Each replicate independently samples whole measured bitstrings without replacement within each path, then averages the four path populations equally. Site correlations are preserved. The shot levels share nested subsets within a replicate.

## Actual populations

[Population profiles and pointwise central 95% resampling bands](population_subsamples.png) ([PDF](population_subsamples.pdf)).
[Dependence on shot count](shot_dependence.png) ([PDF](shot_dependence.pdf)).

| Site | Full-shot hardware | Matched four-path ideal | Physical target | Latest hardware |
|---|---:|---:|---:|---:|
| 1 | 0.622742 | 0.585298 | 0.580000 | 0.620728 |
| 2 | 0.554596 | 0.472304 | 0.500000 | 0.582520 |
| 3 | 0.562225 | 0.539824 | 0.500000 | 0.567139 |
| 4 | 0.543365 | 0.522405 | 0.500000 | 0.530518 |
| 5 | 0.551147 | 0.534246 | 0.500000 | 0.537109 |
| 6 | 0.503082 | 0.475177 | 0.500000 | 0.506348 |
| 7 | 0.476105 | 0.550738 | 0.500000 | 0.473145 |
| 8 | 0.470490 | 0.515943 | 0.500000 | 0.447388 |
| 9 | 0.454010 | 0.489764 | 0.500000 | 0.419556 |
| 10 | 0.357880 | 0.380747 | 0.420000 | 0.354248 |

## Subsampling variability

Deviations below are from the full-shot hardware populations, not from the ideal prediction. The 95th-percentile maximum uses all ten sites jointly.

| Shots/path | Total Z shots | Median max deviation | 95th-percentile max deviation | All sites within 0.02 | MAE vs physical: median [2.5%, 97.5%] |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.45663 | 0.56223 | 0.00% | 0.19100 [0.10000, 0.30000] |
| 4 | 16 | 0.22610 | 0.33163 | 0.00% | 0.10275 [0.05625, 0.15625] |
| 16 | 64 | 0.11163 | 0.17099 | 0.00% | 0.06094 [0.03594, 0.09062] |
| 64 | 256 | 0.05634 | 0.08557 | 0.13% | 0.04611 [0.03009, 0.06291] |
| 128 | 512 | 0.03952 | 0.06082 | 1.40% | 0.04377 [0.03166, 0.05607] |
| 256 | 1024 | 0.02786 | 0.04245 | 12.26% | 0.04279 [0.03400, 0.05197] |
| 512 | 2048 | 0.01938 | 0.02982 | 54.98% | 0.04245 [0.03615, 0.04875] |
| 1024 | 4096 | 0.01324 | 0.02042 | 94.01% | 0.04215 [0.03788, 0.04643] |
| 2048 | 8192 | 0.00864 | 0.01328 | 99.99% | 0.04201 [0.03918, 0.04485] |
| 4096 | 16384 | 0.00497 | 0.00766 | 100.00% | 0.04188 [0.04026, 0.04355] |
| 8192 | 32768 | 0.00000 | 0.00000 | 100.00% | 0.04187 [0.04187, 0.04187] |

## Interpretation and limits

The full-shot hardware MAE is 0.04186743 against the physical target and 0.03866137 against the same four ideal paths. Increasing subsample size concentrates the estimates around this measured hardware profile; it does not move their expectation toward ideal.

At 2048 shots/path, four paths use 8192 Z measurements, matching the total number of Z measurements in the latest one-shot run. This matches the measurement count only: the trajectory ensemble and hardware execution occasion differ.

With one shot on each of only four paths, each site estimate is restricted to 0, 0.25, 0.5, 0.75 or 1. This is very different from one shot on each of 8192 paths.

For path r with N saved shots, the exact conditional covariance of an S-shot subsample mean is (1-S/N) times the unbiased sample covariance of whole bitstrings, divided by S. The equal-weight four-path mean has the sum of these covariances divided by 4 squared. Exact and empirical covariance matrices are saved in analysis.json.

These are finite-pool resampling ranges, not confidence bands for the true hardware or ideal state. Selecting all 8192 saved shots gives zero deviation from the full-shot estimate by construction; an independent new 8192-shot run would still fluctuate. The original full-shot populations have nominal per-site shot SE about 0.0026–0.0028, conditional on the four paths.

The data are aggregated counts without shot ordering. This study cannot establish temporal drift, identify gate/reset/readout mechanisms, or reproduce the newer trajectory ensemble. Central 95% profile bands are per-site; only the maximum-deviation statistic in the table addresses all sites jointly.

Empirical single-site resampling variances agree with their exact conditional values to within 3.44% relative error across all nonzero-variance shot levels/sites (10,000 Monte Carlo replicates).

At the matched measurement budget (2048 shots per old path), the median population MAE versus physical is 0.042008, with central 95% resampling range [0.039176, 0.044852].

The latest observed population MAE is 0.049002, outside that central range. Under the earlier fixed empirical distributions, reducing the measurement count usually does not reproduce the newer error. This is descriptive, not a significance test comparing runs: the four-path ensemble, execution conditions, and uncertainty in the finite original pool are not held equal to the newer experiment.

## Reproduction and preservation

```bash
MPLCONFIGDIR=/tmp/model3-subsampling-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python Model3/Experiment5/subsample_populations.py \
  --run-directory Model3/Experiment5/hardware_runs/ibm_phoenix_20261020T190507_427277Z \
  --compare-analysis Model3/Experiment5/analysis/hardware/ibm_phoenix_20261021T102532_376654Z/analysis.json
```

CSV files contain the summary, every site quantile and every path/site quantile. No Runtime connection, circuit execution or trajectory evolution was used. Input hashes were verified unchanged after writing the analysis.
