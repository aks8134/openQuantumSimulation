# Aer command-line progress

Run from the repository root in PowerShell:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

python -u Model2/Experiment5/aer_progress.py --paths 16384 --shots 64 --threads 4 --chunk 16 --seed 29 --aer-seed 2026 --resume
```

This uses the existing sampler and its exact checkpoint identity. The underlying `aer_sampling.py` file is unchanged. All sampler flags pass through, so the wrapper reuses checkpoints from the previous command with the same settings and output directory. Final observable decoding now processes small path chunks to avoid expanding the whole ensemble into large bit/sign arrays; the decoded results match the original estimator.

The terminal shows completed trajectories and percentage, saved measurement-outcome counts, elapsed time for the current invocation, estimated time remaining, reused trajectory count, and the checkpoint folder. It updates after each saved chunk and prints a heartbeat every 20 seconds while work continues. Set `--progress-interval 10` to change the heartbeat interval.

The completed counter advances after durable checkpoint writes. A batch being simulated does not contribute to that counter yet. ETA uses the observed wall time for newly simulated work, including circuit construction and checkpointing; it waits for the first newly simulated batch rather than treating cached checkpoint loading as simulation throughput. ETA is a performance estimate, independent of the statistical accuracy checks.

Rerun the same command after an interruption to resume. An already-running sampler keeps its original display, including its line after every 16 trajectories; the richer display applies on the next start or resume. No full experiment is started merely by adding this wrapper.

For a fully sampled run interrupted during final analysis, [AerRecovery.md](AerRecovery.md) provides an analysis-only command that verifies and compares 32/64-shot samples without loading Aer or modifying the original checkpoints.
