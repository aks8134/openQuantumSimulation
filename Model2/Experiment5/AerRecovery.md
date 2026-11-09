# Recover analysis after an Aer run finishes sampling

When the progress display reaches all trajectories and says `finalizing analysis`, the chunk samples and their metadata have already been saved. The old final estimator expands the entire ensemble into bit, sign and product arrays. At 16,384 trajectories × 64 shots × three bases, a single full floating-point bit array is 240 MiB; several such arrays and intermediates can coexist. A final `Killed` message alone does not identify whether memory, a scheduler limit or another external termination caused the stop.

Analyze the existing samples without any simulator execution:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u Model2/Experiment5/analyze_aer_samples.py --study Model2/Experiment5/runs/aer_sampling/aer_415156585c7c/study.json --shots 32 64 --plot
```

The analyzer verifies the source hash and every checkpoint hash, metadata record, shape, bit range and contiguous trajectory coverage. It accepts an interrupted final analysis only when every requested trajectory already has a valid saved chunk. Missing or corrupt data cause an error; the script never starts Aer or contacts IBM.

It reads one small sample archive at a time and retains just 19 per-path observable values for each requested shot count, rather than the full shot/bit tensor. The 32-shot result uses the first 32 actual outcomes from every trajectory and basis in the existing 64-shot run. Both profiles therefore use the same paths and matched measurements.

Results default to `Model2/Experiment5/analysis/aer_415156585c7c/`: `Analysis.md`, `analysis.json`, `shot_comparison.csv`, `observables.csv`, and PNG/PDF plots with `--plot`. Analysis progress appears every 64 checkpoints. Input source, study, metadata and NPZ files remain unchanged; a successful analysis is recorded separately rather than changing the original study's incomplete flag.

The original `aer_sampling.py` file remains unchanged so its checkpoint identity remains valid. The updated progress wrapper also uses a bounded final estimator for future runs and resumes. Use the analysis-only command above for this completed sample set: it avoids importing Aer, loading circuits, or repeating simulation.
