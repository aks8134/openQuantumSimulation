# Experiment data and Git

Keep Python code, tests, Markdown reports, compact analysis summaries, and
figures in Git. Generated numerical archives (`.npz`), circuit binaries
(`.qpy`), full replay/checkpoint JSON in experiment `results/`, `runs/`, and
`study/` directories, and generated certificate JSON stay local. Hardware
outputs and credentials also stay local. `.gitignore` enforces this policy
for new files; it does not automatically untrack historical files.

Existing experiment outputs have not been deleted. Run the experiment's
documented command to regenerate outputs in a fresh checkout. Reports may
link to those local outputs even when the outputs are not stored in Git.
Use external storage or release assets for sharing complete datasets.

## Portable cosine hardware input

The small tracked file
`Model3/Experiment5/inputs/cosine72_seed29_paths0-3.json` contains the exact
72-step cosine schedule, the first four saved jump paths, their final ideal
covariances, preparation, and mean-channel references. Its provenance records
the hash and assessment of the original R=8192 replay. The compact input is
not a new four-trajectory accuracy validation.

The hardware runner defaults to this file. From the repository root in
Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\Model3\Experiment5\run_hardware.py `
  --backend ibm_phoenix `
  --account-file ".\IBMRuntime\apikey.json" `
  --trajectory-indices 0 1 2 3 `
  --shots 8192
```

To export another subset from a full local replay:

```bash
.venv/bin/python Model3/Experiment5/export_hardware_input.py \
  --source Model3/Experiment7/runs/comparison_d3fd93cb452e/cosine/replay_seed29_R8192.json \
  --trajectory-indices 0 1 2 3 \
  --output Model3/Experiment5/inputs/cosine72_seed29_paths0-3.json
```

Exported files must be put outside ignored output directories if they are to
be committed. A hardware input can also be supplied to `circuit_depth.py`
with `--source`. Backend compilation and measurement settings are unchanged.
