# Experiment7 implementation checks

Validated on 2026-11-06 (America/New_York). No IBM service jobs were submitted.
No Aer noise/performance study or QPU effectiveness claim is included here.

The focused suite passed **11 tests** on each setup:

| Setup | Qiskit | Runtime | Plotly | Result |
| --- | --- | --- | --- | --- |
| Existing workspace, visualization isolated in `/tmp` | 2.5.2 | 0.49.0 | 6.9.0 | 11 passed |
| Dependencies isolated in `/tmp` | 2.6.2 | 0.51.0 | 7.1.0 | 11 passed |

Both used Python 3.14.7. Existing workspace packages were not upgraded. Runtime
0.51's actual SamplerV2 dry-run signature and options schema were also checked
without calling `run`. Actual cloud dry-run access/acceptance remains untested.

The suite checks the ideal open-system channel across a reset using SuperOp,
original local operation times, system-only placement, no post-readout DD,
minimum idle rules, real IBM-drawer HTML output from synthetic timing metadata,
branch/controller gap context, complete matched packing and controls, policy
provenance, result tampering, target timing drift, maintenance, ambiguous/known
job recovery, mock-count rejection, paired population/current reconstruction and
refusal to clone a dry-run whose reported added X pulses are missing.

The service-bookkeeping tests use synthetic IBM-shaped timing and mocked jobs.
They are not simulated or observed Phoenix schedules. The HTML test invokes the
real `draw_circuit_schedule_timing` implementation on that synthetic fixture.

Additional full-depth local checks used a 12-qubit GenericBackendV2 with a CZ
chain, seed 17, trajectory seed 29 and transpiler seed 11, optimization level 1:

- A two-trajectory scout compiled all three bases, including all 44 resets.
  Its footprint fit one job: 9 circuits and 6,336 intended shots with controls.
- For trajectory 0 in Z, a full native/ALAP/DD pair retained **880 CZ gates,
  44 resets and every original operation's local block/start/duration**.
- A deliberately broad test policy allowed all ten system sites with idle ratio
  2. It added 1,190 X pulses in 595 fitting XX windows and zero bath DD pulses.
  This is a construction stress test, not the policy chosen for Phoenix.

The full-depth check verifies placement and preservation of the compiled
operation stream, not a full 12-qubit density-matrix/channel simulation. The
short-channel SuperOp test supplies an independent ideal-channel check. Neither
establishes a benefit under real idle noise, calibration drift or pulse error.

Run the focused suite in an environment with pytest and the experiment
dependencies:

```powershell
python -m pytest tests/test_model2_dd_experiment.py -q
```

All new Python files compile, and `git diff --check` passes. Previous Model2 and
Model3 experiment code/manifests were not modified.

## Collection/render recovery update, 2026-11-07

The focused suite now passes **13 tests on both SDK setups** above. New checks
cover download-only collection (no plotting, metrics, usage or resubmission),
local reuse of saved results without account discovery, and fetching optional
job details later without changing raw timing/count hashes or policy evidence.
Interrupted rendering retains its completed view and index, resumes only the
missing view, and rebuilds changed HTML detected by its output hash. Rendering
can select one trajectory, basis and view. The actual IBM drawer HTML test also
passes with atomic HTML output. Service data in these recovery tests is mocked;
the user's Windows job was not downloaded or inspected here.
