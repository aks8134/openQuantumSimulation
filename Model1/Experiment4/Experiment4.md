# Experiment 4: Dynamic Lie circuits on IBM hardware

Experiment 4 retains only the optimized dynamic Lie–Trotter construction
from Experiment 3 and adds a common execution path for local Aer sampling
and real IBM Quantum hardware. The system is the same \(N\)-site
boundary-damped \(XX\) chain,

\[
H=\frac{J}{2}\sum_{n=0}^{N-2}
(X_nX_{n+1}+Y_nY_{n+1})
+\frac{h}{2}\sum_{n=0}^{N-1}Z_n,
\]

\[
V_1=\sqrt{\gamma}\,\sigma^-_0,
\qquad
V_2=\sqrt{\gamma}\,\sigma^-_{N-1}.
\]

The circuit uses \(N+2\) qubits: \(N\) system qubits and two shared-ancilla
qubits. Every physical step applies one even bond layer, one odd bond
layer, the first boundary exchange, measurement of the low ancilla, and a
classically conditioned second boundary exchange. Both ancillas are then
reset:

\[
H_{\mathrm{even}}\rightarrow H_{\mathrm{odd}}\rightarrow K_1
\rightarrow\operatorname{measure}(a_L)
\rightarrow
\begin{cases}
K_2,&c=0,\\
I,&c=1,
\end{cases}
\rightarrow\operatorname{reset}(a_L,a_H).
\]

The implementation is [`dynamic_lie_trotter.py`](dynamic_lie_trotter.py).
The number of system qubits and execution backend are command-line
arguments rather than source-code constants.

## Why hardware execution uses Sampler

IBM's `EstimatorV2` does not support dynamic circuits containing
mid-circuit measurement and feed-forward. IBM recommends using
`SamplerV2`, adding terminal measurements in the required Pauli bases,
and reconstructing expectation values from the sampled bit strings. See
IBM's [dynamic-circuit execution guide](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits).

Experiment 4 consequently uses five terminal-measurement settings at every
saved time:

| Setting | Site basis | Quantities obtained |
|---|---|---|
| `Z` | \(Z\) on every site | All local populations |
| `X` | \(X\) on every site | All \(X_iX_{i+1}\) terms |
| `Y` | \(Y\) on every site | All \(Y_iY_{i+1}\) terms |
| `XY` | \(X\) on even sites, \(Y\) on odd sites | Alternating \(X_iY_{i+1}\) or \(Y_iX_{i+1}\) terms |
| `YX` | \(Y\) on even sites, \(X\) on odd sites | The complementary cross terms |

The sampled Pauli expectations reconstruct the same observable families
used in Experiment 3:

\[
\langle n_i\rangle=\frac{1-\langle Z_i\rangle}{2},
\]

\[
C_i^{XY}=\langle X_iX_{i+1}\rangle
+\langle Y_iY_{i+1}\rangle,
\]

\[
I_{i\rightarrow i+1}
=\frac{J}{2}\left(
\langle Y_iX_{i+1}\rangle
-\langle X_iY_{i+1}\rangle
\right).
\]

Thus 51 saved times produce \(51\times5=255\) sampling circuits, independent
of \(N\). Aer submits up to 300 circuits per job. IBM hardware instead
defaults to five circuits per job—one saved time with all five measurement
bases—because multiple dynamic circuits are concatenated in the controller
and their memory use accumulates. `--batch-size` can override the group size.
If IBM returns
[controller-memory error 6073](https://quantum.cloud.ibm.com/docs/en/errors),
the script reports the failure without automatically resubmitting any
circuits; rerun it with a smaller `--batch-size`.

Classical bit 0 is reserved for the mid-circuit ancilla measurement.
System site \(n\) is terminally measured into classical bit \(n+1\).

## Credentials

For a real backend, the script loads the API key and Cloud Resource Name
(CRN) from

```text
IBMRuntime/apikey.json
```

using the repository's `load_ibm_account` function. The expected fields are
`apikey` and `crn`. The credential file is already excluded by `.gitignore`.
The script does not print the key or write it to the result file.

An alternative credential file can be selected with `--account-file`.

## Running on Aer

The safe default is local Aer execution:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py
```

Choose a different system size with

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 7 \
  --backend aer
```

A short pilot run is useful before submitting to hardware:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 7 \
  --backend aer \
  --time-points 11 \
  --shots 1024
```

## Uniform and arbitrary saved times

By default, `--time-points M --t-final T` creates the uniform grid

\[
t_k=k\frac{T}{M-1}.
\]

An explicit nonuniform grid can instead be supplied with `--times`. The
values must start at zero and be strictly increasing. Space-separated and
comma-separated forms are both accepted:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --times 0 0.1 0.25 0.7 1.5 3.0
```

or equivalently:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --times 0,0.1,0.25,0.7,1.5,3.0
```

`--times` overrides `--time-points` and `--t-final`. The internal Trotter
resolution can be set independently with `--trotter-delta-t`. This value is
the maximum substep size, not a requirement that saved times lie on a fixed
grid. For example,

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --times 0 0.7 \
  --trotter-delta-t 0.2
```

constructs the (t=0.7) circuit from substeps (0.2,0.2,0.2,0.1). In
general, each interval \(\Delta t_k=t_{k+1}-t_k\) receives as many full
`trotter-delta-t` substeps as fit, followed by one shorter positive remainder
when necessary. Every individual substep uses its own system angle
\(2J\,dt\), jump angle \(2\sqrt{\gamma\,dt}\), measurement/feed-forward,
and ancilla reset. Thus no internal step exceeds the requested value and
every circuit lands exactly on its saved time. If the option is omitted,
the backward-compatible behavior is one substep per saved-time interval.

Every listed saved time still requires five measurement circuits; decreasing
`--trotter-delta-t` increases the depth of each circuit rather than the
number of sampling circuits.

## Running on IBM hardware

Pass any accessible dynamic-circuit backend by name. For example:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 7 \
  --backend ibm_kingston \
  --batch-size 5 \
  --shots 8192
```

The backend name is not hard-coded. For example, replacing
`ibm_kingston` with another accessible device changes the target without
changing the circuit code:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend ibm_fez
```

Backend availability depends on the account instance identified by the
CRN. The selected device must have at least \(N+2\) qubits and support
`if_else`, mid-circuit `measure`, and `reset`. IBM documents
`ibm_kingston` as a Heron backend supporting these instructions, but the
script still resolves the backend through the user's account at run time.
See IBM's [backend-information guide](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information).

Specifying a non-`aer` backend submits real QPU work and can consume the
account's allocated runtime. The full default workload is 255 circuits with
8192 shots per circuit, divided into jobs of five circuits by default. A
smaller `--time-points` and `--shots` pilot should be used before a full
hardware run.

## Classical references

Exact Lindblad and coherent unsplit-dilation trajectories are optional:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 7 \
  --backend ibm_kingston \
  --classical-reference
```

Without this flag, the fourth plot panel reports shot-noise standard errors.
With it, the panel instead reports aggregate observable errors relative to
the exact Lindblad and coherent Lie references. Since these reference
calculations scale exponentially, the option should only be used for
classically manageable \(N\).

## Outputs

Each run writes

- a figure to `Model1/Experiment4/figures/`; and
- a JSON record to `Model1/Experiment4/results/`.

The filenames contain the backend name and \(N\), for example
`dynamic_lie_trotter_ibm_kingston_7.png` and
`dynamic_lie_trotter_ibm_kingston_7.json`.

The JSON record contains the physical parameters, time grid, measurement
settings, observable estimates, propagated shot-noise standard errors, raw
counts, provider job IDs, and pre/post-transpilation operation counts and
depths for every circuit. It contains no IBM credentials.

Use `--output-directory` to redirect both output folders:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend aer \
  --output-directory /tmp/experiment4
```

## Checkpoints, recovery, and resume

The hardware script atomically updates a `*_checkpoint.json` file after
every completed batch. A provider failure therefore leaves all earlier raw
counts and job IDs on disk. Resume the same time grid without resubmitting
the completed circuit prefix by adding `--resume`; (N), backend, shots,
time grid, and compiler settings must match, while `--batch-size` may be
changed.

Completed jobs from a run made before checkpointing was enabled can be
downloaded with the read-only recovery utility. First inspect a narrow time
window:

```bash
uv run python Model1/Experiment4/recover_dynamic_lie_trotter.py \
  --n-qubits 4 \
  --backend ibm_kingston \
  --created-after "2026-09-21T18:15:00-04:00" \
  --created-before "2026-09-21T18:38:30-04:00" \
  --list-only
```

Remove `--list-only` to download completed Sampler results and create a
checkpoint, partial JSON record, and partial figure. The recovery utility
never submits or cancels QPU work. Explicit repeated `--job-id` arguments,
in oldest-to-newest order, are the safest option when the time window also
contains unrelated jobs.

Run `--help` for all options:

```bash
uv run python Model1/Experiment4/dynamic_lie_trotter.py --help
```
