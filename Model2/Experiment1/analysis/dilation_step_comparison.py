"""Compare Lie and Strang shared-ancilla dilation against exact XX evolution.

This runner evaluates the same channels as simulation.dilation_kraus. It
exponentiates local jump rotations analytically, then propagates density
matrices in excitation-number blocks. The latter optimization requires the
number-block-diagonal initial states used here; it is not a Gaussian closure.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import LinearOperator, lgmres, splu

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Model2.Experiment1 import simulation
from Model2.Experiment1.analysis.analyze_exact import correlations
from Model2.Experiment1.analysis.channel_powers import channel_powers


def number_sectors(n):
    return [np.asarray([i for i in range(2**n) if i.bit_count() == count])
            for count in range(n + 1)]


def analytical_kraus(H, jumps, dt, formula):
    """Apply the exact factors to the initial ancilla-zero isometry.

    Each boundary jump is a scaled partial isometry with disjoint source
    and destination computational-basis states. Its enlarged exponential
    is therefore a collection of two-dimensional sine/cosine rotations.
    """
    if formula not in {"lie", "strang"} or dt <= 0:
        raise ValueError("require positive dt and formula lie or strang")
    dimension = H.shape[0]
    isometry = np.zeros((len(jumps) + 1, dimension, dimension), complex)
    isometry[0] = np.eye(dimension)
    unitary = expm(-1j * dt * H.toarray())

    def rotate(values, label, fraction):
        jump = jumps[label - 1].tocoo()
        destination, source = jump.row, jump.col
        amplitudes = jump.data
        if (len(np.unique(source)) != len(source) or
                len(np.unique(destination)) != len(destination)):
            raise ValueError("jump must be a scaled partial permutation")
        angle = fraction * np.sqrt(dt) * np.abs(amplitudes)
        phase = amplitudes / np.abs(amplitudes)
        a = values[0, source].copy()
        b = values[label, destination].copy()
        cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
        values[0, source] = cosine * a - 1j * sine * phase.conj()[:, None] * b
        values[label, destination] = cosine * b - 1j * sine * phase[:, None] * a

    if formula == "lie":
        isometry[0] = unitary @ isometry[0]
        for label in range(1, len(jumps) + 1):
            rotate(isometry, label, 1)
    else:
        for label in range(1, len(jumps) + 1):
            rotate(isometry, label, 0.5)
        isometry[0] = unitary @ isometry[0]
        for label in range(len(jumps), 0, -1):
            rotate(isometry, label, 0.5)
    return isometry


def prepare_channel(kraus, sectors):
    transitions = []
    # A final ancilla label fixes the net change in system excitation number.
    shifts = [0, 1, -1, 1, -1]
    for k, shift in enumerate(shifts):
        allowed = np.zeros_like(kraus[k])
        for source, indices in enumerate(sectors):
            destination = source + shift
            if 0 <= destination < len(sectors):
                target = sectors[destination]
                block = np.ascontiguousarray(kraus[k][np.ix_(target, indices)])
                allowed[np.ix_(target, indices)] = block
                transitions.append((source, destination, block, np.ascontiguousarray(block.conj().T)))
        np.testing.assert_allclose(kraus[k], allowed, atol=2e-14, rtol=0)
    return transitions


def apply_channel(blocks, transitions):
    if isinstance(transitions, dict):
        vectors = np.concatenate([block.reshape(len(block), -1) for block in blocks], axis=1).T
        evolved = transitions['operator'] @ vectors
        return [evolved[start:end].T.reshape(len(blocks[0]), size, size)
                for start, end, size in zip(transitions['offsets'][:-1], transitions['offsets'][1:], transitions['sizes'])]
    result = [np.zeros_like(block) for block in blocks]
    for source, destination, K, Kdagger in transitions:
        result[destination] += (K @ blocks[source]) @ Kdagger
    return result


def prepare_sparse_channel(kraus, sectors, *, cutoff=1e-18):
    """Assemble the discrete Kraus channel, with a measured truncation bound.

    The number-block vector uses row-major density entries, hence K tensor
    K.conjugate(). Discarding entries below cutoff accelerates propagation;
    the omitted induced vector 1-norm is recorded. Multiplying that norm by
    system dimension bounds the omitted map's induced trace norm, so N
    repeated steps have error bounded by expm1(N * dimension * omitted_norm).
    """
    transitions = prepare_channel(kraus, sectors)
    sizes = [len(indices) for indices in sectors]
    grouped = {}
    for source, destination, K, _ in transitions:
        grouped.setdefault((destination, source), []).append(K)
    matrix_blocks = [[None for _ in sectors] for _ in sectors]
    discarded_columns = [np.zeros(size**2) for size in sizes]
    for (destination, source), factors in grouped.items():
        block = sum(np.kron(K, K.conj()) for K in factors)
        mask = np.abs(block) < cutoff
        discarded_columns[source] += np.sum(np.where(mask, np.abs(block), 0), axis=0)
        block[mask] = 0
        matrix_blocks[destination][source] = csr_matrix(block)
    return dict(operator=bmat(matrix_blocks, format='csr'), sizes=sizes,
                offsets=np.r_[0, np.cumsum(np.square(sizes))],
                omitted_vector_one_norm=float(max(values.max() for values in discarded_columns)), cutoff=cutoff)


def extract_blocks(density, sectors):
    return [np.ascontiguousarray(density[:, indices[:, None], indices]) for indices in sectors]


def full_density(blocks, sectors):
    density = np.zeros((blocks[0].shape[0], sum(map(len, sectors)), sum(map(len, sectors))), complex)
    for indices, block in zip(sectors, blocks):
        density[:, indices[:, None], indices] = block
    return density


def observables(blocks, sectors, Zdiag, currents):
    Z = sum(np.diagonal(block, axis1=-2, axis2=-1).real @ Zdiag[:, indices].T
            for indices, block in zip(sectors, blocks))
    I = sum(np.einsum("sij,bji->sb", block, current).real
            for block, current in zip(blocks, currents))
    return Z, I


def stationary_preconditioner(H, jumps, sectors, target):
    dimension = H.shape[0]
    indices = np.concatenate([(sector[:, None] + dimension * sector[None, :]).ravel() for sector in sectors])
    generator = simulation.classical.build_liouvillian(H, jumps).tocsr()
    reduced = generator[indices, :][:, indices]
    target_vector = np.concatenate([target[np.ix_(sector, sector)].ravel() for sector in sectors])
    trace_vector = np.concatenate([np.eye(len(sector)).ravel() for sector in sectors])
    correction = csr_matrix(target_vector[:, None]) @ csr_matrix(trace_vector[None, :])
    factor = splu((-reduced + correction).tocsc())
    return LinearOperator(reduced.shape, matvec=factor.solve, dtype=complex)


def stationary_state(transitions, sectors, target, dt, preconditioner=None):
    """Solve the trace-constrained fixed-point equation in the same blocks."""
    shapes = [(1, len(indices), len(indices)) for indices in sectors]
    offsets = np.r_[0, np.cumsum([np.prod(shape) for shape in shapes])]

    def unpack(vector):
        return [vector[start:end].reshape(shape)
                for start, end, shape in zip(offsets[:-1], offsets[1:], shapes)]

    def pack(blocks):
        return np.concatenate([block.ravel() for block in blocks])

    target_vector = pack(extract_blocks(target[None], sectors))

    def action(vector):
        blocks = unpack(vector)
        trace = sum(np.trace(block[0]) for block in blocks)
        return (vector - pack(apply_channel(blocks, transitions))) / dt + trace * target_vector

    operator = LinearOperator((offsets[-1], offsets[-1]), matvec=action, dtype=complex)
    solution, info = lgmres(operator, target_vector, x0=target_vector,
                            rtol=1e-10, atol=1e-12, maxiter=200, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"stationary-state solve did not converge: {info}")
    blocks = unpack(solution)
    residual = np.max(np.abs(solution - pack(apply_channel(blocks, transitions))))
    return full_density(blocks, sectors)[0], float(residual)


def write_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((output / "comparison.json").read_text())
    params, records = data["parameters"], data["records"]
    lines = ["# Lie and Strang dilation: seven-qubit timestep comparison", "",
             f"N={params['number_of_qubits']}, J={params['J']:g}, Gamma={params['gamma']:g}, "
             f"p={params['p']:g}, delta={params['delta']:g}, exchange prefactor={params['exchange_prefactor']:g}, "
             f"final time={data['times'][-1]:g}.", "",
             "Both methods use the existing first-order Ding–Li–Lin shared-ancilla dilation. "
             "Each jump factor is a unitary coupling to its label in the same ancilla. "
             "The ancilla is traced/reset only after the complete step. Strang uses half "
             "jump rotations, the full Hamiltonian factor, and reversed half jump rotations.", "",
             "The criterion is maximum population error <=0.001 and maximum excitation-current "
             "error divided by J <=0.001, across all sites/bonds and all three initial states. "
             "For the recommended step, require population error <=0.0005 as a margin.", "",
             "The common observation grid samples every 0.01 through t=1 and the saved exact "
             "times thereafter. Every actual step through t=0.1 is additionally checked "
             "for both populations and currents. These are sampled observable errors, "
             "not a uniform-in-time, arbitrary-state, or full-density-matrix error bound.", "",
             "| Method | Maximum step | Reset steps | Max population error | Max current error / J | Stationary population bias | Propagation time (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['formula']} | {r['max_dt']:g} | {r['total_steps']} | "
                     f"{r['maximum_population_error']:.6g} | {r['maximum_current_error_divided_by_J']:.6g} | "
                     f"{r['stationary_maximum_population_bias']:.6g} | {r['propagation_seconds']:.2f} |")
    selections = {}
    for formula in ("lie", "strang"):
        passing = [r for r in records if r['formula'] == formula and
                   r['maximum_population_error'] <= 0.001 and
                   r['maximum_current_error_divided_by_J'] <= 0.001]
        conservative = [r for r in passing if r['maximum_population_error'] <= 0.0005]
        selections[formula] = {"largest_passing_tested_step": max((r['max_dt'] for r in passing), default=None),
                               "recommended_tested_step": max((r['max_dt'] for r in conservative), default=None)}
    lines += ["", "Tested step selections:", "",
              "| Method | Largest passing tested step | Tested step with population margin |",
              "|---|---:|---:|"]
    for formula, selection in selections.items():
        lines.append(f"| {formula} | {selection['largest_passing_tested_step']} | {selection['recommended_tested_step']} |")
    common_steps = sorted(set(r['max_dt'] for r in records if r['formula']=='lie') &
                          set(r['max_dt'] for r in records if r['formula']=='strang'))
    if common_steps:
        paired = { (r['formula'],r['max_dt']):r for r in records }
        if all(paired['lie',dt]['maximum_population_error'] < paired['strang',dt]['maximum_population_error'] for dt in common_steps):
            lines += ['', 'For this benchmark, Lie has smaller maximum transient population error at every '
                      'common tested step. Strang does not justify increasing the timestep. '
                      'Keep the tested Lie step 0.00025 for the population margin.', '']
    lines += ["", "The stationary biases come from a trace-constrained fixed-point solve of "
              "each constant-step channel, rather than treating t=12 as infinite time. "
              "Full stationary and final populations/currents, fixed-point residuals, trace, "
              "Hermiticity, minimum eigenvalues, and validation checks are saved in comparison.json.", "",
              "Runtimes use the same number-sector channel-power evaluation for both methods with "
              "one BLAS thread. They exclude channel preparation, exact-reference evaluation, "
              "stationary solves, early direct-iteration checks and reporting. They are classical timings, not measured "
              "compiled circuit costs. Lie has five unitary factors per reset; Strang has nine. "
              "The ancilla resets prevent simply merging jump half-factors across neighboring steps.", "",
              "The exact XX covariance reference agrees with the saved full Liouvillian run. "
              "The optimized Kraus factors are checked against the original enlarged-unitary "
              "implementation, and block propagation is checked against full-density Kraus "
              "propagation after two steps. This is a full density-matrix channel simulation "
              "within invariant number blocks, not an approximate covariance evolution.", "",
              "Sparse channel entries below 1e-18 are omitted. The measured omitted vector "
              "1-norm gives an accumulated trace-norm truncation bound saved for every run; "
              "the runner requires that bound to remain below 1e-8. It concerns this sparsification "
              "only, not all floating-point roundoff or the physical dilation error.", "",
              "The default Krylov propagation evaluates integer powers of the same discrete channel. "
              "Its Arnoldi residual is checked at every integer step to obtain an a posteriori "
              "trace-norm bound below 1e-9 in exact arithmetic, with the tiny sparse-map "
              "norm excess included. Each early step through t=0.1 "
              "is also compared against direct sparse-channel iteration to check numerical "
              "roundoff. These calculations accelerate the classical simulator; the displayed "
              "reset counts still describe the original dilation algorithm. Use --propagation steps "
              "to perform literal repeated channel applications.", "",
              "![Timestep comparison](comparison.png)", "",
              "Reproduce:", "", "```bash",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \\",
              "  Model2/Experiment1/analysis/dilation_step_comparison.py", "```", ""]
    data["step_selections"] = selections
    for formula in ('lie', 'strang'):
        fine = sorted((r for r in records if r['formula'] == formula), key=lambda r:r['max_dt'])[:2]
        if len(fine) == 2:
            order = np.log(fine[1]['maximum_population_error']/fine[0]['maximum_population_error']) / np.log(fine[1]['max_dt']/fine[0]['max_dt'])
            lines += [f"Observed population-error order from the two finest {formula} steps: {order:.4f}.", ""]
    (output / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "Comparison.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for formula in ("lie", "strang"):
        selected = sorted((r for r in records if r['formula'] == formula), key=lambda r:r['max_dt'])
        steps = [r['max_dt'] for r in selected]
        for ax, field in zip(axes, ("maximum_population_error", "maximum_current_error_divided_by_J", "stationary_maximum_population_bias")):
            ax.loglog(steps, [r[field] for r in selected], "o-", label=formula.capitalize())
    for ax, title in zip(axes, ("Population error vs exact", "Excitation current error / J", "Stationary population bias")):
        ax.set_title(title)
        ax.set_xlabel("Physical timestep")
        ax.set_ylabel("Maximum absolute error")
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes[:2]:
        ax.axhline(0.001, color="black", ls="--")
    axes[0].axhline(0.0005, color="gray", ls=":")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", type=Path, default=ROOT / "Model2/Experiment1/results/N7_J10_G20")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.005, 0.002, 0.001, 0.0005, 0.00025])
    parser.add_argument("--formulas", choices=("lie", "strang"), nargs="+", default=["lie", "strang"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "strang_step_selection")
    parser.add_argument("--propagation", choices=('steps', 'krylov'), default='krylov')
    args = parser.parse_args()
    if any(not np.isfinite(dt) or dt <= 0 for dt in args.steps):
        parser.error("steps must be finite and positive")
    path = args.exact_dir / "exact_N7_p0.25_c0.25.json"
    params = json.loads(path.read_text())["parameters"]
    if params["delta"] != 0:
        parser.error("the exact correlation reference requires delta=0")
    n = params["number_of_qubits"]
    H, jumps, Zops, _, Iops, target = simulation.build_model(
        n, params["J"], params["delta"], params["gamma"], params["p"], params["exchange_prefactor"])
    labels, initial = simulation.initial_states(n)
    with np.load(path.with_suffix(".npz")) as saved:
        saved_times = saved["times"].copy()
        reference_Z, reference_I, _ = correlations(params, labels, saved_times)
        reference_errors = dict(Z=float(np.max(np.abs(reference_Z - saved["Z_exact"]))),
                                current=float(np.max(np.abs(reference_I - saved["I_exact"]))))
        np.testing.assert_allclose(reference_Z, saved["Z_exact"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(reference_I, saved["I_exact"], atol=1e-11, rtol=0)
    times = np.unique(np.round(np.r_[np.arange(0, min(1, saved_times[-1]) + 0.0001, 0.01), saved_times], 12))
    exact_Z, exact_I, _ = correlations(params, labels, times)
    sectors = number_sectors(n)
    Zdiag = np.asarray([op.diagonal().real for op in Zops])
    currents = [np.asarray([op.toarray()[np.ix_(indices, indices)] for op in Iops]) for indices in sectors]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Preparing stationary-state preconditioner', flush=True)
    preconditioner = stationary_preconditioner(H, jumps, sectors, target)
    records, validations = [], {}
    for formula in args.formulas:
        dt = args.steps[0]
        kraus = analytical_kraus(H, jumps, dt, formula)
        original = simulation.dilation_kraus(H, jumps, dt, formula)
        np.testing.assert_allclose(kraus, original, atol=3e-13, rtol=0)
        transitions = prepare_sparse_channel(kraus, sectors)
        blocks = extract_blocks(initial, sectors)
        density = initial.copy()
        for _ in range(2):
            blocks = apply_channel(blocks, transitions)
            density = sum(K @ density @ K.conj().T for K in original)
        block_error = np.max(np.abs(full_density(blocks, sectors) - density))
        np.testing.assert_allclose(full_density(blocks, sectors), density, atol=4e-13, rtol=0)
        validations[formula] = dict(kraus_agreement=float(np.max(np.abs(kraus-original))),
                                   two_step_density_agreement=float(block_error))
        for max_dt in args.steps:
            print(f"Starting {formula}, dt={max_dt:g}", flush=True)
            cache = {}
            blocks = extract_blocks(initial, sectors)
            Z, I = np.empty_like(exact_Z), np.empty_like(exact_I)
            previous, total_steps = 0.0, 0
            early_times, early_Z, early_I = [], [], []
            preparation_seconds, propagation_seconds, truncation_budget = 0.0, 0.0, 0.0
            power_diagnostics = None
            if args.propagation == 'krylov':
                steps_at_times = np.rint(times / max_dt).astype(int)
                if np.max(np.abs(steps_at_times * max_dt - times)) > 1e-10:
                    parser.error('Krylov propagation requires observation times on the constant-step grid')
                started = time.monotonic()
                K = analytical_kraus(H, jumps, max_dt, formula)
                prepared = prepare_sparse_channel(K, sectors)
                preparation_seconds += time.monotonic() - started
                early_steps = np.arange(1, int(np.floor(0.1/max_dt + 1e-10))+1)
                requested = np.unique(np.r_[steps_at_times, early_steps])
                vectors = np.concatenate([block.reshape(len(initial), -1) for block in blocks], axis=1)
                started = time.monotonic()
                powered, power_diagnostics = channel_powers(prepared['operator'], vectors, requested, max_dt, sectors,
                    trace_norm_excess=(2**n)*prepared['omitted_vector_one_norm'])
                propagation_seconds = time.monotonic() - started
                direct = vectors.T.copy()
                direct_error = 0.0
                for step in early_steps:
                    direct = prepared['operator'] @ direct
                    direct_error = max(direct_error, float(np.max(np.abs(direct.T-powered[:, np.searchsorted(requested, step)]))))
                if direct_error > 1e-10:
                    raise RuntimeError(f'Krylov/direct early-step disagreement: {direct_error:g}')
                power_diagnostics['early_direct_entry_agreement'] = direct_error
                def blocks_at(step):
                    values = powered[:, np.searchsorted(requested, step)]
                    return [values[:, first:last].reshape(len(initial), size, size)
                            for first, last, size in zip(prepared['offsets'][:-1], prepared['offsets'][1:], prepared['sizes'])]
                for index, step in enumerate(steps_at_times):
                    Z[:, index], I[:, index] = observables(blocks_at(step), sectors, Zdiag, currents)
                for step in early_steps:
                    z, current = observables(blocks_at(step), sectors, Zdiag, currents)
                    early_times.append(float(step*max_dt))
                    early_Z.append(z)
                    early_I.append(current)
                blocks = blocks_at(steps_at_times[-1])
                total_steps = int(steps_at_times[-1])
                truncation_budget = total_steps * (2**n) * prepared['omitted_vector_one_norm']
            for index, t in enumerate(times if args.propagation == 'steps' else []):
                interval = t - previous
                if interval > 0:
                    count = max(1, int(np.ceil(interval / max_dt - 1e-10)))
                    dt = interval / count
                    key = round(dt, 14)
                    if key not in cache:
                        started = time.monotonic()
                        K = analytical_kraus(H, jumps, dt, formula)
                        np.testing.assert_allclose(sum(k.conj().T @ k for k in K), np.eye(2**n), atol=3e-13, rtol=0)
                        cache[key] = prepare_sparse_channel(K, sectors)
                        preparation_seconds += time.monotonic() - started
                    started = time.monotonic()
                    for step in range(count):
                        blocks = apply_channel(blocks, cache[key])
                        actual_t = previous + (step + 1) * dt
                        if actual_t <= 0.1 + 1e-12:
                            z, current = observables(blocks, sectors, Zdiag, currents)
                            early_times.append(actual_t)
                            early_Z.append(z)
                            early_I.append(current)
                    propagation_seconds += time.monotonic() - started
                    truncation_budget += count * (2**n) * cache[key]['omitted_vector_one_norm']
                    total_steps += count
                Z[:, index], I[:, index] = observables(blocks, sectors, Zdiag, currents)
                previous = t
            density = full_density(blocks, sectors)
            early_times = np.asarray(early_times)
            early_Z = np.stack(early_Z, axis=1)
            early_I = np.stack(early_I, axis=1)
            early_exact_Z, early_exact_I, _ = correlations(params, labels, early_times)
            error, early_error = np.abs(Z - exact_Z)/2, np.abs(early_Z - early_exact_Z)/2
            if early_error.max() > error.max():
                index = np.unravel_index(np.argmax(early_error), early_error.shape)
                worst = dict(state=labels[index[0]], time=float(early_times[index[1]]), site=int(index[2]+1))
            else:
                index = np.unravel_index(np.argmax(error), error.shape)
                worst = dict(state=labels[index[0]], time=float(times[index[1]]), site=int(index[2]+1))
            max_I_error = max(np.max(np.abs(I-exact_I)), np.max(np.abs(early_I-early_exact_I)))
            print(f'Propagation complete: population error={max(error.max(), early_error.max()):.6g}; solving fixed point', flush=True)
            stationary_K = analytical_kraus(H, jumps, max_dt, formula)
            stationary_prepared = prepare_sparse_channel(stationary_K, sectors)
            stationary, residual = stationary_state(stationary_prepared, sectors, target, max_dt, preconditioner)
            truncation_bound = float(np.expm1(truncation_budget))
            if truncation_bound > 1e-8:
                raise RuntimeError(f"sparse channel truncation exceeds numerical budget: {truncation_bound}")
            stationary_Z = stationary.diagonal().real @ Zdiag.T
            stationary_I = np.asarray([np.trace(op.toarray() @ stationary).real for op in Iops])
            record = dict(formula=formula, max_dt=max_dt, total_steps=total_steps,
                          maximum_population_error=float(max(error.max(), early_error.max())),
                          worst_population_error=worst,
                          maximum_current_error=float(max_I_error),
                          maximum_current_error_divided_by_J=float(max_I_error/abs(params['J'])),
                          final_maximum_population_error_vs_exact=float(error[:, -1].max()),
                          final_maximum_population_deviation_from_target=float(np.max(np.abs((1-Z[:, -1])/2-params['p']))),
                          final_populations=((1-Z[:, -1])/2).tolist(), final_currents=I[:, -1].tolist(),
                          stationary_populations=((1-stationary_Z)/2).tolist(), stationary_currents=stationary_I.tolist(),
                          stationary_maximum_population_bias=float(np.max(np.abs((1-stationary_Z)/2-params['p']))),
                          stationary_channel_residual=residual,
                          sparse_operator_nonzeros=int(stationary_prepared['operator'].nnz),
                          sparse_channel_entry_cutoff=stationary_prepared['cutoff'],
                          accumulated_sparse_truncation_trace_norm_bound=truncation_bound,
                          stationary_trace_error=float(abs(np.trace(stationary)-1)),
                          stationary_minimum_eigenvalue=float(np.linalg.eigvalsh(stationary).min()),
                          final_trace_error=float(np.max(np.abs(np.trace(density, axis1=-2, axis2=-1)-1))),
                          final_hermiticity_error=float(np.max(np.abs(density-density.conj().swapaxes(-1,-2)))),
                          final_minimum_eigenvalue=float(np.linalg.eigvalsh(density).min()),
                          preparation_seconds=preparation_seconds, propagation_seconds=propagation_seconds)
            record['propagation_method'] = args.propagation
            record['channel_power_validation'] = power_diagnostics
            records.append(record)
            np.savez_compressed(args.output_dir / f"{formula}_dt{max_dt:g}.npz", times=times,
                                Z=Z, populations=(1-Z)/2, current=I, Z_exact=exact_Z, current_exact=exact_I,
                                early_times=early_times, early_Z=early_Z, early_current=early_I,
                                early_Z_exact=early_exact_Z, early_current_exact=early_exact_I,
                                final_rho=density, stationary_rho=stationary)
            result = dict(parameters=params, initial_states=labels, times=times.tolist(),
                          exact_reference_agreement=reference_errors, channel_validation=validations, records=records)
            (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(record), flush=True)
    write_report(args.output_dir)


if __name__ == "__main__":
    main()
