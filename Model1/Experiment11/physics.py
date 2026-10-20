"""Immutable model, circuit construction and exact references for Experiment 11.

No other experiment is imported. The shared IBMRuntime circuit vocabulary and
the optional full-reference library are the only repository dependencies.
"""
from dataclasses import dataclass, asdict
from math import ceil, pi

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from IBMRuntime import Circuit, H, Measure, Reset, RX, RZ, RZZ, X, XXPlusYY

BASES = ("Z", "X", "XY")


@dataclass(frozen=True, slots=True)
class Model:
    n_qubits: int = 4
    J: float = 1.0
    h: float = 0.0
    gamma: float = 0.2
    initial_site: int | None = None

    def __post_init__(self):
        if not isinstance(self.n_qubits, (int, np.integer)) or isinstance(self.n_qubits, bool) or self.n_qubits < 2:
            raise ValueError("n_qubits must be an integer >= 2")
        if not all(np.isfinite(v) for v in (self.J, self.h, self.gamma)) or self.gamma < 0:
            raise ValueError("J/h must be finite and gamma finite and nonnegative")
        site = self.n_qubits // 2 if self.initial_site is None else self.initial_site
        if not isinstance(site, (int, np.integer)) or isinstance(site, bool) or not 0 <= site < self.n_qubits:
            raise ValueError("initial_site must lie in [0, n_qubits)")
        object.__setattr__(self, "initial_site", int(site))

    def payload(self):
        return {k: int(v) if k in ("n_qubits", "initial_site") else float(v)
                for k, v in asdict(self).items()}


def validate_times(times):
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("times must be a nonempty finite nonnegative list")
    if np.any(np.diff(values) <= 0):
        raise ValueError("times must be strictly increasing and distinct")
    return values


def schedule(times, trajectories, seed, maximum_dt):
    """Observation intervals split into substeps <= maximum_dt, from t=0.

    The seeded R-by-substep random draw matches the previous strategy. The
    exact requested grid/choices are recorded: changing the grid can change
    substeps and seeded prefixes, particularly for trajectories after row 0.
    """
    values = validate_times(times)
    if trajectories < 1 or seed < 0 or not np.isfinite(maximum_dt) or maximum_dt <= 0:
        raise ValueError("invalid trajectory count, seed or maximum timestep")
    dts, counts = [], []
    previous = 0.0
    for target in values:
        interval = float(target) - previous
        if interval > 0:
            count = max(1, int(ceil(interval / maximum_dt - 1e-12)))
            dts.extend([float(maximum_dt)] * (count - 1))
            dts.append(float(interval - (count - 1) * maximum_dt))
        counts.append(len(dts))
        previous = float(target)
    choices = np.random.default_rng(seed).integers(0, 2, size=(trajectories, len(dts)))
    return tuple(dts), tuple(counts), tuple(tuple(map(int, row)) for row in choices)


def system_qubits(model):
    return tuple(model.n_qubits - site for site in range(model.n_qubits))


def logical_roles(model):
    return ("right boundary ancilla a_R",
            *(f"system site {model.n_qubits - 1 - q}" for q in range(model.n_qubits)),
            "left boundary ancilla a_L")


def step_operations(model, dt, boundary):
    if not np.isfinite(dt) or dt <= 0 or boundary not in (0, 1):
        raise ValueError("dt must be positive and boundary 0 (L) or 1 (R)")
    qubits = system_qubits(model)
    operations = []
    for offset, label in ((0, "H_even"), (1, "H_odd")):
        for site in range(offset, model.n_qubits - 1, 2):
            operations.append(XXPlusYY(2 * model.J * dt, 0.0, qubits[site], qubits[site + 1],
                                       f"{label}[{site},{site+1}]"))
    if model.h != 0:
        operations.extend(RZ(model.h * dt, qubit) for qubit in qubits)
    site = 0 if boundary == 0 else model.n_qubits - 1
    ancilla = model.n_qubits + 1 if boundary == 0 else 0
    operations.extend((XXPlusYY(2 * np.sqrt(2 * model.gamma * dt), 0.0, qubits[site], ancilla,
                                "K_L" if boundary == 0 else "K_R"), Reset(ancilla)))
    return tuple(operations)


def one_step(model, dt, boundary):
    return Circuit(model.n_qubits + 2, 0, step_operations(model, dt, boundary),
                   name=f"step_{'L' if boundary == 0 else 'R'}")


def measurement_operations(model, basis):
    if basis not in BASES:
        raise ValueError("unsupported measurement basis")
    operations = []
    for site, qubit in enumerate(system_qubits(model)):
        # Alternating XY means X on even sites and Y on odd sites.
        if basis == "X" or (basis == "XY" and site % 2 == 0):
            operations.append(H(qubit))
        elif basis == "XY":
            operations.extend((RZ(-pi / 2, qubit), H(qubit)))
    operations.extend(Measure(qubit, site) for site, qubit in enumerate(system_qubits(model)))
    return tuple(operations)


def build_circuit(model, dts, choices, basis="Z"):
    if len(dts) != len(choices):
        raise ValueError("one boundary choice is required per substep")
    operations = [X(system_qubits(model)[model.initial_site])]
    for dt, boundary in zip(dts, choices):
        operations.extend(step_operations(model, dt, boundary))
    operations.extend(measurement_operations(model, basis))
    return Circuit(model.n_qubits + 2, model.n_qubits, tuple(operations), name=f"randomized_{basis}")


def explicit_exchange(angle, first, second):
    """Fixed 14-operation/eight-layer synthesis, up to a global minus sign.

    As in the original strategy, these exchanges are never controlled; this
    phase has no effect on the density operator or measured observables.
    """
    return (RZ(pi/2, first), RZ(-pi/2, second), RX(pi/2, first), RX(-pi/2, second),
            RZ(pi/2, first), RZ(-pi/2, second), RZZ(angle/2, first, second),
            RX(-pi/2, first), RX(-pi/2, second), RZZ(angle/2, first, second),
            RZ(-pi/2, first), RZ(-pi/2, second), RX(-pi/2, first), RX(-pi/2, second))


def prepare_for_compilation(circuit):
    operations = []
    for operation in circuit.operations:
        if isinstance(operation, XXPlusYY):
            if operation.phase != 0:
                raise ValueError("explicit synthesis only supports beta=0")
            operations.extend(explicit_exchange(operation.angle, operation.first, operation.second))
        else:
            operations.append(operation)
    return Circuit(circuit.qubit_count, circuit.bit_count, tuple(operations), name=circuit.name)


def exact_reference(model, times, *, full=False):
    """Exact observables; optimized mode uses only N no-jump amplitudes."""
    values = validate_times(times)
    if full:
        from library import classical
        from scipy.sparse import eye
        Hsys, jumps, Xops, Yops, Zops = classical.build_linear_chain(
            model.n_qubits, model.J, model.h, model.gamma)
        generator = classical.build_liouvillian(Hsys, jumps)
        psi = classical.computational_state(model.n_qubits, [model.initial_site])
        vector = np.outer(psi, psi.conj()).reshape(-1, order="F")
        operators = classical.build_observables(eye(2**model.n_qubits, format="csr"),
                                                Xops, Yops, Zops, model.J)
        output = [np.empty((len(family), len(values))) for family in operators]
        previous = 0.0
        trace = generator.diagonal().sum()
        for index, target in enumerate(values):
            dt = float(target) - previous
            if dt > 0:
                vector = expm_multiply(dt * generator, vector, traceA=dt * trace)
            rho = vector.reshape((2**model.n_qubits,) * 2, order="F")
            for destination, family in zip(output, operators):
                for site, op in enumerate(family):
                    destination[site, index] = np.sum(rho * op.toarray().T).real
            previous = float(target)
        return tuple(output)
    diagonal = np.full(model.n_qubits, 1j * model.h, dtype=complex)
    diagonal[[0, -1]] -= model.gamma / 2
    hopping = np.full(model.n_qubits - 1, -1j * model.J)
    generator = diags((hopping, diagonal, hopping), (-1, 0, 1), format="csr")
    trace = generator.diagonal().sum()
    amplitude = np.zeros(model.n_qubits, dtype=complex)
    amplitude[model.initial_site] = 1
    populations = np.empty((model.n_qubits, len(values)))
    correlations = np.empty((model.n_qubits - 1, len(values)))
    flows = np.empty_like(correlations)
    previous = 0.0
    for index, target in enumerate(values):
        dt = float(target) - previous
        if dt > 0:
            amplitude = expm_multiply(dt * generator, amplitude, traceA=dt * trace)
        coherence = amplitude[:-1] * amplitude[1:].conj()
        populations[:, index] = np.abs(amplitude)**2
        correlations[:, index] = 4 * coherence.real
        flows[:, index] = 2 * model.J * coherence.imag
        previous = float(target)
    return populations, correlations, flows
