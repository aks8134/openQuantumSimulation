import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply


# ============================================================
# Single-qubit operators
# ============================================================

def get_single_qubit_operators():
    return(
        I2 := eye(2, dtype=complex, format="csr"),

        X := csr_matrix(
            [[0, 1],
            [1, 0]],
            dtype=complex,
        ),

        Y := csr_matrix(
            [[0, -1j],
            [1j, 0]],
            dtype=complex,
        ),

        Z := csr_matrix(
            [[1, 0],
            [0, -1]],
            dtype=complex,
        ),

        # sigma^- = |0><1|
        SIGMA_MINUS := csr_matrix(
            [[0, 1],
            [0, 0]],
            dtype=complex,
        )
    )

I2, X, Y, Z, SIGMA_MINUS = get_single_qubit_operators()
# ============================================================
# Many-qubit operators
# ============================================================

def operator_on_site(operator, site, number_of_qubits):
    """
    Place a single-qubit operator on one site.

    Tensor-product ordering:
        site 0 ⊗ site 1 ⊗ ... ⊗ site N-1
    """
    if not 0 <= site < number_of_qubits:
        raise ValueError(f"Invalid site: {site}")

    result = csr_matrix([[1.0]], dtype=complex)

    for q in range(number_of_qubits):
        local_operator = operator if q == site else I2
        result = kron(result, local_operator, format="csr")

    return result


def build_linear_chain(number_of_qubits, J, h, gamma):
    """
    Construct the open-chain Hamiltonian

        H = J/2 sum_i (X_i X_{i+1} + Y_i Y_{i+1})
            + h/2 sum_i Z_i

    and amplitude-damping jump operators at the two boundaries

        V_left  = sqrt(gamma) sigma^-_0,
        V_right = sqrt(gamma) sigma^-_{N-1}.

    The interior sites have no jump operators.
    """
    if number_of_qubits < 2:
        raise ValueError("Use at least two qubits.")

    dimension = 2**number_of_qubits

    X_ops = [
        operator_on_site(X, i, number_of_qubits)
        for i in range(number_of_qubits)
    ]

    Y_ops = [
        operator_on_site(Y, i, number_of_qubits)
        for i in range(number_of_qubits)
    ]

    Z_ops = [
        operator_on_site(Z, i, number_of_qubits)
        for i in range(number_of_qubits)
    ]

    H = csr_matrix((dimension, dimension), dtype=complex)

    # Nearest-neighbor exchange terms
    for i in range(number_of_qubits - 1):
        H += 0.5 * J * (
            X_ops[i] @ X_ops[i + 1]
            + Y_ops[i] @ Y_ops[i + 1]
        )

    # Uniform Z field
    for Z_i in Z_ops:
        H += 0.5 * h * Z_i

    jump_operators = [
        np.sqrt(gamma)
        * operator_on_site(SIGMA_MINUS, site, number_of_qubits)
        for site in (0, number_of_qubits - 1)
    ]

    return H, jump_operators, X_ops, Y_ops, Z_ops


# ============================================================
# Lindblad generator
# ============================================================

def build_liouvillian(H, jump_operators):
    """
    Construct the sparse Liouvillian satisfying

        d vec(rho)/dt = L vec(rho),

    using column-major vectorization.
    """
    dimension = H.shape[0]
    identity = eye(dimension, dtype=complex, format="csr")

    # Hamiltonian contribution: -i[H, rho]
    liouvillian = -1j * (
        kron(identity, H, format="csr")
        - kron(H.T, identity, format="csr")
    )

    # Dissipative contributions
    for V in jump_operators:
        V_dagger_V = V.getH() @ V

        liouvillian += (
            kron(V.conjugate(), V, format="csr")
            - 0.5 * kron(identity, V_dagger_V, format="csr")
            - 0.5 * kron(V_dagger_V.T, identity, format="csr")
        )

    return liouvillian.tocsr()


# ============================================================
# Exact and split time evolution
# ============================================================

def build_lie_trotter_liouvillians(H, jump_operators):
    """Return the Hamiltonian and individual dissipative generators.

    The system generator contains only ``-1j * [H, rho]``. Each entry
    in ``jump_liouvillians`` contains the complete dissipator associated
    with exactly one jump operator.
    """
    system_liouvillian = build_liouvillian(H, [])

    zero_hamiltonian = csr_matrix(H.shape, dtype=complex)
    jump_liouvillians = [
        build_liouvillian(zero_hamiltonian, [jump_operator])
        for jump_operator in jump_operators
    ]

    return system_liouvillian, jump_liouvillians


def lie_trotter_evolve(
    initial_density_matrix,
    system_liouvillian,
    jump_liouvillians,
    times,
    *,
    steps_per_interval=1,
):
    """Evolve a density matrix with first-order Lie--Trotter splitting.

    For each substep, apply the system propagator first and then each
    individual jump propagator. The returned array has shape
    ``(len(times), dimension, dimension)``.
    """
    times = np.asarray(times, dtype=float)
    initial_density_matrix = np.asarray(
        initial_density_matrix,
        dtype=complex,
    )

    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if (
        not isinstance(steps_per_interval, (int, np.integer))
        or steps_per_interval < 1
    ):
        raise ValueError("steps_per_interval must be a positive integer")
    if (
        initial_density_matrix.ndim != 2
        or initial_density_matrix.shape[0]
        != initial_density_matrix.shape[1]
    ):
        raise ValueError("initial_density_matrix must be square")

    dimension = initial_density_matrix.shape[0]
    vector_dimension = dimension**2
    all_liouvillians = [system_liouvillian, *jump_liouvillians]

    for liouvillian in all_liouvillians:
        if liouvillian.shape != (vector_dimension, vector_dimension):
            raise ValueError(
                "Every Liouvillian must act on the vectorized "
                "initial density matrix"
            )

    evolved_density_matrices = np.empty(
        (times.size, dimension, dimension),
        dtype=complex,
    )
    evolved_density_matrices[0] = initial_density_matrix

    density_vector = initial_density_matrix.reshape(-1, order="F")
    generator_traces = [
        liouvillian.diagonal().sum()
        for liouvillian in all_liouvillians
    ]

    for time_index, interval in enumerate(np.diff(times), start=1):
        dt = interval / steps_per_interval

        for _ in range(steps_per_interval):
            for liouvillian, generator_trace in zip(
                all_liouvillians,
                generator_traces,
            ):
                density_vector = expm_multiply(
                    dt * liouvillian,
                    density_vector,
                    traceA=dt * generator_trace,
                )

        evolved_density_matrices[time_index] = density_vector.reshape(
            dimension,
            dimension,
            order="F",
        )

    return evolved_density_matrices


def strang_trotter_evolve(
    initial_density_matrix,
    system_liouvillian,
    jump_liouvillians,
    times,
    *,
    steps_per_interval=1,
):
    """Evolve a density matrix with second-order Strang splitting.

    One substep applies the jump generators as half steps in forward
    order, the system generator as a full step in the middle, and the
    jump generators as half steps in reverse order. For two jumps this is

        exp(dt/2 L_jump[0])
        exp(dt/2 L_jump[1])
        exp(dt   L_system)
        exp(dt/2 L_jump[1])
        exp(dt/2 L_jump[0])

    in the order in which the propagators act on the density matrix.
    The returned array has shape
    ``(len(times), dimension, dimension)``.
    """
    times = np.asarray(times, dtype=float)
    initial_density_matrix = np.asarray(
        initial_density_matrix,
        dtype=complex,
    )

    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if (
        not isinstance(steps_per_interval, (int, np.integer))
        or steps_per_interval < 1
    ):
        raise ValueError("steps_per_interval must be a positive integer")
    if (
        initial_density_matrix.ndim != 2
        or initial_density_matrix.shape[0]
        != initial_density_matrix.shape[1]
    ):
        raise ValueError("initial_density_matrix must be square")

    dimension = initial_density_matrix.shape[0]
    vector_dimension = dimension**2
    all_liouvillians = [system_liouvillian, *jump_liouvillians]

    for liouvillian in all_liouvillians:
        if liouvillian.shape != (vector_dimension, vector_dimension):
            raise ValueError(
                "Every Liouvillian must act on the vectorized "
                "initial density matrix"
            )

    evolved_density_matrices = np.empty(
        (times.size, dimension, dimension),
        dtype=complex,
    )
    evolved_density_matrices[0] = initial_density_matrix

    density_vector = initial_density_matrix.reshape(-1, order="F")
    generator_traces = [
        liouvillian.diagonal().sum()
        for liouvillian in all_liouvillians
    ]

    system_generator_trace = generator_traces[0]
    forward_jump_half_steps = list(zip(
        jump_liouvillians,
        generator_traces[1:],
    ))
    reverse_jump_half_steps = tuple(
        reversed(forward_jump_half_steps)
    )

    for time_index, interval in enumerate(np.diff(times), start=1):
        dt = interval / steps_per_interval

        for _ in range(steps_per_interval):
            for liouvillian, generator_trace in forward_jump_half_steps:
                density_vector = expm_multiply(
                    0.5 * dt * liouvillian,
                    density_vector,
                    traceA=0.5 * dt * generator_trace,
                )

            density_vector = expm_multiply(
                dt * system_liouvillian,
                density_vector,
                traceA=dt * system_generator_trace,
            )

            for liouvillian, generator_trace in reverse_jump_half_steps:
                density_vector = expm_multiply(
                    0.5 * dt * liouvillian,
                    density_vector,
                    traceA=0.5 * dt * generator_trace,
                )

        evolved_density_matrices[time_index] = density_vector.reshape(
            dimension,
            dimension,
            order="F",
        )

    return evolved_density_matrices


def exact_evolve(initial_density_matrix, liouvillian, times):
    """Evaluate unsplit Liouvillian evolution on a uniform time grid."""
    times = np.asarray(times, dtype=float)
    initial_density_matrix = np.asarray(
        initial_density_matrix,
        dtype=complex,
    )

    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if times.size > 2 and not np.allclose(
        np.diff(times),
        times[1] - times[0],
    ):
        raise ValueError("exact evolution requires a uniform time grid")
    if (
        initial_density_matrix.ndim != 2
        or initial_density_matrix.shape[0]
        != initial_density_matrix.shape[1]
    ):
        raise ValueError("initial_density_matrix must be square")

    dimension = initial_density_matrix.shape[0]
    vector_dimension = dimension**2
    if liouvillian.shape != (vector_dimension, vector_dimension):
        raise ValueError(
            "liouvillian must act on the vectorized initial density matrix"
        )

    if times.size == 1:
        return initial_density_matrix[np.newaxis, :, :].copy()

    initial_vector = initial_density_matrix.reshape(-1, order="F")
    evolved_vectors = expm_multiply(
        liouvillian,
        initial_vector,
        start=0.0,
        stop=times[-1] - times[0],
        num=times.size,
        endpoint=True,
        traceA=liouvillian.diagonal().sum(),
    )

    return np.asarray([
        vector.reshape(dimension, dimension, order="F")
        for vector in evolved_vectors
    ])


# ============================================================
# Initial state
# ============================================================

def computational_state(number_of_qubits, excited_sites):
    """
    Return a computational-basis state with |1> on excited_sites
    and |0> on every other site.
    """
    excited_sites = set(excited_sites)

    if any(
        site < 0 or site >= number_of_qubits
        for site in excited_sites
    ):
        raise ValueError("An excited site lies outside the chain.")

    basis_index = 0

    for site in range(number_of_qubits):
        bit = int(site in excited_sites)
        basis_index = (basis_index << 1) | bit

    state = np.zeros(2**number_of_qubits, dtype=complex)
    state[basis_index] = 1.0

    return state


# ============================================================
# Observable utilities
# ============================================================

def expectation_series(density_matrices, operator):
    """
    Compute Tr[rho(t) operator] at every requested time.
    """
    operator_dense = operator.toarray()

    values = np.einsum(
        "tij,ji->t",
        density_matrices,
        operator_dense,
    )

    return np.real_if_close(values).real


def build_observables(identity, X_ops, Y_ops, Z_ops, J):
    """
    Construct site and nearest-neighbor bond observables.
    """
    number_of_qubits = len(Z_ops)

    # n_i = (I - Z_i)/2
    population_operators = [
        0.5 * (identity - Z_ops[i])
        for i in range(number_of_qubits)
    ]

    # C_i^XY = X_i X_{i+1} + Y_i Y_{i+1}
    exchange_operators = [
        X_ops[i] @ X_ops[i + 1]
        + Y_ops[i] @ Y_ops[i + 1]
        for i in range(number_of_qubits - 1)
    ]

    # Current oriented from site i toward site i+1.
    #
    # This normalization gives the Hamiltonian transport contribution
    #
    #   dn_i/dt = I_{i-1 -> i} - I_{i -> i+1} + dissipative terms.
    #
    # For the boundary-damped chain, the loss term -gamma*n_i is
    # present only for i = 0 and i = N - 1.
    #
    # I_{i -> i+1}
    #   = -J/2 <X_i Y_{i+1} - Y_i X_{i+1}>
    flow_operators = [
        -0.5 * J * (
            X_ops[i] @ Y_ops[i + 1]
            - Y_ops[i] @ X_ops[i + 1]
        )
        for i in range(number_of_qubits - 1)
    ]

    return (
        population_operators,
        exchange_operators,
        flow_operators,
    )
