"""Correlated ideal Z and alternating XY/YX readout of Gaussian path states.

The covariance argument is the real gauge used by Experiments 1 and 2.
No circuit, statevector or IBM dependency is used by the production sampler.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from FinalModel.Experiment1 import phoenix_resources as p
from FinalModel.Experiment2 import sampling as trajectories

DIRECTORY = Path(__file__).resolve().parent
BASES = ("z", "xy", "yx")
FLAGS = trajectories.FLAGS


def majorana(covariance):
    """Gamma_ab=<i gamma_a gamma_b>, gamma_(2j)=Z_<j X_j.

    <a_i^dagger a_j>=i^i (-i)^j C_real[i,j].  Vacuum has Gamma[2j,2j+1]=-1.
    """
    C = np.asarray(covariance, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T, atol=1e-11):
        raise ValueError("Expected a real symmetric covariance")
    n = len(C)
    phase = (1j)**np.arange(n)
    physical = phase[:, None]*phase.conj()[None, :]*C
    result = np.zeros((2*n, 2*n))
    result[::2, ::2] = result[1::2, 1::2] = -2*physical.imag
    result[::2, 1::2] = 2*physical.real-np.eye(n)
    result[1::2, ::2] = -result[::2, 1::2].T
    return result


def measured_pairs(n, basis):
    if n < 2 or basis not in BASES:
        raise ValueError("Expected N>=2 and z, xy or yx readout")
    if basis == "z":
        return [(2*j, 2*j+1, -1) for j in range(n)]
    result = []
    for j in range(n-1):
        # X_j Y_(j+1)=-i gamma_(2j+1) gamma_(2j+3).
        # Y_j X_(j+1)=+i gamma_(2j) gamma_(2j+2).
        xy = basis[j % 2] == "x"
        result.append((2*j+1, 2*j+3, -1) if xy else (2*j, 2*j+2, 1))
    return result


def ordered_covariance(covariance, basis):
    n = len(covariance)
    pairs = measured_pairs(n, basis)
    measured = [value for a, b, _ in pairs for value in (a, b)]
    if len(set(measured)) != len(measured):
        raise ValueError("Readout pair mapping is not a disjoint Majorana matching")
    order = measured+[j for j in range(2*n) if j not in measured]
    G = majorana(covariance)
    return np.ascontiguousarray(G[np.ix_(order, order)]), np.array([sign for _, _, sign in pairs], dtype=np.int8)


def conditional(G, sign):
    """Measure i gamma_0 gamma_1=sign and discard that pair."""
    if sign not in (-1, 1):
        raise ValueError("A measured parity is +/-1")
    denominator = 1+sign*G[0, 1]
    if denominator <= 0:
        return 0., G[2:, 2:].copy()
    rest = G[2:, 2:].copy()
    rest += sign/denominator*(np.outer(G[2:, 0], G[1, 2:])-np.outer(G[2:, 1], G[0, 2:]))
    rest = (rest-rest.T)/2
    return denominator/2, rest


def exact_probabilities(covariance, basis):
    """Enumerate small-system joint bitstring probabilities for validation only."""
    n = len(covariance)
    if n > 10:
        raise ValueError("Exact enumeration is restricted to validation systems")
    G, signs = ordered_covariance(covariance, basis)
    probabilities = np.zeros(1 << n)

    def visit(state, depth, features, weight):
        if depth == len(signs):
            if basis == "z":
                packed = sum(int(value == -1) << j for j, value in enumerate(features))
                probabilities[packed] += weight
            else:
                # Parity-even states give equal probability to globally flipped X/Y outcomes.
                for first in (-1, 1):
                    values = [first]
                    for parity in features:
                        values.append(values[-1]*parity)
                    packed = sum(int(value == -1) << j for j, value in enumerate(values))
                    probabilities[packed] += weight/2
            return
        for sign in (-1, 1):
            probability, remaining = conditional(state, sign)
            if probability > 1e-15:
                visit(remaining, depth+1, features+[sign*int(signs[depth])], weight*probability)

    visit(G, 0, [], 1.)
    return probabilities


@lru_cache(maxsize=1)
def build_kernel():
    source = DIRECTORY/"readout.cpp"
    compiler = subprocess.run(["/usr/bin/g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    included = [DIRECTORY.parent/"Experiment2/trajectory.cpp", DIRECTORY.parent/"Experiment1/channel.cpp"]
    includes = {str(path.relative_to(ROOT)): p.file_digest(path) for path in included}
    identity = p.digest(dict(source=p.file_digest(source), includes=includes, flags=FLAGS, compiler=compiler))
    directory = ROOT/".local-artifacts/readout-kernel"/identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory/"readout.so"
    if not binary.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".so", delete=False) as stream:
            temporary = Path(stream.name)
        subprocess.run(["/usr/bin/g++", *FLAGS, str(source), "-o", str(temporary)], check=True)
        temporary.replace(binary)
    return binary, dict(source_sha256=p.file_digest(source), included_sources=includes, binary_sha256=p.file_digest(binary),
                        flags=FLAGS, compiler=compiler)


@lru_cache(maxsize=1)
def kernel():
    binary, _ = build_kernel()
    lib = ctypes.CDLL(str(binary))
    i, u = ctypes.c_int, ctypes.c_uint64
    a = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    out = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
    lib.gaussian_sample.argtypes = [a, i, i, i, u, u, i, out]
    lib.gaussian_sample.restype = i
    choices = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.trajectory_snapshot.argtypes = [i]+[ctypes.c_double]*5+[a, a, i, i, u, u, i, a, choices]
    lib.trajectory_snapshot.restype = None
    return lib


def sample_features(covariance, basis, shots, seed, *, first=0, threads=1):
    G, signs = ordered_covariance(covariance, basis)
    if shots < 1 or seed < 0 or first < 0 or threads < 1:
        raise ValueError("Invalid readout sampling arguments")
    result = np.empty((shots, len(signs)), dtype=np.int8)
    code = kernel().gaussian_sample(G, len(G), len(signs), shots, seed, first, threads, result)
    if code:
        raise ValueError("Nonphysical conditional readout probability")
    result *= signs
    return result


def conditional_shot_variance(populations, currents, prefactor=.5):
    """One shot per each required basis; current combines independent XY/YX settings."""
    populations, currents = np.asarray(populations), np.asarray(currents)
    return np.concatenate([populations*(1-populations), 2*prefactor**2-currents**2/2], axis=-1)


def path_state(case, seed=29, path_index=0, continuation=0):
    if min(seed, path_index, continuation) < 0:
        raise ValueError("Invalid trajectory snapshot arguments")
    spec, n = case["physics"], case["N"]
    dts = np.ascontiguousarray(p.steps(case), dtype=np.float64)
    initial = np.ascontiguousarray(trajectories.dynamics.initial(n, spec["preparation"]))
    rng_seed = int(trajectories.physical_identity(case)[:16], 16)^seed
    rank_one = spec["preparation"] == "center_excitation" and spec["p_left"] == spec["p_right"] == 0
    covariance = np.empty((n, n))
    choices = np.empty(len(dts)+continuation, dtype=np.int32)
    kernel().trajectory_snapshot(n, spec["prefactor"], spec["J"], spec["gamma"], spec["p_left"],
        spec["p_right"], initial, dts, len(dts), continuation, rng_seed, path_index,
        int(rank_one), covariance, choices)
    return covariance, choices
