"""Measure population errors of the existing shared-ancilla Lie/Strang maps.

Uses excitation-number blocks of density matrices. No Gaussian approximation
is made to the dilation; exact reference populations use XX correlations.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eig, expm, solve_continuous_lyapunov
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from library import classical
from Model2.Experiment1 import simulation as engines
from Model3.Experiment1 import simulation as model


def build_kraus(n, dt, formula):
    H, _, *_ = engines.build_model(n, exchange_prefactor=0.5)
    jumps = []
    for site, p in ((0, 0.9), (n - 1, 0.1)):
        loss = classical.operator_on_site(classical.SIGMA_MINUS, site, n)
        jumps.extend((np.sqrt(2 * p) * loss.getH(), np.sqrt(2 * (1-p)) * loss))
    _, terms = classical.build_hamiltonian_dilation_terms(H, jumps, dt)
    d = 2**n
    iso = np.zeros((5*d, d), complex)
    iso[:d] = np.eye(d)
    factors = ([(0, 1)] + [(k, 1) for k in range(1, 5)] if formula == "lie" else
               [(k, 0.5) for k in range(1, 5)] + [(0, 1)] + [(k, 0.5) for k in range(4, 0, -1)])
    U = expm(-1j * dt * H.toarray())
    for k, scale in factors:
        if k == 0:
            iso[:d] = U @ iso[:d]
        else:
            G = scale * terms[k-1]
            theta = float(np.max(np.abs(G.data)))
            v = G @ iso
            # Each individual jump generator obeys G^3=theta^2 G.
            iso = iso - 1j * np.sin(theta)/theta * v + (np.cos(theta)-1)/theta**2 * (G @ v)
    return iso.reshape(5, d, d)


class BlockChannel:
    def __init__(self, n, dt, formula):
        self.n = n
        weights = np.asarray([int(k).bit_count() for k in range(2**n)])
        self.sectors = [np.flatnonzero(weights == k) for k in range(n+1)]
        self.sizes = [len(s) for s in self.sectors]
        kraus = build_kraus(n, dt, formula)
        self.maps = []
        for k, shift in enumerate((0, 1, -1, 1, -1)):
            for source in range(n+1):
                target = source + shift
                if 0 <= target <= n:
                    K = kraus[k][np.ix_(self.sectors[target], self.sectors[source])]
                    self.maps.append((source, target, K, K.conj().T))
        self.dimension = sum(s*s for s in self.sizes)
        self.populations = [np.asarray([[int(bit) for bit in f"{k:0{n}b}"] for k in sector])
                            for sector in self.sectors]

    def apply(self, blocks):
        output = [np.zeros_like(block) for block in blocks]
        for source, target, K, adjoint in self.maps:
            output[target] += K @ blocks[source] @ adjoint
        return output

    def pack(self, blocks):
        return np.concatenate([b.ravel() for b in blocks])

    def unpack(self, vector):
        blocks = []
        offset = 0
        for s in self.sizes:
            blocks.append(vector[offset:offset+s*s].reshape(s,s))
            offset += s*s
        return blocks

    def expectations(self, blocks):
        return sum(np.diagonal(b, axis1=-2, axis2=-1).real @ pops
                   for b, pops in zip(blocks, self.populations))

    def stationary(self, dt):
        # Trace constraint in a rank-one augmentation of (Id-Phi)/dt.
        anchor = [np.zeros((s,s),complex) for s in self.sizes]
        anchor[0][0,0] = 1
        packed_anchor = self.pack(anchor)
        def matvec(vector):
            blocks = self.unpack(vector)
            trace = sum(np.trace(b) for b in blocks)
            return (vector - self.pack(self.apply(blocks)))/dt + trace*packed_anchor
        operator = LinearOperator((self.dimension, self.dimension), matvec=matvec, dtype=complex)
        solution, info = gmres(operator, packed_anchor, rtol=1e-10, atol=1e-12,
                               restart=150, maxiter=30)
        if info != 0:
            raise RuntimeError(f"Stationary GMRES failed with info={info}")
        blocks = self.unpack(solution)
        residual = float(np.linalg.norm(self.pack(self.apply(blocks))-solution))
        return self.expectations(blocks), residual


def exact_populations(n, times):
    labels = ("0"*n, "1"*n, "1"+"0"*(n-1))
    # Use eigendecomposition to evaluate arbitrary times in a single batch.
    h = np.diag(np.ones(n-1),1) + np.diag(np.ones(n-1),-1)
    rates = np.zeros(n); rates[[0,-1]] = 2
    A = -1j*h - np.diag(rates)/2
    eigenvalues, V = eig(A)
    inverse = np.linalg.inv(V)
    injection = np.zeros(n); injection[[0,-1]] = (1.8,0.2)
    Cstar = solve_continuous_lyapunov(A, -np.diag(injection))
    initial = np.asarray([np.diag([int(bit) for bit in label]) for label in labels])
    transformed = inverse @ (initial-Cstar) @ inverse.conj().T
    coefficients = np.einsum("ja,sab,jb->sjab", V, transformed, V.conj())
    exponentials = np.exp(times[:,None,None]*(eigenvalues[:,None]+eigenvalues.conj()[None,:]))
    return np.diag(Cstar).real + np.einsum("sjab,tab->stj", coefficients, exponentials).real


def measure(n, dt, formula, t_final=20, sample_interval=0.02):
    started = time.perf_counter()
    channel = BlockChannel(n, dt, formula)
    blocks = [np.zeros((3,s,s),complex) for s in channel.sizes]
    for state, index in enumerate((0, 2**n-1, 2**(n-1))):
        sector = int(index).bit_count()
        local = int(np.flatnonzero(channel.sectors[sector] == index)[0])
        blocks[sector][state,local,local] = 1
    steps = int(np.floor(t_final/dt + 1e-10))
    stride = max(1, round(sample_interval/dt))
    sampled_steps = np.unique(np.r_[np.arange(0, steps+1, stride), steps])
    times = sampled_steps*dt
    exact = exact_populations(n, times)
    measured = np.empty((3, len(times), n))
    sample = 0
    for step in range(steps+1):
        if step == sampled_steps[sample]:
            measured[:,sample] = channel.expectations(blocks)
            if sample < len(times)-1:
                sample += 1
        if step < steps:
            blocks = channel.apply(blocks)
    errors = abs(measured-exact)
    state, moment, site = np.unravel_index(np.argmax(errors), errors.shape)
    stationary, residual = channel.stationary(dt)
    target = model.xx_stationary_targets(n)["populations"]
    return dict(n=n, method=formula, dt=dt, t_final=t_final,
                sample_interval_requested=sample_interval, sampled_step_stride=stride,
                maximum_population_error=float(errors.max()),
                worst_initial_state=("0"*n,"1"*n,"1"+"0"*(n-1))[state],
                worst_time=float(times[moment]), worst_site=int(site+1),
                stationary_population_error=float(np.max(abs(stationary-target))),
                stationary_populations=stationary.tolist(), stationary_map_residual=residual,
                elapsed_seconds=time.perf_counter()-started)


def stationary_threshold(n, formula, tolerance=0.001):
    target = model.xx_stationary_targets(n)["populations"]
    def error(dt):
        populations, _ = BlockChannel(n, dt, formula).stationary(dt)
        return float(np.max(abs(populations-target)))
    dt = brentq(lambda dt: error(dt)-tolerance, 0.001, 0.05, xtol=1e-8)
    return dict(n=n, method=formula, stationary_dt_threshold=dt,
                stationary_population_error=error(dt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[3,7])
    parser.add_argument("--dt", type=float, nargs="+", default=[0.01,0.005,0.0025])
    parser.add_argument("--methods", nargs="+", choices=("lie","strang"), default=["lie","strang"])
    parser.add_argument("--t-final", type=float, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.02,
                        help="set to zero to inspect every physical step")
    parser.add_argument("--stationary-thresholds", action="store_true",
                        help="find stationary population-error thresholds instead of evolving")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"population_steps.json")
    args = parser.parse_args()
    rows=[]
    for n in args.n:
        for method in args.methods:
            if args.stationary_thresholds:
                row = stationary_threshold(n, method)
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(rows,indent=2)+"\n")
                continue
            for dt in args.dt:
                row=measure(n,dt,method,args.t_final,args.sample_interval)
                rows.append(row)
                print(json.dumps(row),flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()
