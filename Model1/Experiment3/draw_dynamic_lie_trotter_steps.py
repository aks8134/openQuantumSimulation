"""Draw labeled dynamic Lie steps only; never simulate or submit a job.

Display wires are ordered by model site (system_0,...,system_{N-1}),
followed by the two shared-ancilla encoding qubits. This is a relabeling
and wire-order change for presentation, not a change of the physical map.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IBMRuntime.interpreters.qiskit import _to_qiskit
from Model1.Experiment3 import dynamic_lie_trotter as dynamic


def build_display_circuit(number_of_qubits=7, *, steps=3, dt=0.2):
    """Repeat the existing one-step map, retaining its measurement/if/reset.

    Original q_(N-1-i) is displayed as system_i. a_L and a_H retain their
    encoding roles; the classical measurement bit is overwritten each step.
    Step barriers are visual separators, not new evolution operations.
    """
    if (not isinstance(number_of_qubits, int) or number_of_qubits < 2
            or not isinstance(steps, int) or steps < 1
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("require integer N>=2, integer steps>=1, and finite dt>0")
    system = QuantumRegister(number_of_qubits, "system")
    low = QuantumRegister(1, "ancilla_L")
    high = QuantumRegister(1, "ancilla_H")
    measured = ClassicalRegister(1, "m")
    circuit = QuantumCircuit(system, low, high, measured,
                             name=f"dynamic_lie_{steps}_steps_N{number_of_qubits}")
    jump_angle = 2 * np.sqrt(dynamic.gamma * dt)
    native_step = _to_qiskit(dynamic.build_one_step_circuit(number_of_qubits, dt, jump_angle))
    presentation_wires = [*reversed(system), low[0], high[0]]
    for step in range(1, steps + 1):
        circuit.barrier(*circuit.qubits, label=f"Step {step}")
        circuit.compose(native_step, qubits=presentation_wires, clbits=list(measured), inplace=True)
    return circuit


def draw_steps(number_of_qubits=7, *, steps=3, dt=0.2, output_stem=None):
    circuit = build_display_circuit(number_of_qubits, steps=steps, dt=dt)
    stem = (Path(output_stem) if output_stem is not None else
        Path(__file__).resolve().parent / "figures" /
        f"dynamic_lie_trotter_{steps}_steps_N{number_of_qubits}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure = circuit.draw(output="mpl", fold=-1, idle_wires=True,
                          cregbundle=False, style={"fontsize": 11, "subfontsize": 8})
    axes = figure.axes[0]
    center = sum(axes.get_xlim()) / 2
    axes.text(center, 1.15,
        f"Dynamic Lie-Trotter: {steps} steps, {number_of_qubits} system qubits + 2 ancillas",
        ha="center", va="bottom", fontsize=15)
    axes.text(center, -number_of_qubits - 3,
        f"system[i] = model site i = original q[{number_of_qubits-1}-i]; "
        "H_even/odd[i,j] acts on system[i], system[j].\n"
        f"ancilla_L = low encoding qubit; ancilla_H = high encoding qubit; "
        f"m is reused; K_2 acts only if m=0.  dt={dt:g}, J={dynamic.J:g}, gamma={dynamic.gamma:g}.",
        ha="center", va="top", fontsize=11)
    try:
        paths = tuple(stem.with_suffix(suffix) for suffix in (".png", ".pdf"))
        for path in paths:
            figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        return paths
    finally:
        plt.close(figure)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=7)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output-stem", type=Path, help="output path without extension; writes PNG and PDF")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    options = parse_arguments()
    for path in draw_steps(options.n_qubits, steps=options.steps, dt=options.dt,
                           output_stem=options.output_stem):
        print(f"Saved {path}")
