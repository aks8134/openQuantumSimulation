"""Topology, physical mapping table, and labeled dynamic-step circuit.

This program only reads backend metadata and transpiles locally. It does
not create a Sampler/Estimator, run a simulator, or submit a hardware job.
The lower panel is the compact pre-transpilation circuit; the upper panel
is its initial logical-to-physical placement chosen by the transpiler.
"""

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IBMRuntime import (
    Circuit, CompilerConfig, Err, FakeIBMBackend, IBMHardware, Ok,
    RuntimeEnvironment, draw_transpiled_circuit_layout_sync, load_ibm_account,
)
from Model1.Experiment3 import dynamic_lie_trotter as dynamic
from Model1.Experiment3.draw_dynamic_lie_trotter_steps import build_display_circuit


def build_layout_source(number_of_qubits, steps, dt):
    """Same existing dynamic map as the diagram; no new evolution method."""
    if number_of_qubits < 2 or steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("require N>=2, steps>=1, and finite dt>0")
    single = dynamic.build_one_step_circuit(
        number_of_qubits, dt, 2 * np.sqrt(dynamic.gamma * dt))
    return Circuit(single.qubit_count, single.bit_count, single.operations * steps,
                   name=f"dynamic_lie_{steps}_steps_N{number_of_qubits}")


def logical_roles(number_of_qubits):
    return (
        *(f"system[{number_of_qubits-1-q}]" for q in range(number_of_qubits)),
        "ancilla_L (low)", "ancilla_H (high)",
    )


def resolve_target(backend, account_file):
    if backend in ("fake_fez", "fake_kingston", "fake_miami"):
        return FakeIBMBackend(backend), RuntimeEnvironment()
    if backend == "aer" or backend.startswith("fake_"):
        raise ValueError("choose an IBM backend or fake_fez/fake_kingston/fake_miami for a device topology")
    match load_ibm_account(account_file):
        case Err(error):
            raise RuntimeError(f"Could not load IBM account: {error.message}")
        case Ok(account):
            return IBMHardware(backend), RuntimeEnvironment(ibm_account=account)


def draw_layout(options):
    source = build_layout_source(options.n_qubits, options.steps, options.dt)
    target, environment = resolve_target(options.backend, options.account_file)
    print(f"Reading {options.backend} metadata and transpiling {options.steps} steps "
          f"for {options.n_qubits}+2 qubits. No job will be submitted.", flush=True)
    result = draw_transpiled_circuit_layout_sync(
        source, target,
        CompilerConfig(optimization_level=options.optimization_level,
                       seed_transpiler=options.seed_transpiler),
        environment, view="physical", logical_labels=logical_roles(options.n_qubits))
    match result:
        case Err(error):
            raise RuntimeError(f"Transpilation/layout failed: {error}")
        case Ok(figure):
            pass

    try:
        # Reuse the existing Matplotlib physical-index renderer (including
        # its Windows-compatible node labels), rather than Graphviz text.
        figure.set_size_inches(22, 18)
        if len(figure.axes) < 2:
            raise RuntimeError("layout renderer did not return topology and mapping-table axes")
        figure.axes[0].set_position((.025, .49, .59, .45))
        figure.axes[1].set_position((.65, .52, .325, .39))
        figure.axes[1].set_title("Initial logical-to-physical mapping", fontsize=14, pad=14)
        for table in figure.axes[1].tables:
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.4)

        # Presentation-only named registers reverse the original system
        # wire display, so system[i] matches H_even/odd[i,j] directly.
        display = build_display_circuit(options.n_qubits, steps=options.steps, dt=options.dt)
        step_figure = display.draw(output="mpl", fold=-1, idle_wires=True,
                                  cregbundle=False, style={"fontsize": 11, "subfontsize": 8})
        try:
            with BytesIO() as image_buffer:
                step_figure.savefig(image_buffer, format="png", dpi=220, bbox_inches="tight")
                image_buffer.seek(0)
                circuit_image = plt.imread(image_buffer, format="png")
        finally:
            plt.close(step_figure)
        panel = figure.add_axes((.025, .065, .95, .35))
        panel.imshow(circuit_image)
        panel.axis("off")
        panel.set_title(f"{options.steps} dynamic Lie-Trotter steps before transpilation "
                        f"(dt={options.dt:g})", fontsize=15, pad=15)
        figure.suptitle(f"Dynamic Lie-Trotter layout on {options.backend}: "
                       f"{options.n_qubits} system qubits + 2 ancillas", fontsize=19, y=.985)
        figure.text(.5, .025,
            f"system[i] = model site i = original logical q[{options.n_qubits-1}-i].  "
            "ancilla_L/H are the low/high shared-ancilla encoding qubits.\n"
            "Black device nodes show physical indices. Table gives initial placement; "
            "routing SWAPs can change placement later.\n"
            "Lower panel is the matching compact circuit, not its decomposed hardware gate sequence.",
            ha="center", va="center", fontsize=12)
        safe_backend = re.sub(r"[^A-Za-z0-9_.-]+", "_", options.backend)
        stem = options.output_stem or (Path(__file__).resolve().parent / "figures" /
            f"dynamic_lie_trotter_{safe_backend}_N{options.n_qubits}_{options.steps}_steps_transpiled_layout")
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths = tuple(stem.with_suffix(suffix) for suffix in (".png", ".pdf"))
        for path in paths:
            figure.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
            print(f"Saved {path}", flush=True)
        return paths
    finally:
        plt.close(figure)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=.2)
    parser.add_argument("--backend", default="ibm_phoenix")
    parser.add_argument("--account-file", type=Path, default=ROOT / "IBMRuntime" / "apikey.json")
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--seed-transpiler", type=int, default=11)
    parser.add_argument("--output-stem", type=Path, help="output filename without extension; writes PNG and PDF")
    options = parser.parse_args(arguments)
    options.backend = options.backend.lower()
    if options.n_qubits < 2 or options.steps < 1 or not np.isfinite(options.dt) or options.dt <= 0:
        parser.error("require N>=2, steps>=1, and finite dt>0")
    if options.seed_transpiler < 0:
        parser.error("seed-transpiler must be nonnegative")
    return options


if __name__ == "__main__":
    draw_layout(parse_arguments())
