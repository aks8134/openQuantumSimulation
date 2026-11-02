"""Verify saved Model2 schedules with explicit weighted branch matrices.

Reads a study, reconstructs every selected mean step independently of the
search/shared channel implementation, and exports portable schedules/plots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm


def explicit_step(state, dt, config):
    n = config["n_qubits"]
    angle = 2*config["exchange_prefactor"]*config["J"]*dt
    if config["system_formula"] == "exact":
        generator = np.diag(np.full(n-1, -angle), 1)+np.diag(np.full(n-1, angle), -1)
        unitary = expm(generator)
    else:
        unitary = np.eye(n)
        for parity in (0, 1):
            layer = np.eye(n)
            for j in range(parity, n-1, 2):
                layer[j:j+2, j:j+2] = [[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]]
            unitary = layer@unitary
    p = config["p"]
    probabilities = np.array([p, 1-p, p, 1-p])/2
    cosine = np.cos(np.sqrt(2*config["gamma"]*dt))
    average = np.zeros_like(state)
    for index, site in enumerate([0, 0, n-1, n-1]):
        mask = np.eye(n)
        mask[site, site] = cosine
        injection = np.zeros((n, n))
        if index % 2 == 0:
            injection[site, site] = 1-cosine*cosine
        if config["order"] == "jump-first":
            matrix, offset = unitary@mask, unitary@injection@unitary.T
        else:
            matrix, offset = mask@unitary, injection
        average += probabilities[index]*(matrix@state@matrix.T+offset)
    return average


def distance(state, config):
    delta = state-config["p"]*np.eye(config["n_qubits"])
    return float(max(np.max(np.abs(np.diag(delta))),
                     4*config["exchange_prefactor"]*np.max(np.abs(np.diag(delta, 1)))))


def permanent_order_check(state, dt, config):
    """A stronger permanent local bound when the signed deviation decreases.

    For B>=0 and B-M(B)>=0, positivity gives 0<=M^(k+1)(B)<=M^k(B).
    If 4c<=1, every local current/J is bounded by the largest diagonal of B.
    Strict positive eigenvalue margins avoid treating roundoff as positivity.
    """
    deviation = state-config["p"]*np.eye(config["n_qubits"])
    sign = 1 if np.trace(deviation) >= 0 else -1
    positive = sign*deviation
    decrease = sign*(state-explicit_step(state, dt, config))
    minimum = float(np.linalg.eigvalsh(positive).min())
    decrease_minimum = float(np.linalg.eigvalsh(decrease).min())
    established = bool(config["exchange_prefactor"] <= .25 and minimum > 1e-10 and decrease_minimum > 1e-10)
    bound = float(np.max(np.diag(positive))) if established else None
    return {"positive_deviation_minimum_eigenvalue": minimum,
            "one_step_decrease_minimum_eigenvalue": decrease_minimum,
            "matrix_order_established": established, "permanent_local_error_bound": bound,
            "permanent_mean_target_met": bool(established and bound <= config["mean_tolerance"]),
            "scope": "floating-point PSD-order checks, same final timestep, ideal mean nodes only"}


def main(directory):
    study = json.loads((directory/"study.json").read_text())
    config = study["configuration"]
    n = config["n_qubits"]
    initial = {"half_mixed": .5*np.eye(n), "vacuum": np.zeros((n, n)),
               "filled": np.eye(n), "left_excitation": np.diag([1.]+[0.]*(n-1))}[config["preparation"]]
    report = {"numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "study_sha256": hashlib.sha256((directory/"study.json").read_bytes()).hexdigest(),
              "families": {}}
    paths = {}
    for name, result in study["families"].items():
        selected = result["selected"]
        if selected is None:
            continue
        state = initial.copy()
        states = [state.copy()]
        for dt in selected["step_dts"]:
            state = explicit_step(state, dt, config)
            states.append(state.copy())
        discrepancy = float(np.max(np.abs(state-np.asarray(selected["output_covariance"]))))
        permanent = permanent_order_check(state, selected["step_dts"][-1], config)
        errors = [distance(state, config)]
        for _ in range(selected["forward"]["steps"]):
            state = explicit_step(state, selected["step_dts"][-1], config)
            errors.append(distance(state, config))
        forward_difference = float(np.max(np.abs(np.asarray(errors)-selected["forward"]["errors"])))
        row = {"preparation_steps": selected["steps"], "covariance_agreement": discrepancy,
               "permanent_order_check": permanent,
               "forward_error_agreement": forward_difference, "forward_maximum": max(errors),
               "passed": bool(discrepancy < 1e-11 and forward_difference < 1e-11
                              and max(errors) <= config["mean_tolerance"])}
        report["families"][name] = row
        if not row["passed"]:
            raise ValueError(f"independent verification failed for {name}: {row}")
        with (directory/f"{name}_schedule.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["step", "dt", "physical_time"])
            for index, (dt, t) in enumerate(zip(selected["step_dts"], selected["time_nodes"][1:]), 1):
                writer.writerow([index, dt, t])
        paths[name] = np.asarray(states)
    (directory/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Independent saved-schedule verification", "",
             "Every preparation and forward node was reconstructed as an explicit weighted sum of four affine branch matrices, independently of the search/shared channel implementation. This checks ideal numerical means; hardware and global minimality are outside its scope.", "",
             "| Family | Steps | Output covariance agreement | Forward error agreement | Permanent order check | Passed |",
             "|---|---:|---:|---:|---|---|"]
    for name, row in report["families"].items():
        lines.append(f"| {name} | {row['preparation_steps']} | {row['covariance_agreement']:.3g} | {row['forward_error_agreement']:.3g} | {row['permanent_order_check']['permanent_mean_target_met']} | {row['passed']} |")
    lines += ["", "The additional permanent order check tests a signed covariance deviation B >= 0 and B-M(B) >= 0 with eigenvalue margins above 1e-10. Positivity then implies 0 <= M^(k+1)(B) <= M^k(B). For Model2's prefactor 1/4, every population/current error is bounded by the largest diagonal of B, so passing establishes the mean allowance at all later constant-final-step nodes. This can succeed when the coarser spectral-norm bound in Study.md does not. It is a floating-point matrix-order check, not an interval-arithmetic certificate, finite-sampling guarantee or full-state bound."]
    (directory/"Verified.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for name, states in paths.items():
        times = study["families"][name]["selected"]["time_nodes"]
        for site in sorted(set([0, n//2, n-1])):
            axes[0].plot(times, states[:, site, site], label=f"{name}, site {site}")
        for bond in sorted(set([0, (n-2)//2, n-2])):
            axes[1].plot(times, 4*config["exchange_prefactor"]*states[:, bond, bond+1],
                         label=f"{name}, bond {bond}")
    axes[0].axhline(config["p"], linestyle=":", color="black", label="Target")
    axes[1].axhline(0., linestyle=":", color="black", label="Target")
    axes[0].set(ylabel="Population")
    axes[1].set(xlabel="Physical time", ylabel="Excitation current / J")
    for axis in axes:
        axis.legend(fontsize=7, ncol=2)
        axis.grid(alpha=.2)
    fig.savefig(directory/"observables.png", dpi=160)
    fig.savefig(directory/"observables.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
