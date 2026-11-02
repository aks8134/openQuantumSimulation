"""Portable, frozen Model2 schedule and reproducible finite-path predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from library import reservoir_scaling as physics

DEFAULT_SPEC = Path(__file__).parent/"inputs/cosine44.json"
SPEC_SHA256 = "a718135fd2db8bccd5973d2f476d15030dd3064a1e8fd870db1a012b59d9c2bd"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_spec(path=DEFAULT_SPEC):
    data = json.loads(Path(path).read_text())
    if digest(data) != SPEC_SHA256:
        raise ValueError("this experiment requires the frozen validated cosine44 specification")
    p, schedule = data["physics"], data["schedule"]
    if (p != {"n_qubits": 10, "J": 1., "gamma": 2., "p_left": .25, "p_right": .25,
              "exchange_prefactor": .25, "delta": 0.} or schedule["steps"] != 44
            or data["preparation"]["populations"] != [.5]*10
            or data["channel"] != {"order": "jump-first", "system_formula": "even-odd", "sampling_policy": "rate-weighted"}):
        raise ValueError("frozen Model2 physics or preparation changed")
    h = schedule["final_dt"]+.5*(schedule["initial_dt"]-schedule["final_dt"])*(1+np.cos(
        np.pi*(np.arange(44)/43)**schedule["power"]))
    if not np.allclose(h, schedule["step_dts"], rtol=1e-14, atol=1e-14):
        raise ValueError("saved steps disagree with the frozen warped cosine")
    return data


def choices(seed, paths, steps, probabilities):
    streams = np.random.SeedSequence(seed).spawn(paths)
    uniforms = np.asarray([np.random.default_rng(stream).random(steps) for stream in streams])
    return np.searchsorted(np.cumsum(probabilities), uniforms, side="right").astype(np.uint8)


def make_source(spec, paths, seed):
    if paths < 2 or seed < 0:
        raise ValueError("need at least two paths and a nonnegative seed")
    p = spec["physics"]
    model = physics.Model(p["n_qubits"], p["gamma"]/p["J"], p["p_left"], p["p_right"], p["exchange_prefactor"])
    dts = np.asarray(spec["schedule"]["step_dts"])
    probabilities = np.array([p["p_left"], 1-p["p_left"], p["p_right"], 1-p["p_right"]])/2
    draws = choices(seed, paths, len(dts), probabilities)
    states = np.broadcast_to(np.eye(model.n)/2, (1, paths, model.n, model.n)).copy()
    mean = np.eye(model.n)/2
    for index, dt in enumerate(dts):
        branch = physics.channel(model, p["J"]*dt, "rate-weighted", "jump-first", "even-odd")
        states = physics.sampled_step(states, draws[:, index], branch)
        mean = physics.mean_step(mean, branch)
    covariances = states[0]
    sample = covariances.mean(axis=0)
    target = p["p_left"]*np.eye(model.n)
    distance = lambda a, b: max(physics.observable_error(a, b, model.prefactor).values())
    two_se = 2*covariances.std(axis=0, ddof=1)/np.sqrt(paths)
    errors = {"ideal_subset_target_error": distance(sample, target), "sampling_distance": distance(sample, mean),
              "maximum_two_standard_errors": distance(two_se, np.zeros_like(mean))}
    budget = spec["validation"]
    errors["empirical_classical_checks_passed"] = bool(errors["ideal_subset_target_error"] <= budget["total_tolerance"]
        and errors["sampling_distance"] <= budget["sampling_tolerance"]
        and errors["maximum_two_standard_errors"] <= budget["sampling_tolerance"])
    errors["scope"] = "This exact finite path ensemble, analytic initial mixture, no measurement shots or device noise; two SE descriptive. Full 16384-path validation is separate."
    observables = lambda C: {"populations": np.diag(C).tolist(), "excitation_flows": (4*model.prefactor*p["J"]*np.diag(C, 1)).tolist()}
    mean_observables = {k: [[v]] for k, v in observables(mean).items()}
    source = {"experiment": "Model2/Experiment5", "artifact_type": "model2_hardware_input_v1",
        "configuration": {"physics": p, "trajectories": paths, "seed": seed, **spec["channel"]},
        "preparation": spec["preparation"], "times": np.r_[0., np.cumsum(dts)].tolist(),
        "schedule": {"step_dts": dts.tolist(), "jump_probabilities": probabilities.tolist(), "jump_choices": draws.tolist()},
        "final_covariances_real_gauge": covariances.tolist(), "mean_channel_observables": mean_observables,
        "stationary_observables": observables(target), "finite_path_assessment": errors,
        "provenance": {"spec_sha256": digest(spec), "full_validation": spec["validation"],
                       "jump_choices_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}}
    if paths == budget["classical_trajectories"]:
        replay = next((r for r in budget["replays"] if r["seed"] == seed and r["trajectories"] == paths), None)
        if replay is not None and replay["choices_sha256"] != source["provenance"]["jump_choices_sha256"]:
            raise ValueError("regenerated paths differ from the validated replay")
    return source
