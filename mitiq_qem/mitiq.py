"""Effectful boundary to the optional upstream :mod:`mitiq` package."""

from __future__ import annotations

from random import getstate, seed as seed_random, setstate
from typing import Any

import numpy as np

from ._dependency import require
from .dynamical_decoupling import DDDRule
from .qiskit import scale_unitary_regions, validate_mitiq_circuit
from .readout import ReadoutPlan
from .zne import ZNEScaling


def _is_qiskit(value: Any) -> bool:
    return value.__class__.__module__.startswith("qiskit")


def _scaler(specification: ZNEScaling):
    scaling = require("mitiq.zne.scaling")
    return {
        "random_local": scaling.fold_gates_at_random,
        "fold_all": scaling.fold_all,
        "global": scaling.fold_global,
    }[specification.method]


def scale_noise(
    circuit: Any,
    scale_factor: float,
    random_seed: int | None,
    *,
    scaling: ZNEScaling,
) -> Any:
    """Apply one Mitiq scaling operation with Qiskit boundary protection."""

    scaler = _scaler(scaling)
    features = validate_mitiq_circuit(circuit) if _is_qiskit(circuit) else None
    if scaling.method == "global" and features is not None and features.resets:
        raise ValueError("global ZNE folding is invalid across reset operations")

    def apply(value: Any, factor: float) -> Any:
        if scaling.method == "random_local":
            return scaler(
                value,
                factor,
                seed=random_seed,
                fidelities=dict(scaling.fidelities),
            )
        return scaler(value, factor)

    return (
        scale_unitary_regions(circuit, scale_factor, apply)
        if features is not None
        and features.resets
        and scaling.method != "global"
        else apply(circuit, scale_factor)
    )


def pauli_twirl(
    circuit: Any,
    variants: int,
    random_seed: int | None,
) -> tuple[Any, ...]:
    """Generate Mitiq CNOT/CZ Pauli-twirled variants reproducibly."""

    if _is_qiskit(circuit):
        validate_mitiq_circuit(circuit)
    function = require("mitiq.pt").generate_pauli_twirl_variants
    state = getstate()
    if random_seed is not None:
        seed_random(random_seed)
    try:
        generated = function(
            circuit,
            num_circuits=variants,
            random_state=random_seed,
        )
    finally:
        setstate(state)
    return tuple(generated)


def insert_ddd(
    circuit: Any,
    rule: DDDRule,
    spacing: int,
    trials: int,
) -> tuple[Any, ...]:
    """Insert one of Mitiq's standard XX, YY, or XYXY sequences."""

    if _is_qiskit(circuit):
        validate_mitiq_circuit(circuit)
    ddd = require("mitiq.ddd")
    selected_rule = getattr(ddd.rules, rule)
    generated = ddd.construct_circuits(
        circuit,
        rule=selected_rule,
        rule_args={"spacing": spacing},
        num_trials=trials,
    )
    return tuple(generated)


def mitigate_measurement_result(
    measurement_result: Any,
    plan: ReadoutPlan,
) -> Any:
    inverse_module = require("mitiq.rem.inverse_confusion_matrix")
    return inverse_module.mitigate_measurements(
        measurement_result,
        np.asarray(plan.inverse_confusion_matrix, dtype=float),
    )
