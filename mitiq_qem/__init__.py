"""OCaml-style immutable facade for selected Mitiq QEM techniques."""

from ._dependency import MitiqUnavailable
from .core import (
    AveragedResult,
    CircuitBatch,
    CircuitVariant,
    Stage,
    average,
    circuit_batch,
)
from .dynamical_decoupling import (
    DDDPlan,
    combine_ddd,
    construct_ddd,
    ddd_plan,
)
from .pauli_twirling import (
    PauliTwirlingPlan,
    combine_pauli_twirling,
    construct_pauli_twirling,
    pauli_twirling_plan,
)
from .readout import (
    ReadoutPlan,
    ReadoutProbability,
    ReadoutResult,
    diagonal_expectation,
    mitigate_counts,
    mitigate_measurements,
    readout_plan,
    uncorrelated_readout_plan,
)
from .zne import (
    ZNEInference,
    ZNEPlan,
    ZNEResult,
    ZNEScaling,
    combine_zne,
    construct_zne,
    inference,
    scaling,
    zne_plan,
)


__all__ = (
    "AveragedResult",
    "CircuitBatch",
    "CircuitVariant",
    "DDDPlan",
    "MitiqUnavailable",
    "PauliTwirlingPlan",
    "ReadoutPlan",
    "ReadoutProbability",
    "ReadoutResult",
    "Stage",
    "ZNEInference",
    "ZNEPlan",
    "ZNEResult",
    "ZNEScaling",
    "average",
    "circuit_batch",
    "combine_ddd",
    "combine_pauli_twirling",
    "combine_zne",
    "construct_ddd",
    "construct_pauli_twirling",
    "construct_zne",
    "ddd_plan",
    "diagonal_expectation",
    "inference",
    "mitigate_counts",
    "mitigate_measurements",
    "pauli_twirling_plan",
    "readout_plan",
    "scaling",
    "uncorrelated_readout_plan",
    "zne_plan",
)
