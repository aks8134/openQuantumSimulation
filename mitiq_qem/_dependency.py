"""Lazy access to the optional upstream Mitiq dependency."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


class MitiqUnavailable(ImportError):
    """Raised when a transformation is requested without Mitiq installed."""


def require(module: str = "mitiq") -> ModuleType:
    try:
        return import_module(module)
    except ImportError as error:
        raise MitiqUnavailable(
            "Mitiq is required for circuit transformations. Install the "
            "Qiskit frontend with: uv pip install 'mitiq[qiskit]'"
        ) from error
