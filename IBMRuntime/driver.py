"""Generic execution entry points; experiment programs belong to consumers."""

from .api import run_async, run_sync


__all__ = ("run_async", "run_sync")
