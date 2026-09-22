"""Synchronous interpretation of the abstract runtime effect."""

from collections.abc import Callable
from typing import TypeVar

from ..execute import RuntimeFailure
from ..result import Err, Ok, Result


A = TypeVar("A")
B = TypeVar("B")


def pure(value: A) -> Result[A, RuntimeFailure]:
    return Ok(value)


def fail(error: RuntimeFailure) -> Result[A, RuntimeFailure]:
    return Err(error)


def bind(
    effect: Result[A, RuntimeFailure],
    continuation: Callable[[A], Result[B, RuntimeFailure]],
) -> Result[B, RuntimeFailure]:
    match effect:
        case Err(error):
            return Err(error)
        case Ok(value):
            return continuation(value)


def defer(thunk: Callable[[], Result[A, RuntimeFailure]]) -> Result[A, RuntimeFailure]:
    return thunk()


def blocking(thunk: Callable[[], Result[A, RuntimeFailure]]) -> Result[A, RuntimeFailure]:
    return thunk()


def wait_until(
    is_finished: Callable[[], Result[bool, RuntimeFailure]],
    thunk: Callable[[], Result[A, RuntimeFailure]],
    *,
    poll_interval: float,
    poll_in_thread: bool,
) -> Result[A, RuntimeFailure]:
    del is_finished, poll_interval, poll_in_thread
    return thunk()
