"""Asynchronous interpretation of the abstract runtime effect."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..execute import RuntimeFailure
from ..result import Err, Ok, Result


A = TypeVar("A")
B = TypeVar("B")


async def _pure(value: A) -> Result[A, RuntimeFailure]:
    return Ok(value)


def pure(value: A) -> Awaitable[Result[A, RuntimeFailure]]:
    return _pure(value)


async def _fail(error: RuntimeFailure) -> Result[A, RuntimeFailure]:
    return Err(error)


def fail(error: RuntimeFailure) -> Awaitable[Result[A, RuntimeFailure]]:
    return _fail(error)


async def bind(
    effect: Awaitable[Result[A, RuntimeFailure]],
    continuation: Callable[[A], Awaitable[Result[B, RuntimeFailure]]],
) -> Result[B, RuntimeFailure]:
    match await effect:
        case Err(error):
            return Err(error)
        case Ok(value):
            return await continuation(value)


async def _defer(
    thunk: Callable[[], Result[A, RuntimeFailure]],
) -> Result[A, RuntimeFailure]:
    # Qiskit's transpiler must run on the interpreter's calling thread on all
    # supported stacks. Long job execution is handled by polling below.
    return thunk()


def defer(thunk: Callable[[], Result[A, RuntimeFailure]]) -> Awaitable[Result[A, RuntimeFailure]]:
    return _defer(thunk)


async def _blocking(
    thunk: Callable[[], Result[A, RuntimeFailure]],
) -> Result[A, RuntimeFailure]:
    return await asyncio.to_thread(thunk)


def blocking(
    thunk: Callable[[], Result[A, RuntimeFailure]],
) -> Awaitable[Result[A, RuntimeFailure]]:
    return _blocking(thunk)


async def _poll_once(
    outcome: asyncio.Future[Result[A, RuntimeFailure]],
    is_finished: Callable[[], Result[bool, RuntimeFailure]],
    thunk: Callable[[], Result[A, RuntimeFailure]],
    poll_interval: float,
    poll_in_thread: bool,
) -> None:
    if outcome.done():
        return

    status = await asyncio.to_thread(is_finished) if poll_in_thread else is_finished()
    match status:
        case Err(error):
            outcome.set_result(Err(error))
        case Ok(True):
            outcome.set_result(thunk())
        case Ok(False):
            asyncio.get_running_loop().call_later(
                poll_interval,
                _schedule_poll,
                outcome,
                is_finished,
                thunk,
                poll_interval,
                poll_in_thread,
            )


def _schedule_poll(
    outcome: asyncio.Future[Result[A, RuntimeFailure]],
    is_finished: Callable[[], Result[bool, RuntimeFailure]],
    thunk: Callable[[], Result[A, RuntimeFailure]],
    poll_interval: float,
    poll_in_thread: bool,
) -> None:
    asyncio.create_task(
        _poll_once(
            outcome,
            is_finished,
            thunk,
            poll_interval,
            poll_in_thread,
        )
    )


async def _wait_until(
    is_finished: Callable[[], Result[bool, RuntimeFailure]],
    thunk: Callable[[], Result[A, RuntimeFailure]],
    poll_interval: float,
    poll_in_thread: bool,
) -> Result[A, RuntimeFailure]:
    outcome = asyncio.get_running_loop().create_future()
    _schedule_poll(
        outcome,
        is_finished,
        thunk,
        poll_interval,
        poll_in_thread,
    )
    return await outcome


def wait_until(
    is_finished: Callable[[], Result[bool, RuntimeFailure]],
    thunk: Callable[[], Result[A, RuntimeFailure]],
    *,
    poll_interval: float,
    poll_in_thread: bool,
) -> Awaitable[Result[A, RuntimeFailure]]:
    return _wait_until(is_finished, thunk, poll_interval, poll_in_thread)
