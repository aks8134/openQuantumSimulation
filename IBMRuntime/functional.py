"""OCaml-style immutable traversal combinators used by the wrapper."""

from collections.abc import Callable, Iterable
from functools import reduce
from itertools import accumulate
from typing import TypeVar, cast


A = TypeVar("A")
B = TypeVar("B")
Accumulator = TypeVar("Accumulator")


def fold_left(
    function: Callable[[Accumulator, A], Accumulator],
    initial: Accumulator,
    values: Iterable[A],
) -> Accumulator:
    return reduce(function, values, initial)


def map_tuple(function: Callable[[A], B], values: Iterable[A]) -> tuple[B, ...]:
    return tuple(map(function, values))


def scan_left(
    function: Callable[[Accumulator, A], Accumulator],
    initial: Accumulator,
    values: Iterable[A],
) -> tuple[Accumulator, ...]:
    """Return the initial value and every successive fold state."""

    return tuple(accumulate(values, function, initial=initial))


def filter_map(
    function: Callable[[A], B | None],
    values: Iterable[A],
) -> tuple[B, ...]:
    mapped = map(function, values)
    present = filter(lambda value: value is not None, mapped)
    return cast(tuple[B, ...], tuple(present))


def concat_map(
    function: Callable[[A], Iterable[B]],
    values: Iterable[A],
) -> tuple[B, ...]:
    return fold_left(
        lambda accumulated, current: (*accumulated, *current),
        (),
        map(function, values),
    )


def exists(predicate: Callable[[A], bool], values: Iterable[A]) -> bool:
    return any(map(predicate, values))
