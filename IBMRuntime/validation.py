"""Small composable helpers for pure validation functions."""


def issue_if(condition: bool, message: str) -> tuple[str, ...]:
    """Return one issue when ``condition`` holds, otherwise no issues."""

    return (message,) if condition else ()
