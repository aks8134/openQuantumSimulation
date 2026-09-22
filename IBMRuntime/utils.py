"""Explicit file-system adapters for runtime configuration."""

import json
from dataclasses import dataclass
from pathlib import Path

from .result import Err, Ok, Result
from .runtime import IBMAccount


@dataclass(frozen=True, slots=True)
class AccountLoadFailure:
    message: str


def load_ibm_account(path: str | Path) -> Result[IBMAccount, AccountLoadFailure]:
    """Load an IBM account without printing or globally saving its secret."""

    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return Err(AccountLoadFailure("the account file must contain a JSON object"))

        api_key = data.get("apikey", data.get("api_key"))
        if not isinstance(api_key, str) or not api_key:
            return Err(AccountLoadFailure("the account file has no API key"))

        instance = data.get("crn", data.get("instance"))
        if instance is not None and not isinstance(instance, str):
            return Err(AccountLoadFailure("the account instance must be a string or null"))

        channel = data.get("channel", "ibm_quantum_platform")
        if not isinstance(channel, str) or not channel:
            return Err(AccountLoadFailure("the account channel must be a non-empty string"))

        return Ok(IBMAccount(api_key=api_key, instance=instance, channel=channel))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exception:
        return Err(AccountLoadFailure(str(exception)))
